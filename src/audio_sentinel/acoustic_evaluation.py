"""A3.4 reproducible clip-level evaluation for the pinned acoustic baseline."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import soundfile as sf

from audio_sentinel.acoustic_inference import expected_yamnet_patches
from audio_sentinel.acoustic_loader import LoadedAcousticModel
from audio_sentinel.acoustic_model import LABEL_MAPPING, LABEL_MAPPING_VERSION, YAMNET
from audio_sentinel.audio_transforms import convert_to_mono, resample_audio
from audio_sentinel.contracts import EventLabel


EVALUATION_SCHEMA_VERSION = "1.0"
SAMPLING_SEED = "audio-sentinel-a3.4-v1"
EVALUATED_LABELS = (EventLabel.SIREN, EventLabel.GLASS_BREAK, EventLabel.GUNSHOT)
MAX_AUDIO_BYTES = 16_777_216
MAX_DECODED_BYTES = 64_000_000
MAX_DURATION_SECONDS = 15.0
MAX_METADATA_BYTES = 8_388_608
MAX_LICENSE_BYTES = 8_388_608
MAX_SELECTED_SAMPLES = 10_000


class AcousticEvaluationError(RuntimeError):
    """A stable evaluation failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    metadata_path: str
    audio_directory: str
    license_paths: tuple[str, ...]
    calibration_folds: tuple[int, ...]
    holdout_folds: tuple[int, ...]
    calibration_per_category: int
    holdout_per_category: int
    positive_categories: Mapping[EventLabel, tuple[str, ...]]


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="esc50",
        metadata_path="esc50/meta/esc50.csv",
        audio_directory="esc50/audio",
        license_paths=("esc50/LICENSE",),
        calibration_folds=(1, 2, 3, 4),
        holdout_folds=(5,),
        calibration_per_category=8,
        holdout_per_category=4,
        positive_categories={
            EventLabel.SIREN: ("siren",),
            EventLabel.GLASS_BREAK: ("glass_breaking",),
        },
    ),
    DatasetSpec(
        dataset_id="urbansound8k",
        metadata_path="UrbanSound8K/metadata/UrbanSound8K.csv",
        audio_directory="UrbanSound8K/audio",
        license_paths=(
            "UrbanSound8K/UrbanSound8K_README.txt",
            "UrbanSound8K/FREESOUNDCREDITS.txt",
        ),
        calibration_folds=(1, 2, 3, 4, 5, 6, 7, 8),
        holdout_folds=(9, 10),
        calibration_per_category=8,
        holdout_per_category=4,
        positive_categories={
            EventLabel.SIREN: ("siren",),
            EventLabel.GUNSHOT: ("gun_shot",),
        },
    ),
)


@dataclass(frozen=True)
class LabeledSample:
    dataset_id: str
    split: Literal["calibration", "holdout"]
    fold: int
    category: str
    relative_audio_path: str


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None
    balanced_accuracy: float | None
    average_precision: float | None

    @property
    def sample_count(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @property
    def positive_count(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def negative_count(self) -> int:
        return self.false_positive + self.true_negative

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "sample_count": self.sample_count,
                "positive_count": self.positive_count, "negative_count": self.negative_count}


def _sha256(document: bytes) -> str:
    return hashlib.sha256(document).hexdigest()


def _safe_file(raw_root: Path, relative_path: str) -> Path:
    if not relative_path or "\\" in relative_path:
        raise AcousticEvaluationError("invalid_path", "Dataset paths must use nonempty POSIX-relative paths.")
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise AcousticEvaluationError("invalid_path", "Dataset paths must stay within data/raw.")
    try:
        root = raw_root.resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise AcousticEvaluationError("invalid_path", "A required dataset file is absent or outside data/raw.") from error
    if not path.is_file():
        raise AcousticEvaluationError("invalid_path", "A required dataset path is not a regular file.")
    return path


def _read_bounded(path: Path, maximum: int, description: str) -> bytes:
    try:
        size = path.stat().st_size
        if size <= 0 or size > maximum:
            raise AcousticEvaluationError("invalid_file_size", f"A required {description} file has an invalid size.")
        document = path.read_bytes()
    except AcousticEvaluationError:
        raise
    except OSError as error:
        raise AcousticEvaluationError("file_unreadable", f"A required {description} file could not be read.") from error
    if len(document) != size:
        raise AcousticEvaluationError("source_changed", f"A required {description} file changed while it was read.")
    return document


def _metadata_rows(raw_root: Path, spec: DatasetSpec) -> tuple[bytes, list[tuple[str, int, str]]]:
    path = _safe_file(raw_root, spec.metadata_path)
    try:
        document = _read_bounded(path, MAX_METADATA_BYTES, "metadata")
        reader = csv.DictReader(document.decode("utf-8-sig").splitlines())
        if spec.dataset_id == "esc50":
            required = {"filename", "fold", "category"}
            rows = [(row["filename"], int(row["fold"]), row["category"]) for row in reader]
        elif spec.dataset_id == "urbansound8k":
            required = {"slice_file_name", "fold", "class"}
            rows = [(row["slice_file_name"], int(row["fold"]), row["class"]) for row in reader]
        else:
            raise AcousticEvaluationError("invalid_dataset", "The evaluation plan contains an unknown dataset.")
        if not required <= set(reader.fieldnames or ()):
            raise AcousticEvaluationError("invalid_metadata", "Dataset metadata columns differ from the evaluation contract.")
    except AcousticEvaluationError:
        raise
    except (KeyError, UnicodeError, ValueError, csv.Error) as error:
        raise AcousticEvaluationError("invalid_metadata", "Dataset metadata could not be parsed.") from error
    if not rows or len({row[0] for row in rows}) != len(rows):
        raise AcousticEvaluationError("invalid_metadata", "Dataset metadata must contain unique nonempty samples.")
    return document, rows


def _rank(sample: tuple[str, int, str], dataset_id: str, split: str) -> str:
    filename, fold, category = sample
    value = f"{SAMPLING_SEED}|{dataset_id}|{split}|{category}|{fold}|{filename}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_labeled_samples(raw_root: Path, specs: Sequence[DatasetSpec] = DATASETS) -> tuple[
    tuple[LabeledSample, ...], tuple[dict[str, object], ...]
]:
    """Select a deterministic category-balanced calibration/holdout benchmark."""
    selected: list[LabeledSample] = []
    provenance: list[dict[str, object]] = []
    for spec in specs:
        document, rows = _metadata_rows(raw_root, spec)
        categories = sorted({row[2] for row in rows})
        mapped_categories = [
            category for label, values in spec.positive_categories.items()
            for category in values if label in EVALUATED_LABELS
        ]
        if (
            not set(spec.positive_categories) <= set(EVALUATED_LABELS)
            or len(mapped_categories) != len(set(mapped_categories))
            or not set(mapped_categories) <= set(categories)
        ):
            raise AcousticEvaluationError(
                "invalid_dataset", "Positive-category mappings must be unique and present in dataset metadata."
            )
        dataset_start = len(selected)
        split_counts: dict[str, int] = {}
        for split, folds, limit in (
            ("calibration", spec.calibration_folds, spec.calibration_per_category),
            ("holdout", spec.holdout_folds, spec.holdout_per_category),
        ):
            for category in categories:
                candidates = [row for row in rows if row[1] in folds and row[2] == category]
                if len(candidates) < limit:
                    raise AcousticEvaluationError(
                        "insufficient_samples",
                        f"{spec.dataset_id} lacks enough {category} samples in its {split} folds.",
                    )
                for filename, fold, _ in sorted(
                    candidates, key=lambda row: (_rank(row, spec.dataset_id, split), row[0])
                )[:limit]:
                    folder = f"fold{fold}" if spec.dataset_id == "urbansound8k" else ""
                    relative = "/".join(part for part in (spec.audio_directory, folder, filename) if part)
                    _safe_file(raw_root, relative)
                    selected.append(LabeledSample(spec.dataset_id, split, fold, category, relative))
                    split_counts[split] = split_counts.get(split, 0) + 1
                    if len(selected) > MAX_SELECTED_SAMPLES:
                        raise AcousticEvaluationError(
                            "too_many_samples", "The evaluation plan exceeds the selected-sample limit."
                        )
        licenses = []
        for relative in spec.license_paths:
            license_bytes = _read_bounded(
                _safe_file(raw_root, relative), MAX_LICENSE_BYTES, "license"
            )
            licenses.append({"path": relative, "sha256": _sha256(license_bytes)})
        provenance.append({
            "dataset_id": spec.dataset_id,
            "metadata_path": spec.metadata_path,
            "metadata_sha256": _sha256(document),
            "license_files": licenses,
            "calibration_folds": list(spec.calibration_folds),
            "holdout_folds": list(spec.holdout_folds),
            "per_category": {
                "calibration": spec.calibration_per_category,
                "holdout": spec.holdout_per_category,
            },
            "category_count": len(categories),
            "selected_counts": split_counts,
            "selected_total": len(selected) - dataset_start,
        })
    if len({sample.relative_audio_path for sample in selected}) != len(selected):
        raise AcousticEvaluationError("duplicate_sample", "A sample appears more than once in the evaluation plan.")
    return tuple(selected), tuple(provenance)


def _read_waveform(path: Path) -> tuple[NDArray[np.float32], dict[str, object]]:
    try:
        details = path.stat()
        if details.st_size <= 0 or details.st_size > MAX_AUDIO_BYTES:
            raise AcousticEvaluationError("invalid_audio_size", "An evaluation clip exceeds the file-size policy.")
        document = path.read_bytes()
        if len(document) != details.st_size or len(document) > MAX_AUDIO_BYTES:
            raise AcousticEvaluationError("source_changed", "An evaluation clip changed while it was read.")
        with sf.SoundFile(BytesIO(document)) as audio:
            if not 8_000 <= audio.samplerate <= 192_000 or not 1 <= audio.channels <= 8:
                raise AcousticEvaluationError("invalid_audio", "An evaluation clip has unsupported audio properties.")
            if audio.frames <= 0 or audio.frames / audio.samplerate > MAX_DURATION_SECONDS:
                raise AcousticEvaluationError("invalid_audio", "An evaluation clip has an invalid duration.")
            if audio.frames * audio.channels * 4 > MAX_DECODED_BYTES:
                raise AcousticEvaluationError("decoded_audio_too_large", "An evaluation clip exceeds the decode budget.")
            samples = audio.read(audio.frames, dtype="float32", always_2d=True)
            if samples.shape != (audio.frames, audio.channels) or len(audio.read(1, dtype="float32")):
                raise AcousticEvaluationError("invalid_audio", "An evaluation clip decoded to an unexpected length.")
            source_rate = int(audio.samplerate)
            channels = int(audio.channels)
            source_frames = int(audio.frames)
    except AcousticEvaluationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise AcousticEvaluationError("invalid_audio", "An evaluation clip could not be decoded.") from error
    if not np.isfinite(samples).all() or np.any(samples < -1) or np.any(samples > 1):
        raise AcousticEvaluationError("invalid_audio", "Evaluation samples must be finite and within [-1, 1].")
    try:
        mono = convert_to_mono(np.ascontiguousarray(samples, dtype=np.float32))
        resampled = resample_audio(
            mono, source_rate, YAMNET.sample_rate_hz, max_decoded_bytes=MAX_DECODED_BYTES
        )[:, 0]
    except (MemoryError, RuntimeError, ValueError) as error:
        raise AcousticEvaluationError("invalid_audio", "An evaluation clip could not be transformed.") from error
    waveform = np.ascontiguousarray(resampled, dtype=np.float32)
    return waveform, {
        "source_sha256": _sha256(document),
        "source_size_bytes": details.st_size,
        "source_sample_rate_hz": source_rate,
        "source_channels": channels,
        "source_num_frames": source_frames,
        "model_input_num_samples": len(waveform),
        "duration_seconds": source_frames / source_rate,
    }


def _score_waveform(model: object, waveform: NDArray[np.float32]) -> dict[str, float]:
    patches = expected_yamnet_patches(len(waveform))
    try:
        output = model(waveform)
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise ValueError("wrong output inventory")
        arrays = [np.asarray(item.numpy() if hasattr(item, "numpy") else item) for item in output]
    except Exception as error:
        raise AcousticEvaluationError("model_failed", "YAMNet failed on an evaluation clip.") from error
    scores, embeddings, spectrogram = arrays
    frames = YAMNET.patch_frames + (patches - 1) * (YAMNET.patch_frames // 2)
    try:
        valid = (
            scores.dtype == np.float32 and scores.shape == (patches, YAMNET.num_classes)
            and embeddings.dtype == np.float32 and embeddings.shape == (patches, 1024)
            and spectrogram.dtype == np.float32 and spectrogram.shape == (frames, YAMNET.mel_bands)
            and all(np.isfinite(item).all() for item in arrays)
            and not np.any(scores < 0) and not np.any(scores > 1)
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise AcousticEvaluationError("invalid_model_output", "YAMNet returned an invalid evaluation tensor.")
    return {
        label.value: float(np.max(scores[:, [item.index for item in classes]]))
        for label, classes in LABEL_MAPPING.items()
    }


def _truth(sample: LabeledSample, label: EventLabel, specs: Sequence[DatasetSpec]) -> bool:
    spec = next(item for item in specs if item.dataset_id == sample.dataset_id)
    return sample.category in spec.positive_categories.get(label, ())


def _divide(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def average_precision(scores: Sequence[float], truth: Sequence[bool]) -> float | None:
    if len(scores) != len(truth) or not scores:
        raise ValueError("scores and truth must be nonempty and aligned")
    positives = sum(truth)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, truth, strict=True), key=lambda item: item[0], reverse=True)
    hits = 0
    seen = 0
    total = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        end = index
        group_hits = 0
        while end < len(ordered) and ordered[end][0] == score:
            group_hits += int(ordered[end][1])
            end += 1
        hits += group_hits
        seen += end - index
        total += (group_hits / positives) * (hits / seen)
        index = end
    return total


def binary_metrics(scores: Sequence[float], truth: Sequence[bool], threshold: float) -> BinaryMetrics:
    if len(scores) != len(truth) or not scores:
        raise ValueError("scores and truth must be nonempty and aligned")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    predictions = [score >= threshold for score in scores]
    tp = sum(prediction and actual for prediction, actual in zip(predictions, truth, strict=True))
    fp = sum(prediction and not actual for prediction, actual in zip(predictions, truth, strict=True))
    fn = sum(not prediction and actual for prediction, actual in zip(predictions, truth, strict=True))
    tn = sum(not prediction and not actual for prediction, actual in zip(predictions, truth, strict=True))
    precision, recall = _divide(tp, tp + fp), _divide(tp, tp + fn)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)
    specificity = _divide(tn, tn + fp)
    balanced = None if recall is None or specificity is None else (recall + specificity) / 2
    return BinaryMetrics(threshold, tp, fp, fn, tn, precision, recall, f1, specificity, balanced,
                         average_precision(scores, truth))


def select_f1_threshold(scores: Sequence[float], truth: Sequence[bool]) -> BinaryMetrics:
    """Maximize calibration F1; then precision, recall, and finally threshold."""
    if not any(truth) or all(truth):
        raise ValueError("threshold selection needs both positive and negative samples")
    values = {0.0, 1.0, *(float(score) for score in scores)}
    maximum = max(scores)
    if maximum < 1:
        values.add(float(np.nextafter(np.float64(maximum), np.float64(1.0))))
    candidates = [binary_metrics(scores, truth, threshold) for threshold in sorted(values)]
    return max(candidates, key=lambda item: (
        -1.0 if item.f1 is None else item.f1,
        -1.0 if item.precision is None else item.precision,
        -1.0 if item.recall is None else item.recall,
        item.threshold,
    ))


def evaluate_labeled_samples(
    raw_root: Path,
    loaded: LoadedAcousticModel,
    *,
    specs: Sequence[DatasetSpec] = DATASETS,
    now: datetime | None = None,
) -> dict[str, object]:
    """Run the pinned model and return a self-contained JSON-ready report."""
    created_at = now if now is not None else datetime.now(UTC)
    if created_at.utcoffset() is None:
        raise AcousticEvaluationError("invalid_time", "Evaluation timestamps must include a timezone.")
    if (
        loaded.metadata.model_id != YAMNET.model_id
        or loaded.metadata.model_version != "1"
        or loaded.metadata.artifact_sha256 != YAMNET.artifact_sha256
        or loaded.metadata.class_map_sha256 != YAMNET.class_map_sha256
        or loaded.metadata.label_mapping_version != LABEL_MAPPING_VERSION
        or loaded.metadata.runtime_distribution != YAMNET.runtime_distribution
        or loaded.metadata.runtime_version != YAMNET.runtime_version
        or loaded.metadata.num_classes != YAMNET.num_classes
    ):
        raise AcousticEvaluationError("model_mismatch", "Evaluation requires the pinned YAMNet artifact.")
    samples, datasets = select_labeled_samples(raw_root, specs)
    results: list[dict[str, object]] = []
    for sample in samples:
        path = _safe_file(raw_root, sample.relative_audio_path)
        waveform, audio = _read_waveform(path)
        scores = _score_waveform(loaded.model, waveform)
        results.append({
            **asdict(sample),
            **audio,
            "truth": {label.value: _truth(sample, label, specs) for label in EVALUATED_LABELS},
            "scores": scores,
        })

    thresholds: dict[str, object] = {}
    for label in EVALUATED_LABELS:
        calibration = [row for row in results if row["split"] == "calibration"]
        calibration_scores = [float(row["scores"][label.value]) for row in calibration]  # type: ignore[index]
        calibration_truth = [bool(row["truth"][label.value]) for row in calibration]  # type: ignore[index]
        chosen = select_f1_threshold(calibration_scores, calibration_truth)
        holdout = [row for row in results if row["split"] == "holdout"]

        def metrics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
            return binary_metrics(
                [float(row["scores"][label.value]) for row in rows],  # type: ignore[index]
                [bool(row["truth"][label.value]) for row in rows],  # type: ignore[index]
                chosen.threshold,
            ).as_dict()

        thresholds[label.value] = {
            "threshold": chosen.threshold,
            "selection_objective": "maximum_clip_level_f1",
            "tie_break": "precision_then_recall_then_higher_threshold",
            "calibration": chosen.as_dict(),
            "holdout": metrics(holdout),
            "holdout_by_dataset": {
                spec.dataset_id: metrics([row for row in holdout if row["dataset_id"] == spec.dataset_id])
                for spec in specs
            },
        }

    coverage = {
        label.value: {
            "status": "evaluated" if label in EVALUATED_LABELS else "not_evaluated",
            "reason": None if label in EVALUATED_LABELS else {
                EventLabel.SPEECH_PRESENT: "No approved direct speech-presence category mapping in this benchmark.",
                EventLabel.SMOKE_ALARM: "No positive smoke-alarm category is present in the local benchmark.",
                EventLabel.EXPLOSION: "Fireworks are retained as confounders, not relabeled as explosions.",
            }[label],
        }
        for label in LABEL_MAPPING
    }
    report: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "created_at": created_at.astimezone(UTC).isoformat(),
        "purpose": "offline_labeled_baseline",
        "decision_status": "research_baseline_not_production_calibration",
        "model": loaded.metadata.as_dict(),
        "label_mapping_version": LABEL_MAPPING_VERSION,
        "sampling": {
            "seed": SAMPLING_SEED,
            "method": "sha256_rank_within_dataset_split_category",
            "calibration_and_holdout_are_disjoint": True,
            "clip_score": "maximum_over_yamnet_patches_and_classes_mapped_to_project_label",
        },
        "datasets": list(datasets),
        "coverage": coverage,
        "thresholds": thresholds,
        "sample_count": len(results),
        "samples": results,
        "limitations": [
            "Public environmental-sound clips do not represent deployment prevalence or microphone conditions.",
            "Single-label metadata can contain unannotated background sounds, so some negative labels may be noisy.",
            "Thresholds optimize clip-level F1 on a small deterministic calibration sample and require broader validation.",
            "Only siren, glass_break, and gunshot have direct positive categories in the local approved datasets.",
            "This evaluation does not validate event timing, incident decisions, risk scoring, or speech semantics.",
        ],
    }
    identity = {key: value for key, value in report.items() if key != "created_at"}
    semantic = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    report["evaluation_id"] = f"a3-4-{_sha256(semantic)[:24]}"
    return report


def save_evaluation_report(report: Mapping[str, object], destination: Path) -> None:
    """Write a report without overwriting an existing result."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(document)
    except FileExistsError as error:
        raise AcousticEvaluationError("output_exists", "Evaluation output already exists; it was not overwritten.") from error
