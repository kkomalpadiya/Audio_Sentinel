"""A3.4 evaluation-plan, metric, inference-boundary, and report tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_sentinel import acoustic_evaluation as evaluation
from audio_sentinel.acoustic_evaluation import (
    AcousticEvaluationError,
    DatasetSpec,
    average_precision,
    binary_metrics,
    evaluate_labeled_samples,
    save_evaluation_report,
    select_f1_threshold,
    select_labeled_samples,
)
from audio_sentinel.acoustic_loader import AcousticModelMetadata, LoadedAcousticModel, TensorContract
from audio_sentinel.acoustic_model import LABEL_MAPPING, LABEL_MAPPING_VERSION, YAMNET
from audio_sentinel.contracts import EventLabel


def _metadata() -> AcousticModelMetadata:
    return AcousticModelMetadata(
        model_id=YAMNET.model_id,
        model_version="1",
        model_handle=YAMNET.model_handle,
        model_path="yamnet/1",
        artifact_sha256=YAMNET.artifact_sha256,
        class_map_sha256=YAMNET.class_map_sha256,
        label_mapping_version=LABEL_MAPPING_VERSION,
        runtime_distribution=YAMNET.runtime_distribution,
        runtime_version=YAMNET.runtime_version,
        exported_with_tensorflow="2.3.0",
        exported_with_tensorflow_git="unknown",
        signature_name="serving_default",
        input=TensorContract("waveform", (None,), "float32"),
        outputs=(
            TensorContract("output_0", (None, 521), "float32"),
            TensorContract("output_1", (None, 1024), "float32"),
            TensorContract("output_2", (None, 64), "float32"),
        ),
        num_classes=YAMNET.num_classes,
    )


class FakeModel:
    def __call__(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        level = float(np.mean(np.abs(waveform)))
        scores = np.full((1, YAMNET.num_classes), 0.05, dtype=np.float32)
        expected = {
            EventLabel.SIREN: 0.10,
            EventLabel.GLASS_BREAK: 0.20,
            EventLabel.GUNSHOT: 0.30,
        }
        for label, amplitude in expected.items():
            value = 0.9 if abs(level - amplitude) < 0.03 else 0.1
            scores[0, [item.index for item in LABEL_MAPPING[label]]] = value
        return (
            scores,
            np.zeros((1, 1024), dtype=np.float32),
            np.zeros((96, YAMNET.mel_bands), dtype=np.float32),
        )


def _fixture_dataset(tmp_path: Path) -> tuple[Path, DatasetSpec]:
    raw = tmp_path / "raw"
    metadata = raw / "esc50" / "meta"
    audio = raw / "esc50" / "audio"
    metadata.mkdir(parents=True)
    audio.mkdir(parents=True)
    categories = {
        "siren": 0.10,
        "glass_breaking": 0.20,
        "gun_shot": 0.30,
        "dog": 0.01,
    }
    rows = ["filename,fold,category"]
    for fold in (1, 5):
        for index, (category, amplitude) in enumerate(categories.items()):
            filename = f"{fold}-{index}.wav"
            rows.append(f"{filename},{fold},{category}")
            sf.write(audio / filename, np.full(8_000, amplitude, dtype=np.float32), 16_000, subtype="PCM_16")
    (metadata / "esc50.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (raw / "esc50" / "LICENSE").write_text("test license", encoding="utf-8")
    spec = DatasetSpec(
        dataset_id="esc50",
        metadata_path="esc50/meta/esc50.csv",
        audio_directory="esc50/audio",
        license_paths=("esc50/LICENSE",),
        calibration_folds=(1,),
        holdout_folds=(5,),
        calibration_per_category=1,
        holdout_per_category=1,
        positive_categories={
            EventLabel.SIREN: ("siren",),
            EventLabel.GLASS_BREAK: ("glass_breaking",),
            EventLabel.GUNSHOT: ("gun_shot",),
        },
    )
    return raw, spec


def test_select_labeled_samples_is_balanced_disjoint_and_deterministic(tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    first, provenance = select_labeled_samples(raw, (spec,))
    second, _ = select_labeled_samples(raw, (spec,))
    assert first == second
    assert len(first) == 8
    assert len({item.relative_audio_path for item in first}) == 8
    assert {item.split for item in first} == {"calibration", "holdout"}
    assert provenance[0]["selected_counts"] == {"calibration": 4, "holdout": 4}
    assert len(provenance[0]["metadata_sha256"]) == 64
    assert len(provenance[0]["license_files"][0]["sha256"]) == 64


def test_missing_required_category_is_rejected(tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    changed = replace(spec, positive_categories={EventLabel.SIREN: ("missing",)})
    with pytest.raises(AcousticEvaluationError) as caught:
        select_labeled_samples(raw, (changed,))
    assert caught.value.code == "invalid_dataset"


def test_duplicate_positive_category_mapping_is_rejected(tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    changed = replace(spec, positive_categories={
        EventLabel.SIREN: ("siren",), EventLabel.GUNSHOT: ("siren",)
    })
    with pytest.raises(AcousticEvaluationError, match="unique"):
        select_labeled_samples(raw, (changed,))


def test_insufficient_split_samples_are_rejected(tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    with pytest.raises(AcousticEvaluationError) as caught:
        select_labeled_samples(raw, (replace(spec, calibration_per_category=2),))
    assert caught.value.code == "insufficient_samples"


def test_empty_metadata_is_rejected_before_csv_parsing(tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    (raw / spec.metadata_path).write_bytes(b"")
    with pytest.raises(AcousticEvaluationError) as caught:
        select_labeled_samples(raw, (spec,))
    assert caught.value.code == "invalid_file_size"


def test_selected_sample_limit_is_enforced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    monkeypatch.setattr(evaluation, "MAX_SELECTED_SAMPLES", 1)
    with pytest.raises(AcousticEvaluationError) as caught:
        select_labeled_samples(raw, (spec,))
    assert caught.value.code == "too_many_samples"


@pytest.mark.parametrize("threshold,expected", [
    (0.5, (1, 1, 1, 1)),
    (0.8, (1, 0, 1, 2)),
    (0.0, (2, 2, 0, 0)),
])
def test_binary_metrics_confusion_counts(threshold: float, expected: tuple[int, ...]) -> None:
    result = binary_metrics([0.9, 0.7, 0.2, 0.1], [True, False, True, False], threshold)
    assert (result.true_positive, result.false_positive, result.false_negative, result.true_negative) == expected


def test_binary_metrics_rates_and_average_precision() -> None:
    result = binary_metrics([0.9, 0.8, 0.1], [True, False, True], 0.5)
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5
    assert result.specificity == 0.0
    assert result.balanced_accuracy == 0.25
    assert result.average_precision == pytest.approx(5 / 6)


def test_average_precision_ties_are_order_invariant() -> None:
    assert average_precision([0.5, 0.5], [True, False]) == 0.5
    assert average_precision([0.5, 0.5], [False, True]) == 0.5


def test_negative_only_average_precision_is_explicitly_unavailable() -> None:
    result = binary_metrics([0.2, 0.1], [False, False], 0.5)
    assert result.average_precision is None
    assert result.recall is None
    assert result.balanced_accuracy is None


def test_threshold_selection_maximizes_f1_without_holdout_data() -> None:
    result = select_f1_threshold([0.9, 0.8, 0.4, 0.1], [True, False, True, False])
    assert result.threshold == pytest.approx(0.4)
    assert result.f1 == pytest.approx(0.8)


@pytest.mark.parametrize("truth", [[True, True], [False, False]])
def test_threshold_selection_requires_both_classes(truth: list[bool]) -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        select_f1_threshold([0.9, 0.1], truth)


def test_evaluate_builds_auditable_calibration_and_holdout_report(tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    loaded = LoadedAcousticModel(_metadata(), (), FakeModel())
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    report = evaluate_labeled_samples(raw, loaded, specs=(spec,), now=now)
    assert report["schema_version"] == "1.0"
    assert report["sample_count"] == 8
    assert report["decision_status"] == "research_baseline_not_production_calibration"
    assert set(report["thresholds"]) == {"siren", "glass_break", "gunshot"}
    for details in report["thresholds"].values():
        assert details["calibration"]["f1"] == 1.0
        assert details["holdout"]["f1"] == 1.0
        assert details["holdout"]["sample_count"] == 4
    assert report["coverage"]["explosion"]["status"] == "not_evaluated"
    assert report["coverage"]["siren"]["status"] == "evaluated"
    assert str(report["evaluation_id"]).startswith("a3-4-")
    assert all(len(row["source_sha256"]) == 64 for row in report["samples"])
    json.dumps(report, allow_nan=False)


def test_changed_model_identity_is_rejected_before_reading_data(tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    metadata = replace(_metadata(), artifact_sha256="0" * 64)
    with pytest.raises(AcousticEvaluationError) as caught:
        evaluate_labeled_samples(raw, LoadedAcousticModel(metadata, (), FakeModel()), specs=(spec,))
    assert caught.value.code == "model_mismatch"


def test_naive_evaluation_time_is_rejected_before_model_work(tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    with pytest.raises(AcousticEvaluationError) as caught:
        evaluate_labeled_samples(
            raw, LoadedAcousticModel(_metadata(), (), FakeModel()), specs=(spec,),
            now=datetime(2026, 9, 18, 12, 0),
        )
    assert caught.value.code == "invalid_time"


def test_evaluation_identity_excludes_run_timestamp(tmp_path: Path) -> None:
    raw, spec = _fixture_dataset(tmp_path)
    loaded = LoadedAcousticModel(_metadata(), (), FakeModel())
    first = evaluate_labeled_samples(raw, loaded, specs=(spec,), now=datetime(2026, 9, 18, tzinfo=UTC))
    second = evaluate_labeled_samples(raw, loaded, specs=(spec,), now=datetime(2026, 9, 19, tzinfo=UTC))
    assert first["created_at"] != second["created_at"]
    assert first["evaluation_id"] == second["evaluation_id"]


def test_invalid_model_tensor_is_converted_to_stable_error() -> None:
    class InvalidModel:
        def __call__(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (np.array([["bad"]]), np.zeros((1, 1024), np.float32), np.zeros((96, 64), np.float32))

    with pytest.raises(AcousticEvaluationError) as caught:
        evaluation._score_waveform(InvalidModel(), np.zeros(8_000, dtype=np.float32))
    assert caught.value.code == "invalid_model_output"


def test_save_report_never_overwrites_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "result" / "evaluation.json"
    report = {"schema_version": "1.0", "value": 1}
    save_evaluation_report(report, output)
    original = output.read_bytes()
    with pytest.raises(AcousticEvaluationError) as caught:
        save_evaluation_report(report, output)
    assert caught.value.code == "output_exists"
    assert output.read_bytes() == original
