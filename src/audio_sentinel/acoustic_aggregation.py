"""B3.2 deterministic aggregation of overlapping YAMNet patch evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.acoustic_inference import AcousticInferenceResult, expected_yamnet_patches
from audio_sentinel.acoustic_loader import AcousticModelMetadata, TensorContract
from audio_sentinel.acoustic_model import (
    LABEL_MAPPING,
    LABEL_MAPPING_VERSION,
    YAMNET,
    AcousticClass,
)
from audio_sentinel.contracts import EventLabel


class AcousticAggregationError(ValueError):
    """A stable aggregation failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class AcousticLabelThreshold(BaseModel):
    """One explicit experimental threshold; B3.2 does not approve its value."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    label: EventLabel
    minimum_score: float = Field(ge=0, le=1)


class AcousticAggregationSettings(BaseModel):
    """Complete mapped-label thresholds plus a sample-exact merge rule."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    thresholds: tuple[AcousticLabelThreshold, ...]
    merge_gap_samples: int = Field(default=0, ge=0, strict=True)
    max_contributions: int = Field(default=1_000_000, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "AcousticAggregationSettings":
        labels = tuple(item.label for item in self.thresholds)
        if len(labels) != len(set(labels)):
            raise ValueError("threshold labels must be unique")
        if set(labels) != set(LABEL_MAPPING):
            raise ValueError("thresholds must cover every mapped acoustic label exactly once")
        if labels != tuple(sorted(labels, key=lambda label: label.value)):
            raise ValueError("thresholds must be ordered by label value")
        return self

    @classmethod
    def uniform(cls, minimum_score: float, **kwargs: object) -> "AcousticAggregationSettings":
        """Build an explicit structural-test policy; the score is not calibrated."""
        return cls(
            thresholds=tuple(
                AcousticLabelThreshold(label=label, minimum_score=minimum_score)
                for label in sorted(LABEL_MAPPING, key=lambda item: item.value)
            ),
            **kwargs,
        )

    def threshold_for(self, label: EventLabel) -> float:
        return next(item.minimum_score for item in self.thresholds if item.label is label)


@dataclass(frozen=True)
class AcousticPatchContribution:
    """One thresholded mapped score with its exact source-patch provenance."""

    label: EventLabel
    window_id: str
    window_audio_sha256: str
    patch_index: int
    start_sample: int
    end_sample: int
    score: float
    winning_class: AcousticClass

    def as_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["label"] = self.label.value
        return document


@dataclass(frozen=True)
class AcousticEventCandidate:
    """Unioned patch support for one label; not an EventAnnotation or incident."""

    label: EventLabel
    start_sample: int
    end_sample: int
    peak_score: float
    contributions: tuple[AcousticPatchContribution, ...]

    @property
    def source_window_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.window_id for item in self.contributions))

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label.value,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "peak_score": self.peak_score,
            "source_window_ids": self.source_window_ids,
            "contributions": [item.as_dict() for item in self.contributions],
        }


@dataclass(frozen=True)
class AcousticAggregationResult:
    """Candidate acoustic intervals derived from one A3.2 inference snapshot."""

    model: AcousticModelMetadata
    preparation_manifest_path: str
    preparation_manifest_sha256: str
    raw_audio_sha256: str
    clip_id: str
    sample_rate_hz: int
    settings: AcousticAggregationSettings
    input_window_count: int
    input_patch_count: int
    events: tuple[AcousticEventCandidate, ...]

    def to_summary(self) -> dict[str, object]:
        """Return a JSON-ready diagnostic summary, not the A3.3 evidence schema."""
        return {
            "clip_id": self.clip_id,
            "model_id": self.model.model_id,
            "model_version": self.model.model_version,
            "label_mapping_version": self.model.label_mapping_version,
            "preparation_manifest_path": self.preparation_manifest_path,
            "preparation_manifest_sha256": self.preparation_manifest_sha256,
            "raw_audio_sha256": self.raw_audio_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "input_window_count": self.input_window_count,
            "input_patch_count": self.input_patch_count,
            "event_count": len(self.events),
            "events": [event.as_dict() for event in self.events],
        }


def _validate_model(metadata: AcousticModelMetadata) -> None:
    expected_outputs = (
        TensorContract("output_0", (None, YAMNET.num_classes), "float32"),
        TensorContract("output_1", (None, 1024), "float32"),
        TensorContract("output_2", (None, YAMNET.mel_bands), "float32"),
    )
    if (
        metadata.model_id != YAMNET.model_id
        or metadata.model_version != "1"
        or metadata.model_handle != YAMNET.model_handle
        or metadata.artifact_sha256 != YAMNET.artifact_sha256
        or metadata.class_map_sha256 != YAMNET.class_map_sha256
        or metadata.label_mapping_version != LABEL_MAPPING_VERSION
        or metadata.runtime_distribution != YAMNET.runtime_distribution
        or metadata.runtime_version != YAMNET.runtime_version
        or metadata.signature_name != "serving_default"
        or metadata.input != TensorContract("waveform", (None,), "float32")
        or metadata.outputs != expected_outputs
        or metadata.num_classes != YAMNET.num_classes
    ):
        raise AcousticAggregationError("model_mismatch", "Aggregation requires verified YAMNet v1 metadata.")


def _validate_inference(result: AcousticInferenceResult) -> None:
    _validate_model(result.model)
    seen_windows: set[str] = set()
    for window in result.windows:
        if window.window.window_id in seen_windows:
            raise AcousticAggregationError("invalid_inference", "Inference contains a duplicate window ID.")
        seen_windows.add(window.window.window_id)
        patches = expected_yamnet_patches(window.input_num_samples)
        frames = YAMNET.patch_frames + (patches - 1) * (YAMNET.patch_frames // 2)
        if (
            window.scores.dtype != np.dtype("float32")
            or window.scores.shape != (patches, YAMNET.num_classes)
            or window.embeddings_shape != (patches, 1024)
            or window.spectrogram_shape != (frames, YAMNET.mel_bands)
            or not np.isfinite(window.scores).all()
            or np.any(window.scores < 0)
            or np.any(window.scores > 1)
        ):
            raise AcousticAggregationError("invalid_inference", "Inference scores violate the YAMNet patch contract.")
        if window.input_num_samples != window.window.end_sample - window.window.start_sample:
            raise AcousticAggregationError("invalid_inference", "Inference input length differs from its real window span.")
        for start, end in window.patch_spans_samples:
            if not window.window.start_sample <= start < end <= window.window.end_sample:
                raise AcousticAggregationError("invalid_inference", "Patch support falls outside its real window span.")


def _contributions(
    result: AcousticInferenceResult, settings: AcousticAggregationSettings,
) -> dict[EventLabel, list[AcousticPatchContribution]]:
    grouped = {label: [] for label in LABEL_MAPPING}
    count = 0
    for window in result.windows:
        for patch_index, (start, end) in enumerate(window.patch_spans_samples):
            scores = window.scores[patch_index]
            for label, classes in LABEL_MAPPING.items():
                winning_class = max(classes, key=lambda item: (float(scores[item.index]), -item.index))
                score = float(scores[winning_class.index])
                if score < settings.threshold_for(label):
                    continue
                count += 1
                if count > settings.max_contributions:
                    raise AcousticAggregationError(
                        "too_many_contributions", "Thresholded patch evidence exceeds the aggregation limit."
                    )
                grouped[label].append(AcousticPatchContribution(
                    label=label,
                    window_id=window.window.window_id,
                    window_audio_sha256=window.window_audio_sha256,
                    patch_index=patch_index,
                    start_sample=start,
                    end_sample=end,
                    score=score,
                    winning_class=winning_class,
                ))
    return grouped


def _merge(
    label: EventLabel, contributions: list[AcousticPatchContribution], gap: int,
) -> list[AcousticEventCandidate]:
    ordered = sorted(
        contributions,
        key=lambda item: (item.start_sample, item.end_sample, item.window_id, item.patch_index),
    )
    events: list[AcousticEventCandidate] = []
    current: list[AcousticPatchContribution] = []
    start = end = 0
    peak = 0.0
    for contribution in ordered:
        if current and contribution.start_sample > end + gap:
            events.append(AcousticEventCandidate(label, start, end, peak, tuple(current)))
            current = []
        if not current:
            start, end, peak = contribution.start_sample, contribution.end_sample, contribution.score
        else:
            end = max(end, contribution.end_sample)
            peak = max(peak, contribution.score)
        current.append(contribution)
    if current:
        events.append(AcousticEventCandidate(label, start, end, peak, tuple(current)))
    return events


def aggregate_acoustic_events(
    result: AcousticInferenceResult, settings: AcousticAggregationSettings,
) -> AcousticAggregationResult:
    """Map classes, threshold patches, and union overlap without score inflation."""
    settings = AcousticAggregationSettings.model_validate(settings.model_dump())
    _validate_inference(result)
    grouped = _contributions(result, settings)
    events = tuple(sorted(
        (
            event
            for label, contributions in grouped.items()
            for event in _merge(label, contributions, settings.merge_gap_samples)
        ),
        key=lambda event: (event.start_sample, event.end_sample, event.label.value),
    ))
    return AcousticAggregationResult(
        model=result.model,
        preparation_manifest_path=result.preparation_manifest_path,
        preparation_manifest_sha256=result.preparation_manifest_sha256,
        raw_audio_sha256=result.raw_audio_sha256,
        clip_id=result.clip_id,
        sample_rate_hz=YAMNET.sample_rate_hz,
        settings=settings,
        input_window_count=len(result.windows),
        input_patch_count=result.patch_count,
        events=events,
    )
