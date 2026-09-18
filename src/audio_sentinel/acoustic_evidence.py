"""A3.3 versioned, timestamped acoustic-evidence JSON persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.acoustic_aggregation import (
    AcousticAggregationResult,
    AcousticAggregationSettings,
    aggregate_acoustic_events,
)
from audio_sentinel.acoustic_inference import AcousticInferenceResult, expected_yamnet_patches
from audio_sentinel.acoustic_model import LABEL_MAPPING, LABEL_MAPPING_VERSION, YAMNET
from audio_sentinel.audio_loader import validate_processing_consent
from audio_sentinel.config import Paths
from audio_sentinel.contracts import EventLabel
from audio_sentinel.feature_persistence import _read_bytes, _safe_file
from audio_sentinel.persistence import _is_link, _remove_stage
from audio_sentinel.preparation import PreparedAudioManifest, validate_relative_audio_path


class AcousticEvidenceError(ValueError):
    """A stable evidence persistence failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class AcousticEvidencePolicy(BaseModel):
    """Resource limits for producing and reloading one evidence document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_document_bytes: int = Field(default=67_108_864, gt=0, strict=True)
    max_manifest_bytes: int = Field(default=16_777_216, gt=0, strict=True)
    max_window_bytes: int = Field(default=268_435_456, gt=0, strict=True)
    max_source_bytes: int = Field(default=1_073_741_824, gt=0, strict=True)
    max_windows: int = Field(default=10_000, ge=0, strict=True)
    max_events: int = Field(default=1_000_000, ge=0, strict=True)
    max_contributions: int = Field(default=1_000_000, ge=0, strict=True)


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, protected_namespaces=()
    )


class EvidenceTensorContract(EvidenceRecord):
    name: str = Field(min_length=1, max_length=128)
    shape: tuple[int | None, ...]
    dtype: str = Field(min_length=1, max_length=64)


class EvidenceModel(EvidenceRecord):
    model_id: str
    model_version: str
    model_handle: str
    model_path: str
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    class_map_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    label_mapping_version: str
    runtime_distribution: str
    runtime_version: str
    exported_with_tensorflow: str
    exported_with_tensorflow_git: str
    signature_name: str
    input: EvidenceTensorContract
    outputs: tuple[EvidenceTensorContract, ...]
    num_classes: int = Field(gt=0, strict=True)

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        return validate_relative_audio_path(value)


class EvidenceSource(EvidenceRecord):
    preparation_manifest_path: str
    preparation_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    clip_id: str = Field(min_length=3, max_length=128)

    @field_validator("preparation_manifest_path")
    @classmethod
    def validate_manifest_path(cls, value: str) -> str:
        return validate_relative_audio_path(value)


class EvidenceWindow(EvidenceRecord):
    window_id: str = Field(min_length=3, max_length=128)
    audio_path: str
    window_audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    window_seconds: float = Field(gt=0)
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(gt=0, strict=True)
    padding_samples: int = Field(ge=0, strict=True)
    input_num_samples: int = Field(gt=0, strict=True)
    patch_count: int = Field(gt=0, strict=True)

    @field_validator("audio_path")
    @classmethod
    def validate_audio_path(cls, value: str) -> str:
        return validate_relative_audio_path(value)

    @model_validator(mode="after")
    def validate_window(self) -> "EvidenceWindow":
        if self.input_num_samples != self.end_sample - self.start_sample:
            raise ValueError("input_num_samples must equal the real window span")
        if self.patch_count != expected_yamnet_patches(self.input_num_samples):
            raise ValueError("patch_count must follow the pinned YAMNet patch grid")
        return self


class EvidenceAcousticClass(EvidenceRecord):
    index: int = Field(ge=0, strict=True)
    mid: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=256)


class EvidenceContribution(EvidenceRecord):
    label: EventLabel
    window_id: str
    window_audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    patch_index: int = Field(ge=0, strict=True)
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(gt=0, strict=True)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    score: float = Field(ge=0, le=1)
    winning_class: EvidenceAcousticClass


class EvidenceEvent(EvidenceRecord):
    label: EventLabel
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(gt=0, strict=True)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    peak_score: float = Field(ge=0, le=1)
    source_window_ids: tuple[str, ...]
    contributions: tuple[EvidenceContribution, ...]


def _canonical_hash(value: object) -> str:
    document = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


class AcousticEvidenceDocument(EvidenceRecord):
    """Persistent model evidence; deliberately not an incident/risk contract."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, protected_namespaces=(),
        json_schema_extra={"$id": "https://audio-sentinel.local/schemas/v1/acoustic-evidence.schema.json"},
    )

    schema_version: Literal["1.0"] = "1.0"
    document_type: Literal["acoustic_event_candidates"] = "acoustic_event_candidates"
    evidence_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: datetime
    sample_rate_hz: int = Field(gt=0, strict=True)
    source: EvidenceSource
    model: EvidenceModel
    aggregation: AcousticAggregationSettings
    input_window_count: int = Field(ge=0, strict=True)
    input_patch_count: int = Field(ge=0, strict=True)
    windows: tuple[EvidenceWindow, ...]
    events: tuple[EvidenceEvent, ...]

    @model_validator(mode="after")
    def validate_document(self) -> "AcousticEvidenceDocument":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.sample_rate_hz != YAMNET.sample_rate_hz:
            raise ValueError("sample_rate_hz must match the pinned acoustic model")
        expected_outputs = (
            EvidenceTensorContract(name="output_0", shape=(None, YAMNET.num_classes), dtype="float32"),
            EvidenceTensorContract(name="output_1", shape=(None, 1024), dtype="float32"),
            EvidenceTensorContract(name="output_2", shape=(None, YAMNET.mel_bands), dtype="float32"),
        )
        if (
            self.model.model_id != YAMNET.model_id
            or self.model.model_version != "1"
            or self.model.model_handle != YAMNET.model_handle
            or self.model.artifact_sha256 != YAMNET.artifact_sha256
            or self.model.class_map_sha256 != YAMNET.class_map_sha256
            or self.model.label_mapping_version != LABEL_MAPPING_VERSION
            or self.model.runtime_distribution != YAMNET.runtime_distribution
            or self.model.runtime_version != YAMNET.runtime_version
            or self.model.signature_name != "serving_default"
            or self.model.input != EvidenceTensorContract(
                name="waveform", shape=(None,), dtype="float32"
            )
            or self.model.outputs != expected_outputs
            or self.model.num_classes != YAMNET.num_classes
        ):
            raise ValueError("model metadata does not match the verified pinned YAMNet contract")
        if self.input_window_count != len(self.windows):
            raise ValueError("input_window_count must equal the window inventory")
        if self.input_patch_count != sum(item.patch_count for item in self.windows):
            raise ValueError("input_patch_count must equal the window patch inventory")
        if len({item.window_id for item in self.windows}) != len(self.windows):
            raise ValueError("window IDs must be unique")

        windows = {item.window_id: item for item in self.windows}
        expected_events = sorted(
            self.events, key=lambda item: (item.start_sample, item.end_sample, item.label.value)
        )
        if list(self.events) != expected_events:
            raise ValueError("events must use chronological deterministic ordering")

        contribution_count = 0
        for event in self.events:
            if not event.contributions:
                raise ValueError("every event must contain at least one contribution")
            if event.end_sample <= event.start_sample:
                raise ValueError("event end_sample must be greater than start_sample")
            self._validate_seconds(event.start_sample, event.start_seconds)
            self._validate_seconds(event.end_sample, event.end_seconds)
            if event.source_window_ids != tuple(dict.fromkeys(c.window_id for c in event.contributions)):
                raise ValueError("source_window_ids must match contribution order")
            if (event.start_sample, event.end_sample) != (
                min(c.start_sample for c in event.contributions),
                max(c.end_sample for c in event.contributions),
            ):
                raise ValueError("event span must be the union of its contribution support")
            if event.peak_score != max(c.score for c in event.contributions):
                raise ValueError("peak_score must be the maximum contribution score")

            threshold = self.aggregation.threshold_for(event.label)
            previous: tuple[int, int, str, int] | None = None
            for contribution in event.contributions:
                contribution_count += 1
                if contribution.label is not event.label or contribution.score < threshold:
                    raise ValueError("contribution label or threshold does not match its event")
                window = windows.get(contribution.window_id)
                if window is None or window.window_audio_sha256 != contribution.window_audio_sha256:
                    raise ValueError("contribution source window is missing or has a different hash")
                if contribution.patch_index >= window.patch_count:
                    raise ValueError("contribution patch_index is outside its source window")
                expected_start = window.start_sample + contribution.patch_index * YAMNET.patch_hop_samples
                expected_end = min(expected_start + YAMNET.patch_support_samples, window.end_sample)
                if (contribution.start_sample, contribution.end_sample) != (expected_start, expected_end):
                    raise ValueError("contribution span does not match the pinned YAMNet patch grid")
                self._validate_seconds(contribution.start_sample, contribution.start_seconds)
                self._validate_seconds(contribution.end_sample, contribution.end_seconds)
                allowed = LABEL_MAPPING[event.label]
                winner = contribution.winning_class
                if not any(
                    (winner.index, winner.mid, winner.display_name) == (item.index, item.mid, item.display_name)
                    for item in allowed
                ):
                    raise ValueError("winning_class is not mapped to the event label")
                order = (
                    contribution.start_sample, contribution.end_sample,
                    contribution.window_id, contribution.patch_index,
                )
                if previous is not None and order < previous:
                    raise ValueError("contributions must use deterministic temporal ordering")
                previous = order

        if contribution_count > self.aggregation.max_contributions:
            raise ValueError("document exceeds its recorded aggregation contribution limit")
        payload = self.model_dump(mode="json", exclude={"evidence_id", "created_at"})
        if self.evidence_id != _canonical_hash(payload):
            raise ValueError("evidence_id does not match the semantic document content")
        return self

    def _validate_seconds(self, sample: int, seconds: float) -> None:
        if not math.isclose(seconds, sample / self.sample_rate_hz, rel_tol=0, abs_tol=1e-12):
            raise ValueError("seconds must be derived exactly from the sample offset")


@dataclass(frozen=True)
class SavedAcousticEvidence:
    directory: Path
    evidence_path: Path
    evidence: AcousticEvidenceDocument
    reused: bool


def _clock(now: datetime | None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise AcousticEvidenceError("invalid_time", "Evidence creation time must be timezone-aware.")
    return value.astimezone(UTC)


def _event_document(event: object, sample_rate_hz: int) -> dict[str, object]:
    values = event.as_dict()
    contributions = []
    for contribution in event.contributions:
        item = contribution.as_dict()
        item["start_seconds"] = contribution.start_sample / sample_rate_hz
        item["end_seconds"] = contribution.end_sample / sample_rate_hz
        contributions.append(item)
    values["start_seconds"] = event.start_sample / sample_rate_hz
    values["end_seconds"] = event.end_sample / sample_rate_hz
    values["contributions"] = contributions
    return values


def build_acoustic_evidence(
    inference: AcousticInferenceResult,
    aggregation: AcousticAggregationResult,
    *,
    now: datetime | None = None,
) -> AcousticEvidenceDocument:
    """Build and fully validate the v1 evidence contract from A3.2 and B3.2 outputs."""
    try:
        recomputed = aggregate_acoustic_events(inference, aggregation.settings)
        if recomputed != aggregation:
            raise AcousticEvidenceError(
                "aggregation_mismatch", "Aggregation does not match the supplied inference snapshot."
            )
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "document_type": "acoustic_event_candidates",
            "sample_rate_hz": aggregation.sample_rate_hz,
            "source": {
                "preparation_manifest_path": aggregation.preparation_manifest_path,
                "preparation_manifest_sha256": aggregation.preparation_manifest_sha256,
                "raw_audio_sha256": aggregation.raw_audio_sha256,
                "clip_id": aggregation.clip_id,
            },
            "model": aggregation.model.as_dict(),
            "aggregation": aggregation.settings.model_dump(mode="json"),
            "input_window_count": aggregation.input_window_count,
            "input_patch_count": aggregation.input_patch_count,
            "windows": [
                {
                    **item.window.model_dump(mode="json"),
                    "window_audio_sha256": item.window_audio_sha256,
                    "input_num_samples": item.input_num_samples,
                    "patch_count": item.patch_count,
                }
                for item in inference.windows
            ],
            "events": [_event_document(item, aggregation.sample_rate_hz) for item in aggregation.events],
        }
        return AcousticEvidenceDocument(
            evidence_id=_canonical_hash(payload), created_at=_clock(now), **payload
        )
    except AcousticEvidenceError:
        raise
    except Exception as error:
        raise AcousticEvidenceError(
            "invalid_evidence", "Acoustic evidence could not be constructed from the supplied results."
        ) from error


def _policy_limits(document: AcousticEvidenceDocument, policy: AcousticEvidencePolicy) -> None:
    contributions = sum(len(item.contributions) for item in document.events)
    if len(document.windows) > policy.max_windows:
        raise AcousticEvidenceError("too_many_windows", "Evidence window count exceeds the configured limit.")
    if len(document.events) > policy.max_events:
        raise AcousticEvidenceError("too_many_events", "Evidence event count exceeds the configured limit.")
    if contributions > policy.max_contributions:
        raise AcousticEvidenceError(
            "too_many_contributions", "Evidence contribution count exceeds the configured limit."
        )


def _source_current(
    paths: Paths, document: AcousticEvidenceDocument, policy: AcousticEvidencePolicy,
    now: datetime | None,
) -> None:
    try:
        manifest_path = _safe_file(
            paths.root, paths.interim_data, document.source.preparation_manifest_path
        )
        manifest_bytes = _read_bytes(manifest_path, policy.max_manifest_bytes)
        manifest = PreparedAudioManifest.model_validate_json(manifest_bytes)
        validate_processing_consent(manifest.clip.consent, _clock(now))
        if (
            hashlib.sha256(manifest_bytes).hexdigest() != document.source.preparation_manifest_sha256
            or manifest.source.sha256 != document.source.raw_audio_sha256
            or manifest.clip.clip_id != document.source.clip_id
            or manifest.clip.sample_rate_hz != document.sample_rate_hz
            or len(manifest.windows) != len(document.windows)
        ):
            raise AcousticEvidenceError(
                "source_changed", "Preparation manifest differs from the evidence source snapshot."
            )

        used_bytes = len(manifest_bytes)
        for expected, recorded in zip(manifest.windows, document.windows, strict=True):
            if (
                expected.window_id != recorded.window_id
                or expected.audio_path != recorded.audio_path
                or expected.window_seconds != recorded.window_seconds
                or expected.start_sample != recorded.start_sample
                or expected.end_sample != recorded.end_sample
                or expected.padding_samples != recorded.padding_samples
            ):
                raise AcousticEvidenceError(
                    "source_changed", "Prepared window inventory differs from the evidence snapshot."
                )
            audio_path = _safe_file(paths.root, paths.interim_data, expected.audio_path)
            audio_bytes = _read_bytes(audio_path, policy.max_window_bytes)
            used_bytes += len(audio_bytes)
            if used_bytes > policy.max_source_bytes:
                raise AcousticEvidenceError(
                    "source_too_large", "Evidence sources exceed the configured verification byte limit."
                )
            if hashlib.sha256(audio_bytes).hexdigest() != recorded.window_audio_sha256:
                raise AcousticEvidenceError(
                    "source_changed", "A prepared window differs from the evidence source snapshot."
                )
        validate_processing_consent(manifest.clip.consent, _clock(now))
    except AcousticEvidenceError:
        raise
    except Exception as error:
        code = getattr(error, "code", "invalid_source")
        raise AcousticEvidenceError(code, "Acoustic evidence sources could not be verified.") from error


def _output_parent(paths: Paths) -> Path:
    try:
        root = paths.root.resolve(strict=True)
        processed = Path(os.path.abspath(paths.processed_data))
        parts = processed.relative_to(root).parts
        if not parts:
            raise ValueError("root is not an output directory")
        for other in (paths.raw_data, paths.interim_data):
            other = other.resolve()
            if processed.is_relative_to(other) or other.is_relative_to(processed):
                raise ValueError("output overlaps source data")
        current = root
        for part in (*parts, "acoustic-evidence"):
            current = current / part
            if _is_link(current):
                raise ValueError("linked output")
            current.mkdir(exist_ok=True)
        return current.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise AcousticEvidenceError(
            "invalid_output_path",
            "Evidence output must be separate from raw/interim data without linked directories.",
        ) from error


def _document_bytes(document: AcousticEvidenceDocument) -> bytes:
    return (document.model_dump_json(indent=2) + "\n").encode("utf-8")


def _load(
    paths: Paths, evidence_path: str | Path, policy: AcousticEvidencePolicy,
    now: datetime | None,
) -> AcousticEvidenceDocument:
    try:
        path = _safe_file(paths.root, paths.processed_data, evidence_path)
        raw = _read_bytes(path, policy.max_document_bytes)
        document = AcousticEvidenceDocument.model_validate_json(raw)
        expected = paths.processed_data / f"acoustic-evidence/{document.evidence_id}/evidence.json"
        if path != expected or {item.name for item in path.parent.iterdir()} != {"evidence.json"}:
            raise AcousticEvidenceError(
                "output_conflict", "Evidence identity does not match its bundle path or inventory."
            )
        _policy_limits(document, policy)
        _source_current(paths, document, policy, now)
        return document
    except AcousticEvidenceError:
        raise
    except MemoryError as error:
        raise AcousticEvidenceError(
            "insufficient_memory", "Not enough memory to reload acoustic evidence."
        ) from error
    except Exception as error:
        code = getattr(error, "code", "invalid_document")
        raise AcousticEvidenceError(code, "Acoustic evidence could not be read or verified.") from error


def load_acoustic_evidence(
    paths: Paths, evidence_path: str | Path, *,
    policy: AcousticEvidencePolicy | None = None, now: datetime | None = None,
) -> AcousticEvidenceDocument:
    """Reload evidence by a path relative to processed_data and recheck its sources."""
    resolved_policy = AcousticEvidencePolicy.model_validate(
        (policy or AcousticEvidencePolicy()).model_dump()
    )
    return _load(paths, evidence_path, resolved_policy, now)


def save_acoustic_evidence(
    paths: Paths, inference: AcousticInferenceResult, aggregation: AcousticAggregationResult, *,
    policy: AcousticEvidencePolicy | None = None, now: datetime | None = None,
) -> SavedAcousticEvidence:
    """Atomically persist one verified evidence JSON bundle without raw audio."""
    resolved_policy = AcousticEvidencePolicy.model_validate(
        (policy or AcousticEvidencePolicy()).model_dump()
    )
    stage: Path | None = None
    parent: Path | None = None
    try:
        document = build_acoustic_evidence(inference, aggregation, now=now)
        _policy_limits(document, resolved_policy)
        _source_current(paths, document, resolved_policy, now)
        raw = _document_bytes(document)
        if len(raw) > resolved_policy.max_document_bytes:
            raise AcousticEvidenceError(
                "output_too_large", "Acoustic evidence exceeds the configured document byte limit."
            )

        parent = _output_parent(paths)
        destination = parent / document.evidence_id
        relative = f"acoustic-evidence/{document.evidence_id}/evidence.json"
        if _is_link(destination) or (destination.exists() and not destination.is_dir()):
            raise AcousticEvidenceError(
                "output_conflict", "Evidence destination is not a regular bundle directory."
            )
        stage = Path(mkdtemp(prefix=".pending-", dir=parent)).resolve()
        staged_path = stage / "evidence.json"
        with staged_path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if AcousticEvidenceDocument.model_validate_json(
            _read_bytes(staged_path, resolved_policy.max_document_bytes)
        ) != document:
            raise AcousticEvidenceError("verification_failed", "Evidence JSON failed readback.")
        _source_current(paths, document, resolved_policy, now)

        reused = destination.exists()
        if not reused:
            try:
                stage.rename(destination)
                stage = None
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) and not isinstance(
                    error, FileExistsError
                ):
                    raise
                reused = True
        if reused:
            existing = _load(paths, relative, resolved_policy, now)
            comparable = existing.model_copy(update={"created_at": document.created_at})
            if comparable != document:
                raise AcousticEvidenceError(
                    "output_conflict", "Existing evidence differs; it was not overwritten."
                )
            document = existing
        return SavedAcousticEvidence(destination, destination / "evidence.json", document, reused)
    except AcousticEvidenceError:
        raise
    except MemoryError as error:
        raise AcousticEvidenceError(
            "insufficient_memory", "Not enough memory to persist acoustic evidence."
        ) from error
    except OSError as error:
        raise AcousticEvidenceError(
            "write_failed", "Acoustic evidence could not be written or verified."
        ) from error
    finally:
        if stage is not None and parent is not None:
            _remove_stage(stage, parent)


def acoustic_evidence_schema_documents() -> dict[str, dict[str, object]]:
    """Return the checked-in v1 acoustic evidence JSON Schema."""
    return {"acoustic-evidence.schema.json": AcousticEvidenceDocument.model_json_schema()}


def write_acoustic_evidence_schemas(output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, document in acoustic_evidence_schema_documents().items():
        (output_directory / filename).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
