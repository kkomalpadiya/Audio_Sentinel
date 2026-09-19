"""A4.1 versioned speech-evidence contract and deterministic reliability rules."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
import math
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.preparation import Identifier, validate_relative_audio_path


SPEECH_EVIDENCE_SCHEMA_VERSION = "1.0"
SPEECH_RELIABILITY_POLICY_VERSION = "1.0"


class TranscriptConfidenceKind(str, Enum):
    """Meaning of a normalized score; none of these labels makes it a truth probability."""

    MODEL_SCORE = "model_score"
    DERIVED_SCORE = "derived_score"
    CALIBRATED_PROBABILITY = "calibrated_probability"


class TranscriptReliability(str, Enum):
    """Permitted downstream handling for a transcript candidate."""

    NOT_TRANSCRIBED = "not_transcribed"
    REJECTED_LOW_CONFIDENCE = "rejected_low_confidence"
    REVIEW_REQUIRED = "review_required"
    ACCEPTED = "accepted"


class SpeechReliabilityReason(str, Enum):
    TRANSCRIPT_NOT_ATTEMPTED = "transcript_not_attempted"
    BELOW_REVIEW_THRESHOLD = "below_review_threshold"
    BELOW_ACCEPTANCE_THRESHOLD = "below_acceptance_threshold"
    MEETS_ACCEPTANCE_THRESHOLD = "meets_acceptance_threshold"


class SpeechContractRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class SpeechModelDescriptor(SpeechContractRecord):
    """Minimum model provenance needed to reproduce a VAD or ASR result."""

    model_id: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=128)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    runtime_distribution: str = Field(min_length=1, max_length=128)
    runtime_version: str = Field(min_length=1, max_length=128)


class SpeechEvidenceSource(SpeechContractRecord):
    """Privacy-minimized link back to an authorized prepared-audio manifest."""

    preparation_manifest_path: str
    preparation_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    clip_id: Identifier
    consent_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    processing_scope: Literal["acoustic_and_speech"] = "acoustic_and_speech"
    sample_rate_hz: int = Field(ge=8_000, le=192_000, strict=True)
    num_samples: int = Field(gt=0, strict=True)

    _validate_manifest_path = field_validator("preparation_manifest_path")(
        validate_relative_audio_path
    )

    @model_validator(mode="after")
    def validate_manifest_suffix(self) -> "SpeechEvidenceSource":
        if PurePosixPath(self.preparation_manifest_path).suffix.lower() != ".json":
            raise ValueError("preparation_manifest_path must end in .json")
        return self


class SpeechWindowReference(SpeechContractRecord):
    """Prepared waveform window that was available to the speech branch."""

    window_id: Identifier
    audio_path: str
    window_audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(gt=0, strict=True)
    padding_samples: int = Field(ge=0, strict=True)

    _validate_audio_path = field_validator("audio_path")(validate_relative_audio_path)

    @model_validator(mode="after")
    def validate_window(self) -> "SpeechWindowReference":
        if self.end_sample <= self.start_sample:
            raise ValueError("window end_sample must be greater than start_sample")
        if PurePosixPath(self.audio_path).suffix.lower() != ".wav":
            raise ValueError("speech window audio_path must end in .wav")
        return self


class TranscriptCandidate(SpeechContractRecord):
    """One ASR hypothesis plus a clearly typed normalized confidence signal."""

    text: str = Field(min_length=1, max_length=20_000)
    confidence_score: float = Field(ge=0, le=1)
    confidence_kind: TranscriptConfidenceKind
    language: str | None = Field(default=None, pattern=r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
    language_confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcript text must contain a non-whitespace character")
        if value != value.strip():
            raise ValueError("transcript text must not have leading or trailing whitespace")
        if any(character in value for character in ("\x00", "\r")):
            raise ValueError("transcript text contains a disallowed control character")
        return value

    @model_validator(mode="after")
    def validate_language_fields(self) -> "TranscriptCandidate":
        if (self.language is None) != (self.language_confidence is None):
            raise ValueError("language and language_confidence must be present together")
        return self


class SpeechReliabilityAssessment(SpeechContractRecord):
    reliability: TranscriptReliability
    reason_codes: tuple[SpeechReliabilityReason, ...] = Field(min_length=1, max_length=1)
    downstream_text_allowed: bool
    human_review_required: bool


class SpeechReliabilityPolicy(SpeechContractRecord):
    """Versioned gates; scores are evidence signals, not claims that text is correct."""

    policy_version: Literal["1.0"] = SPEECH_RELIABILITY_POLICY_VERSION
    vad_speech_threshold: float = Field(default=0.60, ge=0, le=1)
    transcript_review_threshold: float = Field(default=0.50, ge=0, le=1)
    transcript_acceptance_threshold: float = Field(default=0.80, ge=0, le=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "SpeechReliabilityPolicy":
        if self.transcript_review_threshold >= self.transcript_acceptance_threshold:
            raise ValueError(
                "transcript_review_threshold must be below transcript_acceptance_threshold"
            )
        return self

    def assess(self, transcript: TranscriptCandidate | None) -> SpeechReliabilityAssessment:
        """Apply the recorded transcript thresholds with explicit boundary behavior."""

        if transcript is None:
            return SpeechReliabilityAssessment(
                reliability=TranscriptReliability.NOT_TRANSCRIBED,
                reason_codes=(SpeechReliabilityReason.TRANSCRIPT_NOT_ATTEMPTED,),
                downstream_text_allowed=False,
                human_review_required=False,
            )
        if transcript.confidence_score < self.transcript_review_threshold:
            return SpeechReliabilityAssessment(
                reliability=TranscriptReliability.REJECTED_LOW_CONFIDENCE,
                reason_codes=(SpeechReliabilityReason.BELOW_REVIEW_THRESHOLD,),
                downstream_text_allowed=False,
                human_review_required=False,
            )
        if transcript.confidence_score < self.transcript_acceptance_threshold:
            return SpeechReliabilityAssessment(
                reliability=TranscriptReliability.REVIEW_REQUIRED,
                reason_codes=(SpeechReliabilityReason.BELOW_ACCEPTANCE_THRESHOLD,),
                downstream_text_allowed=False,
                human_review_required=True,
            )
        return SpeechReliabilityAssessment(
            reliability=TranscriptReliability.ACCEPTED,
            reason_codes=(SpeechReliabilityReason.MEETS_ACCEPTANCE_THRESHOLD,),
            downstream_text_allowed=True,
            human_review_required=False,
        )


class SpeechSegmentEvidence(SpeechContractRecord):
    """Sample-exact VAD-positive span and its optional transcript candidate."""

    segment_id: Identifier
    source_window_ids: tuple[Identifier, ...] = Field(min_length=1)
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(gt=0, strict=True)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    vad_score: float = Field(
        ge=0,
        le=1,
        description="Normalized VAD evidence score; not a calibrated correctness probability.",
    )
    transcript: TranscriptCandidate | None = None
    assessment: SpeechReliabilityAssessment

    @model_validator(mode="after")
    def validate_segment(self) -> "SpeechSegmentEvidence":
        if self.end_sample <= self.start_sample:
            raise ValueError("segment end_sample must be greater than start_sample")
        if len(set(self.source_window_ids)) != len(self.source_window_ids):
            raise ValueError("source_window_ids must be unique")
        return self


class SpeechEvidenceDocument(SpeechContractRecord):
    """Candidate speech evidence; deliberately not language, incident, or risk output."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/speech-evidence.schema.json"
        },
    )

    schema_version: Literal["1.0"] = SPEECH_EVIDENCE_SCHEMA_VERSION
    document_type: Literal["speech_evidence_candidates"] = "speech_evidence_candidates"
    evidence_id: Identifier
    created_at: datetime
    source: SpeechEvidenceSource
    reliability_policy: SpeechReliabilityPolicy
    vad_model: SpeechModelDescriptor
    transcription_model: SpeechModelDescriptor | None = None
    input_window_count: int = Field(ge=0, strict=True)
    segment_count: int = Field(ge=0, strict=True)
    transcribed_segment_count: int = Field(ge=0, strict=True)
    windows: tuple[SpeechWindowReference, ...]
    segments: tuple[SpeechSegmentEvidence, ...]

    @model_validator(mode="after")
    def validate_document(self) -> "SpeechEvidenceDocument":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.input_window_count != len(self.windows):
            raise ValueError("input_window_count must equal the window inventory")
        if self.segment_count != len(self.segments):
            raise ValueError("segment_count must equal the segment inventory")
        if self.transcribed_segment_count != sum(
            segment.transcript is not None for segment in self.segments
        ):
            raise ValueError("transcribed_segment_count must equal segments with transcripts")
        if any(segment.transcript is not None for segment in self.segments):
            if self.transcription_model is None:
                raise ValueError("transcribed segments require transcription_model provenance")

        window_ids = [window.window_id for window in self.windows]
        if len(set(window_ids)) != len(window_ids):
            raise ValueError("window IDs must be unique")
        if list(self.windows) != sorted(
            self.windows, key=lambda window: (window.start_sample, window.end_sample, window.window_id)
        ):
            raise ValueError("windows must use deterministic temporal ordering")
        windows = {window.window_id: window for window in self.windows}
        for window in self.windows:
            if window.end_sample > self.source.num_samples:
                raise ValueError("window span cannot exceed the prepared source")

        segment_ids = [segment.segment_id for segment in self.segments]
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("segment IDs must be unique")
        if list(self.segments) != sorted(
            self.segments, key=lambda segment: (
                segment.start_sample,
                segment.end_sample,
                segment.segment_id,
            )
        ):
            raise ValueError("segments must use deterministic temporal ordering")

        previous_end = 0
        for segment in self.segments:
            if segment.end_sample > self.source.num_samples:
                raise ValueError("segment span cannot exceed the prepared source")
            if segment.start_sample < previous_end:
                raise ValueError("speech segments must not overlap")
            previous_end = segment.end_sample
            self._validate_seconds(segment.start_sample, segment.start_seconds)
            self._validate_seconds(segment.end_sample, segment.end_seconds)
            if segment.vad_score < self.reliability_policy.vad_speech_threshold:
                raise ValueError("recorded speech segment is below vad_speech_threshold")
            referenced_windows: list[SpeechWindowReference] = []
            for window_id in segment.source_window_ids:
                window = windows.get(window_id)
                if window is None:
                    raise ValueError("segment references an unknown source window")
                if segment.end_sample <= window.start_sample or segment.start_sample >= window.end_sample:
                    raise ValueError("segment must overlap each referenced source window")
                referenced_windows.append(window)
            ordered_references = sorted(
                referenced_windows,
                key=lambda window: (window.start_sample, window.end_sample, window.window_id),
            )
            if segment.source_window_ids != tuple(window.window_id for window in ordered_references):
                raise ValueError("source_window_ids must use deterministic temporal ordering")
            coverage_end = segment.start_sample
            for window in ordered_references:
                if window.start_sample > coverage_end:
                    raise ValueError("referenced source windows must cover the complete segment")
                coverage_end = max(coverage_end, window.end_sample)
            if coverage_end < segment.end_sample:
                raise ValueError("referenced source windows must cover the complete segment")
            if segment.assessment != self.reliability_policy.assess(segment.transcript):
                raise ValueError("segment assessment does not match the recorded reliability policy")
        return self

    def _validate_seconds(self, sample: int, seconds: float) -> None:
        expected = sample / self.source.sample_rate_hz
        if not math.isclose(seconds, expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError("segment seconds must equal sample offsets / sample_rate_hz")


def speech_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public JSON Schema for the v1 speech-evidence contract."""

    return {"speech-evidence.schema.json": SpeechEvidenceDocument.model_json_schema()}


def write_speech_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the public speech contract for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in speech_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
