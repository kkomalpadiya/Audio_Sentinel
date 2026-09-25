"""A6.1 risk-assessment inputs, scoring bands, and missing-data policy."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.contracts import EventLabel, ProcessingScope, RiskLevel
from audio_sentinel.language_contracts import LanguageCategory, LanguageReasonCode
from audio_sentinel.preparation import Identifier


RISK_ASSESSMENT_SCHEMA_VERSION = "1.0"
RISK_SCORING_POLICY_VERSION = "1.0"
RISK_RULE_FORMAT_VERSION = "1.0"


class RiskEvidenceKind(str, Enum):
    ACOUSTIC = "acoustic"
    SPEECH = "speech"
    LANGUAGE = "language"


class RiskInputStatus(str, Enum):
    """Whether a branch is usable for scoring, absent, or outside consent scope."""

    PRESENT = "present"
    MISSING = "missing"
    NOT_PERMITTED = "not_permitted"
    NOT_APPLICABLE = "not_applicable"
    NO_ACCEPTED_TEXT = "no_accepted_text"


class RiskReasonCode(str, Enum):
    """Portable explanations for a risk score or missing-data review state."""

    NO_RISK_EVIDENCE = "no_risk_evidence"
    ACOUSTIC_EVENT_CANDIDATE = "acoustic_event_candidate"
    ACOUSTIC_HIGH_CONFIDENCE = "acoustic_high_confidence"
    SPEECH_PRESENT = "speech_present"
    TRANSCRIPT_REVIEW_REQUIRED = "transcript_review_required"
    LANGUAGE_DISTRESS = "language_distress"
    LANGUAGE_THREAT = "language_threat"
    LANGUAGE_WEAPON_REFERENCE = "language_weapon_reference"
    LANGUAGE_AMBIGUOUS = "language_ambiguous"
    LANGUAGE_CONTEXT_SUPPRESSED = "language_context_suppressed"
    MISSING_ACOUSTIC_EVIDENCE = "missing_acoustic_evidence"
    MISSING_SPEECH_EVIDENCE = "missing_speech_evidence"
    MISSING_LANGUAGE_EVIDENCE = "missing_language_evidence"
    SPEECH_NOT_PERMITTED = "speech_not_permitted"
    LANGUAGE_NOT_APPLICABLE = "language_not_applicable"
    MISSING_DATA_REVIEW_REQUIRED = "missing_data_review_required"


_REASON_ORDER = {reason: index for index, reason in enumerate(RiskReasonCode)}
_EVIDENCE_DOCUMENT_TYPES = {
    RiskEvidenceKind.ACOUSTIC: "acoustic_event_candidates",
    RiskEvidenceKind.SPEECH: "speech_evidence_candidates",
    RiskEvidenceKind.LANGUAGE: "language_evidence",
}
_MISSING_REASON = {
    RiskEvidenceKind.ACOUSTIC: RiskReasonCode.MISSING_ACOUSTIC_EVIDENCE,
    RiskEvidenceKind.SPEECH: RiskReasonCode.MISSING_SPEECH_EVIDENCE,
    RiskEvidenceKind.LANGUAGE: RiskReasonCode.MISSING_LANGUAGE_EVIDENCE,
}


class RiskContractRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class RiskEvidenceReference(RiskContractRecord):
    """Hash-pinned reference to a Phase 3, 4, or 5 evidence artifact."""

    kind: RiskEvidenceKind
    document_type: Literal[
        "acoustic_event_candidates",
        "speech_evidence_candidates",
        "language_evidence",
    ]
    evidence_id: Identifier
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_document_type(self) -> "RiskEvidenceReference":
        if self.document_type != _EVIDENCE_DOCUMENT_TYPES[self.kind]:
            raise ValueError("risk evidence kind does not match document_type")
        return self


class RiskSource(RiskContractRecord):
    """Privacy-minimized clip identity shared by all risk branches."""

    clip_id: Identifier
    consent_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    processing_scope: ProcessingScope
    sample_rate_hz: int = Field(ge=8_000, le=192_000, strict=True)
    num_samples: int = Field(gt=0, strict=True)


class AcousticRiskSignal(RiskContractRecord):
    """One acoustic candidate event summarized for risk scoring."""

    label: EventLabel
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(gt=0, strict=True)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    peak_score: float = Field(ge=0, le=1)
    source_event_index: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_span(self) -> "AcousticRiskSignal":
        if self.end_sample <= self.start_sample:
            raise ValueError("acoustic signal end_sample must be greater than start_sample")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("acoustic signal end_seconds must be greater than start_seconds")
        return self


class AcousticRiskInputs(RiskContractRecord):
    status: RiskInputStatus
    evidence: RiskEvidenceReference | None = None
    event_count: int = Field(default=0, ge=0, strict=True)
    max_peak_score: float | None = Field(default=None, ge=0, le=1)
    signals: tuple[AcousticRiskSignal, ...] = ()

    @model_validator(mode="after")
    def validate_acoustic_inputs(self) -> "AcousticRiskInputs":
        _validate_status_reference(self.status, self.evidence, RiskEvidenceKind.ACOUSTIC)
        if self.status is not RiskInputStatus.PRESENT:
            if self.event_count or self.max_peak_score is not None or self.signals:
                raise ValueError("non-present acoustic input cannot carry scoring signals")
            return self
        if self.event_count != len(self.signals):
            raise ValueError("acoustic event_count must equal the signal inventory")
        expected_peak = max((signal.peak_score for signal in self.signals), default=None)
        if expected_peak is None:
            if self.max_peak_score is not None:
                raise ValueError("empty acoustic signals require null max_peak_score")
        elif not math.isclose(self.max_peak_score or -1, expected_peak, rel_tol=0, abs_tol=1e-12):
            raise ValueError("acoustic max_peak_score must equal the signal maximum")
        return self


class SpeechRiskInputs(RiskContractRecord):
    status: RiskInputStatus
    evidence: RiskEvidenceReference | None = None
    segment_count: int = Field(default=0, ge=0, strict=True)
    accepted_transcript_count: int = Field(default=0, ge=0, strict=True)
    review_required_transcript_count: int = Field(default=0, ge=0, strict=True)
    max_vad_score: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_speech_inputs(self) -> "SpeechRiskInputs":
        _validate_status_reference(self.status, self.evidence, RiskEvidenceKind.SPEECH)
        if self.status is not RiskInputStatus.PRESENT:
            if (
                self.segment_count
                or self.accepted_transcript_count
                or self.review_required_transcript_count
                or self.max_vad_score is not None
            ):
                raise ValueError("non-present speech input cannot carry scoring counts")
            return self
        if self.accepted_transcript_count > self.segment_count:
            raise ValueError("accepted_transcript_count cannot exceed segment_count")
        if self.review_required_transcript_count > self.segment_count:
            raise ValueError("review_required_transcript_count cannot exceed segment_count")
        if self.segment_count == 0 and self.max_vad_score is not None:
            raise ValueError("zero speech segments require null max_vad_score")
        if self.segment_count > 0 and self.max_vad_score is None:
            raise ValueError("speech segments require max_vad_score")
        return self


class LanguageRiskSignal(RiskContractRecord):
    """One language finding summarized without transcript text."""

    finding_id: Identifier
    segment_id: Identifier
    category: LanguageCategory
    reason_codes: tuple[LanguageReasonCode, ...] = Field(min_length=1)
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(gt=0, strict=True)
    rule_match_count: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_language_signal(self) -> "LanguageRiskSignal":
        if self.end_sample <= self.start_sample:
            raise ValueError("language signal end_sample must be greater than start_sample")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("language reason_codes must be unique")
        if list(self.reason_codes) != sorted(
            self.reason_codes, key=lambda reason: list(LanguageReasonCode).index(reason)
        ):
            raise ValueError("language reason_codes must use canonical ordering")
        return self


class LanguageRiskInputs(RiskContractRecord):
    status: RiskInputStatus
    evidence: RiskEvidenceReference | None = None
    finding_count: int = Field(default=0, ge=0, strict=True)
    signals: tuple[LanguageRiskSignal, ...] = ()

    @model_validator(mode="after")
    def validate_language_inputs(self) -> "LanguageRiskInputs":
        _validate_status_reference(self.status, self.evidence, RiskEvidenceKind.LANGUAGE)
        if self.status is not RiskInputStatus.PRESENT:
            if self.finding_count or self.signals:
                raise ValueError("non-present language input cannot carry scoring signals")
            return self
        if self.finding_count != len(self.signals):
            raise ValueError("language finding_count must equal the signal inventory")
        ids = [signal.finding_id for signal in self.signals]
        if len(set(ids)) != len(ids):
            raise ValueError("language finding IDs must be unique")
        return self


class RiskInputSet(RiskContractRecord):
    """All evidence summaries made available to the risk scorer."""

    source: RiskSource
    acoustic: AcousticRiskInputs
    speech: SpeechRiskInputs
    language: LanguageRiskInputs

    @model_validator(mode="after")
    def validate_input_set(self) -> "RiskInputSet":
        if self.source.processing_scope is ProcessingScope.ACOUSTIC_ONLY:
            if self.speech.status is RiskInputStatus.PRESENT:
                raise ValueError("speech input cannot be present for acoustic_only consent")
            if self.language.status is RiskInputStatus.PRESENT:
                raise ValueError("language input cannot be present for acoustic_only consent")
        if self.language.status is RiskInputStatus.PRESENT and self.speech.status is not RiskInputStatus.PRESENT:
            raise ValueError("language input requires present speech input provenance")
        if (
            self.language.status is RiskInputStatus.NO_ACCEPTED_TEXT
            and self.speech.accepted_transcript_count != 0
        ):
            raise ValueError("no_accepted_text requires zero accepted transcripts")
        previous: tuple[int, int, str] | None = None
        for signal in self.acoustic.signals:
            _validate_seconds(self.source.sample_rate_hz, signal.start_sample, signal.start_seconds)
            _validate_seconds(self.source.sample_rate_hz, signal.end_sample, signal.end_seconds)
            if signal.end_sample > self.source.num_samples:
                raise ValueError("acoustic signal span cannot exceed the source")
            order = (signal.start_sample, signal.end_sample, signal.label.value)
            if previous is not None and order < previous:
                raise ValueError("acoustic signals must use deterministic temporal ordering")
            previous = order
        previous_language: tuple[int, int, str] | None = None
        for signal in self.language.signals:
            if signal.end_sample > self.source.num_samples:
                raise ValueError("language signal span cannot exceed the source")
            order = (signal.start_sample, signal.end_sample, signal.finding_id)
            if previous_language is not None and order < previous_language:
                raise ValueError("language signals must use deterministic temporal ordering")
            previous_language = order
        return self


class RiskSeverityBandDefinition(RiskContractRecord):
    risk_level: RiskLevel
    min_score: float = Field(ge=0, le=100)
    max_score: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_band(self) -> "RiskSeverityBandDefinition":
        if self.max_score < self.min_score:
            raise ValueError("risk band max_score must be greater than or equal to min_score")
        return self


def default_risk_severity_bands() -> tuple[RiskSeverityBandDefinition, ...]:
    """Return the v1 0-100 severity bands with inclusive upper bounds."""

    return (
        RiskSeverityBandDefinition(risk_level=RiskLevel.NONE, min_score=0, max_score=0),
        RiskSeverityBandDefinition(risk_level=RiskLevel.LOW, min_score=1, max_score=24),
        RiskSeverityBandDefinition(risk_level=RiskLevel.MEDIUM, min_score=25, max_score=49),
        RiskSeverityBandDefinition(risk_level=RiskLevel.HIGH, min_score=50, max_score=74),
        RiskSeverityBandDefinition(risk_level=RiskLevel.CRITICAL, min_score=75, max_score=100),
    )


def severity_for_score(score: float) -> RiskLevel:
    """Map a validated 0-100 score to the default v1 severity band."""

    if not 0 <= score <= 100 or not math.isfinite(score):
        raise ValueError("risk score must be finite and between 0 and 100")
    if score == 0:
        return RiskLevel.NONE
    if score <= 24:
        return RiskLevel.LOW
    if score <= 49:
        return RiskLevel.MEDIUM
    if score <= 74:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


class RiskMissingDataPolicy(RiskContractRecord):
    """How Phase 6 treats absent branches without calling them safe."""

    policy_version: Literal["1.0"] = RISK_SCORING_POLICY_VERSION
    required_branches: tuple[RiskEvidenceKind, ...] = (RiskEvidenceKind.ACOUSTIC,)
    speech_required_when_permitted: bool = True
    language_required_when_accepted_text_exists: bool = True
    missing_required_branch_action: Literal["review_required"] = "review_required"

    @model_validator(mode="after")
    def validate_policy(self) -> "RiskMissingDataPolicy":
        if len(set(self.required_branches)) != len(self.required_branches):
            raise ValueError("required_branches must be unique")
        if list(self.required_branches) != sorted(
            self.required_branches, key=lambda item: list(RiskEvidenceKind).index(item)
        ):
            raise ValueError("required_branches must use canonical ordering")
        return self


class RiskMissingDataAssessment(RiskContractRecord):
    missing_branches: tuple[RiskEvidenceKind, ...] = ()
    not_permitted_branches: tuple[RiskEvidenceKind, ...] = ()
    review_required: bool
    reason_codes: tuple[RiskReasonCode, ...] = ()

    @model_validator(mode="after")
    def validate_missing_data(self) -> "RiskMissingDataAssessment":
        _validate_kind_tuple("missing_branches", self.missing_branches)
        _validate_kind_tuple("not_permitted_branches", self.not_permitted_branches)
        if set(self.missing_branches) & set(self.not_permitted_branches):
            raise ValueError("a branch cannot be both missing and not permitted")
        _validate_reason_tuple(self.reason_codes)
        if self.review_required and RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED not in self.reason_codes:
            raise ValueError("review_required needs missing_data_review_required reason")
        if self.missing_branches and not self.review_required:
            raise ValueError("missing branches require review_required")
        return self


class RiskRuleSetDescriptor(RiskContractRecord):
    """Exact configurable rule artifact used to calculate a risk score."""

    rule_set_id: Identifier
    rule_set_version: str = Field(min_length=1, max_length=128)
    rule_format_version: Literal["1.0"] = RISK_RULE_FORMAT_VERSION
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class RiskScoringPolicy(RiskContractRecord):
    """Versioned score range and severity-band definition for B6.1."""

    policy_version: Literal["1.0"] = RISK_SCORING_POLICY_VERSION
    score_min: Literal[0] = 0
    score_max: Literal[100] = 100
    rule_set: RiskRuleSetDescriptor
    severity_bands: tuple[RiskSeverityBandDefinition, ...] = Field(
        default_factory=default_risk_severity_bands
    )
    missing_data_policy: RiskMissingDataPolicy = Field(default_factory=RiskMissingDataPolicy)

    @model_validator(mode="after")
    def validate_scoring_policy(self) -> "RiskScoringPolicy":
        expected_levels = tuple(RiskLevel)
        if tuple(band.risk_level for band in self.severity_bands) != expected_levels:
            raise ValueError("risk severity bands must use canonical risk-level order")
        expected = default_risk_severity_bands()
        if self.severity_bands != expected:
            raise ValueError("v1 risk severity bands must match the documented defaults")
        return self


class RiskAssessmentDocument(RiskContractRecord):
    """Versioned risk score output; not a consensus decision or alert."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/risk-assessment.schema.json"
        },
    )

    schema_version: Literal["1.0"] = RISK_ASSESSMENT_SCHEMA_VERSION
    document_type: Literal["risk_assessment"] = "risk_assessment"
    assessment_id: Identifier
    created_at: datetime
    scoring_policy: RiskScoringPolicy
    inputs: RiskInputSet
    score: float = Field(ge=0, le=100)
    severity: RiskLevel
    reason_codes: tuple[RiskReasonCode, ...] = Field(min_length=1)
    missing_data: RiskMissingDataAssessment
    human_review_required: bool

    @model_validator(mode="after")
    def validate_document(self) -> "RiskAssessmentDocument":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.severity is not severity_for_score(self.score):
            raise ValueError("severity must match the configured score band")
        _validate_reason_tuple(self.reason_codes)
        expected_missing = _branches_with_status(self.inputs, RiskInputStatus.MISSING)
        expected_not_permitted = _branches_with_status(
            self.inputs, RiskInputStatus.NOT_PERMITTED
        )
        if self.missing_data.missing_branches != expected_missing:
            raise ValueError("missing_data missing_branches must match input statuses")
        if self.missing_data.not_permitted_branches != expected_not_permitted:
            raise ValueError("missing_data not_permitted_branches must match input statuses")
        for branch in expected_missing:
            if _MISSING_REASON[branch] not in self.reason_codes:
                raise ValueError("missing branch reason is absent from assessment reasons")
        if self.missing_data.review_required and not self.human_review_required:
            raise ValueError("missing-data review must set human_review_required")
        if self.human_review_required and RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED in self.reason_codes:
            if not self.missing_data.review_required:
                raise ValueError("missing-data review reason must match missing_data state")
        return self


def _validate_status_reference(
    status: RiskInputStatus,
    evidence: RiskEvidenceReference | None,
    expected_kind: RiskEvidenceKind,
) -> None:
    if status is RiskInputStatus.PRESENT:
        if evidence is None:
            raise ValueError("present risk input requires evidence reference")
        if evidence.kind is not expected_kind:
            raise ValueError("risk input evidence has the wrong kind")
    elif evidence is not None:
        raise ValueError("non-present risk input cannot include evidence reference")


def _validate_seconds(sample_rate_hz: int, sample: int, seconds: float) -> None:
    if not math.isclose(seconds, sample / sample_rate_hz, rel_tol=0, abs_tol=1e-9):
        raise ValueError("risk signal seconds must equal sample offsets / sample_rate_hz")


def _validate_reason_tuple(reasons: tuple[RiskReasonCode, ...]) -> None:
    if len(set(reasons)) != len(reasons):
        raise ValueError("risk reason_codes must be unique")
    if list(reasons) != sorted(reasons, key=lambda reason: _REASON_ORDER[reason]):
        raise ValueError("risk reason_codes must use canonical ordering")


def _validate_kind_tuple(field_name: str, values: tuple[RiskEvidenceKind, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")
    if list(values) != sorted(values, key=lambda item: list(RiskEvidenceKind).index(item)):
        raise ValueError(f"{field_name} must use canonical ordering")


def _branches_with_status(
    inputs: RiskInputSet, status: RiskInputStatus
) -> tuple[RiskEvidenceKind, ...]:
    pairs = (
        (RiskEvidenceKind.ACOUSTIC, inputs.acoustic.status),
        (RiskEvidenceKind.SPEECH, inputs.speech.status),
        (RiskEvidenceKind.LANGUAGE, inputs.language.status),
    )
    return tuple(kind for kind, branch_status in pairs if branch_status is status)


def risk_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public JSON Schema for the v1 risk-assessment contract."""

    return {"risk-assessment.schema.json": RiskAssessmentDocument.model_json_schema()}


def write_risk_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the public risk-assessment contract for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in risk_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
