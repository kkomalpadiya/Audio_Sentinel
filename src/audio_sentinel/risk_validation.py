"""A6.4 executable acceptance validation for the deterministic v1 risk policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.contracts import EventLabel, ProcessingScope, RiskLevel
from audio_sentinel.language_contracts import LanguageCategory, LanguageReasonCode
from audio_sentinel.preparation import Identifier
from audio_sentinel.risk_contracts import (
    AcousticRiskInputs,
    AcousticRiskSignal,
    LanguageRiskInputs,
    LanguageRiskSignal,
    RiskAssessmentDocument,
    RiskEvidenceKind,
    RiskEvidenceReference,
    RiskInputSet,
    RiskInputStatus,
    RiskReasonCode,
    RiskRuleSetDescriptor,
    RiskSeverityBandDefinition,
    RiskSource,
    SpeechRiskInputs,
    default_risk_severity_bands,
    severity_for_score,
)
from audio_sentinel.risk_scoring import (
    BUILTIN_RISK_RULE_SET_SHA256,
    LoadedRiskRuleSet,
    load_builtin_risk_rule_set,
    score_risk,
)


RISK_VALIDATION_SCHEMA_VERSION = "1.0"
BUILTIN_RISK_VALIDATION_SUITE_ID = "audio-sentinel-risk-validation-v1"
BUILTIN_RISK_VALIDATION_SUITE_VERSION = "1.0.0"
BUILTIN_RISK_VALIDATION_RESOURCE = "risk-validation-scenarios-v1.json"
BUILTIN_RISK_VALIDATION_SHA256 = (
    "bd23748af8f9e403705e005714b1b369ea4ce20a67db009fac69d5c5d2e49884"
)
MAX_RISK_VALIDATION_BYTES = 131_072


class RiskValidationError(RuntimeError):
    """Stable validation failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RiskValidationRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class RiskValidationCoverage(str, Enum):
    SEVERITY_NONE = "severity_none"
    SEVERITY_LOW = "severity_low"
    SEVERITY_MEDIUM = "severity_medium"
    SEVERITY_HIGH = "severity_high"
    SEVERITY_CRITICAL = "severity_critical"
    ACOUSTIC_CONFIDENCE_BELOW = "acoustic_confidence_below"
    ACOUSTIC_CONFIDENCE_AT = "acoustic_confidence_at"
    HUMAN_REVIEW_BELOW_THRESHOLD = "human_review_below_threshold"
    HUMAN_REVIEW_AT_THRESHOLD = "human_review_at_threshold"
    SPEECH_CAP = "speech_cap"
    LANGUAGE_CAP = "language_cap"
    TOTAL_CAP = "total_cap"
    AMBIGUOUS_LANGUAGE_REVIEW = "ambiguous_language_review"
    CONTEXT_SUPPRESSED_ZERO_WEIGHT = "context_suppressed_zero_weight"
    MISSING_DATA_REVIEW = "missing_data_review"
    CONSENT_LIMITED_NOT_MISSING = "consent_limited_not_missing"
    NO_ACCEPTED_TEXT_NOT_MISSING = "no_accepted_text_not_missing"


class AcousticValidationSignal(RiskValidationRecord):
    label: EventLabel
    peak_score: float = Field(ge=0, le=1)


class AcousticValidationInput(RiskValidationRecord):
    status: RiskInputStatus = RiskInputStatus.PRESENT
    signals: tuple[AcousticValidationSignal, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> "AcousticValidationInput":
        if self.status not in {RiskInputStatus.PRESENT, RiskInputStatus.MISSING}:
            raise ValueError("validation acoustic status must be present or missing")
        if self.status is not RiskInputStatus.PRESENT and self.signals:
            raise ValueError(
                "non-present acoustic validation input cannot carry signals"
            )
        return self


class SpeechValidationInput(RiskValidationRecord):
    status: RiskInputStatus = RiskInputStatus.PRESENT
    segment_count: int = Field(default=0, ge=0, strict=True)
    accepted_transcript_count: int = Field(default=0, ge=0, strict=True)
    review_required_transcript_count: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_counts(self) -> "SpeechValidationInput":
        if self.status not in {
            RiskInputStatus.PRESENT,
            RiskInputStatus.MISSING,
            RiskInputStatus.NOT_PERMITTED,
        }:
            raise ValueError("unsupported speech validation status")
        total_transcripts = (
            self.accepted_transcript_count
            + self.review_required_transcript_count
        )
        if self.status is not RiskInputStatus.PRESENT and (
            self.segment_count or total_transcripts
        ):
            raise ValueError("non-present speech validation input cannot carry counts")
        if total_transcripts > self.segment_count:
            raise ValueError("validation transcript counts exceed speech segments")
        return self


class LanguageValidationInput(RiskValidationRecord):
    status: RiskInputStatus = RiskInputStatus.PRESENT
    categories: tuple[LanguageCategory, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> "LanguageValidationInput":
        if self.status not in {
            RiskInputStatus.PRESENT,
            RiskInputStatus.MISSING,
            RiskInputStatus.NOT_PERMITTED,
            RiskInputStatus.NO_ACCEPTED_TEXT,
        }:
            raise ValueError("unsupported language validation status")
        if self.status is not RiskInputStatus.PRESENT and self.categories:
            raise ValueError(
                "non-present language validation input cannot carry findings"
            )
        return self


class RiskValidationOutcome(RiskValidationRecord):
    score: float = Field(ge=0, le=100)
    severity: RiskLevel
    human_review_required: bool
    reason_codes: tuple[RiskReasonCode, ...] = Field(min_length=1)
    missing_branches: tuple[RiskEvidenceKind, ...] = ()

    @model_validator(mode="after")
    def validate_outcome(self) -> "RiskValidationOutcome":
        if self.severity is not severity_for_score(self.score):
            raise ValueError("validation severity must match the score")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("validation reasons must be unique")
        if self.reason_codes != tuple(
            sorted(self.reason_codes, key=lambda item: list(RiskReasonCode).index(item))
        ):
            raise ValueError("validation reasons must use canonical ordering")
        if len(set(self.missing_branches)) != len(self.missing_branches):
            raise ValueError("validation missing branches must be unique")
        if self.missing_branches != tuple(
            sorted(
                self.missing_branches,
                key=lambda item: list(RiskEvidenceKind).index(item),
            )
        ):
            raise ValueError("validation missing branches must use canonical ordering")
        return self


class RiskValidationCase(RiskValidationRecord):
    case_id: Identifier
    description: str = Field(min_length=1, max_length=256)
    coverage: tuple[RiskValidationCoverage, ...] = Field(min_length=1)
    processing_scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH
    acoustic: AcousticValidationInput = Field(default_factory=AcousticValidationInput)
    speech: SpeechValidationInput = Field(default_factory=SpeechValidationInput)
    language: LanguageValidationInput = Field(default_factory=LanguageValidationInput)
    expected: RiskValidationOutcome

    @model_validator(mode="after")
    def validate_case(self) -> "RiskValidationCase":
        if len(set(self.coverage)) != len(self.coverage):
            raise ValueError("validation case coverage markers must be unique")
        if self.processing_scope is ProcessingScope.NONE:
            raise ValueError("risk validation requires an authorized processing scope")
        if self.processing_scope is ProcessingScope.ACOUSTIC_ONLY and (
            self.speech.status is not RiskInputStatus.NOT_PERMITTED
            or self.language.status is not RiskInputStatus.NOT_PERMITTED
        ):
            raise ValueError("acoustic-only validation requires not-permitted branches")
        if (
            self.language.status is RiskInputStatus.PRESENT
            and self.speech.status is not RiskInputStatus.PRESENT
        ):
            raise ValueError("present validation language requires present speech")
        if self.language.categories and not self.speech.accepted_transcript_count:
            raise ValueError(
                "validation language findings require accepted transcripts"
            )
        if (
            self.language.status is RiskInputStatus.NO_ACCEPTED_TEXT
            and self.speech.accepted_transcript_count
        ):
            raise ValueError(
                "no accepted text conflicts with accepted transcript count"
            )
        return self


class RiskValidationSuite(RiskValidationRecord):
    schema_version: Literal["1.0"] = RISK_VALIDATION_SCHEMA_VERSION
    suite_id: Identifier
    suite_version: str = Field(min_length=1, max_length=128)
    rule_set_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    cases: tuple[RiskValidationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_suite(self) -> "RiskValidationSuite":
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("risk validation case IDs must be unique")
        covered = {marker for case in self.cases for marker in case.coverage}
        if covered != set(RiskValidationCoverage):
            raise ValueError("risk validation suite must cover every required behavior")
        return self


@dataclass(frozen=True)
class LoadedRiskValidationSuite:
    suite: RiskValidationSuite
    artifact_sha256: str
    artifact_size_bytes: int


class RiskPolicyThresholdSnapshot(RiskValidationRecord):
    acoustic_high_confidence_threshold: float = Field(ge=0, le=1)
    human_review_minimum_score: float = Field(gt=0, le=100)
    acoustic_max_points: float = Field(ge=0, le=100)
    speech_max_points: float = Field(ge=0, le=100)
    language_max_points: float = Field(ge=0, le=100)
    total_max_score: float = Field(ge=0, le=100)
    severity_bands: tuple[RiskSeverityBandDefinition, ...]


class RiskValidationCaseResult(RiskValidationRecord):
    case_id: Identifier
    coverage: tuple[RiskValidationCoverage, ...]
    expected: RiskValidationOutcome
    observed: RiskValidationOutcome
    passed: bool

    @model_validator(mode="after")
    def validate_result(self) -> "RiskValidationCaseResult":
        if self.passed is not (self.expected == self.observed):
            raise ValueError("validation pass flag differs from compared outcomes")
        return self


class RiskValidationReport(RiskValidationRecord):
    schema_version: Literal["1.0"] = RISK_VALIDATION_SCHEMA_VERSION
    document_type: Literal["risk_policy_validation"] = "risk_policy_validation"
    validation_id: Identifier
    created_at: datetime
    decision_status: Literal[
        "validated_for_deterministic_v1_scenarios",
        "validation_failed",
    ]
    suite_id: Identifier
    suite_version: str
    suite_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rule_set: RiskRuleSetDescriptor
    thresholds: RiskPolicyThresholdSnapshot
    coverage: tuple[RiskValidationCoverage, ...]
    scenario_count: int = Field(gt=0, strict=True)
    passed_count: int = Field(ge=0, strict=True)
    failed_count: int = Field(ge=0, strict=True)
    cases: tuple[RiskValidationCaseResult, ...]
    limitation: Literal[
        "synthetic_acceptance_validation_not_empirical_incident_calibration"
    ] = "synthetic_acceptance_validation_not_empirical_incident_calibration"

    @model_validator(mode="after")
    def validate_report(self) -> "RiskValidationReport":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("risk validation time must be timezone-aware")
        if self.scenario_count != len(self.cases):
            raise ValueError("validation scenario count differs from case results")
        if self.passed_count != sum(case.passed for case in self.cases):
            raise ValueError("validation passed count differs from case results")
        if self.failed_count != self.scenario_count - self.passed_count:
            raise ValueError("validation failed count differs from case results")
        expected_status = (
            "validated_for_deterministic_v1_scenarios"
            if self.failed_count == 0
            else "validation_failed"
        )
        if self.decision_status != expected_status:
            raise ValueError("validation decision differs from case results")
        if self.coverage != tuple(RiskValidationCoverage):
            raise ValueError("validation report coverage must use canonical order")
        return self


_LANGUAGE_REASONS = {
    LanguageCategory.NO_CONCERNING_MATCH: (LanguageReasonCode.NO_RULE_MATCH,),
    LanguageCategory.DISTRESS: (LanguageReasonCode.KEYWORD_MATCH,),
    LanguageCategory.THREAT: (LanguageReasonCode.KEYWORD_MATCH,),
    LanguageCategory.WEAPON_REFERENCE: (LanguageReasonCode.KEYWORD_MATCH,),
    LanguageCategory.AMBIGUOUS: (
        LanguageReasonCode.KEYWORD_MATCH,
        LanguageReasonCode.INSUFFICIENT_CONTEXT,
    ),
    LanguageCategory.CONTEXT_SUPPRESSED: (
        LanguageReasonCode.KEYWORD_MATCH,
        LanguageReasonCode.EXPLICIT_NEGATION,
    ),
}
_DOCUMENT_TYPES = {
    RiskEvidenceKind.ACOUSTIC: "acoustic_event_candidates",
    RiskEvidenceKind.SPEECH: "speech_evidence_candidates",
    RiskEvidenceKind.LANGUAGE: "language_evidence",
}


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def load_risk_validation_suite_bytes(
    raw: bytes,
    *,
    expected_sha256: str | None = None,
) -> LoadedRiskValidationSuite:
    """Validate bounded suite JSON and optionally pin its exact bytes."""

    if not isinstance(raw, bytes):
        raise TypeError("risk validation suite must be bytes")
    if not raw:
        raise ValueError("risk validation suite must not be empty")
    if len(raw) > MAX_RISK_VALIDATION_BYTES:
        raise ValueError("risk validation suite exceeds the byte limit")
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and artifact_sha256 != expected_sha256:
        raise ValueError("risk validation suite checksum mismatch")
    try:
        suite = RiskValidationSuite.model_validate_json(raw)
    except Exception as error:
        raise ValueError("risk validation suite failed contract validation") from error
    return LoadedRiskValidationSuite(
        suite=suite,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=len(raw),
    )


def load_builtin_risk_validation_suite() -> LoadedRiskValidationSuite:
    """Load the pinned A6.4 acceptance suite without network access."""

    raw = (
        files("audio_sentinel")
        .joinpath(f"resources/{BUILTIN_RISK_VALIDATION_RESOURCE}")
        .read_bytes()
    )
    loaded = load_risk_validation_suite_bytes(
        raw,
        expected_sha256=BUILTIN_RISK_VALIDATION_SHA256,
    )
    if (
        loaded.suite.suite_id != BUILTIN_RISK_VALIDATION_SUITE_ID
        or loaded.suite.suite_version != BUILTIN_RISK_VALIDATION_SUITE_VERSION
        or loaded.suite.rule_set_sha256 != BUILTIN_RISK_RULE_SET_SHA256
    ):
        raise ValueError("bundled risk validation identity differs from specification")
    return loaded


def _evidence_reference(
    case_id: str,
    kind: RiskEvidenceKind,
) -> RiskEvidenceReference:
    return RiskEvidenceReference(
        kind=kind,
        document_type=_DOCUMENT_TYPES[kind],
        evidence_id=f"{case_id}-{kind.value}",
        evidence_sha256=_canonical_hash({"case_id": case_id, "kind": kind.value}),
    )


def build_risk_validation_inputs(case: RiskValidationCase) -> RiskInputSet:
    """Expand one compact validation case through the public A6.1 contract."""

    if not isinstance(case, RiskValidationCase):
        raise TypeError("risk validation case has an invalid type")
    validated = RiskValidationCase.model_validate(case.model_dump(mode="python"))
    acoustic_signals = tuple(
        AcousticRiskSignal(
            label=signal.label,
            start_sample=index * 2_000,
            end_sample=index * 2_000 + 1_000,
            start_seconds=index * 2_000 / 16_000,
            end_seconds=(index * 2_000 + 1_000) / 16_000,
            peak_score=signal.peak_score,
            source_event_index=index,
        )
        for index, signal in enumerate(validated.acoustic.signals)
    )
    acoustic = AcousticRiskInputs(
        status=validated.acoustic.status,
        evidence=(
            _evidence_reference(validated.case_id, RiskEvidenceKind.ACOUSTIC)
            if validated.acoustic.status is RiskInputStatus.PRESENT
            else None
        ),
        event_count=len(acoustic_signals),
        max_peak_score=max(
            (signal.peak_score for signal in acoustic_signals),
            default=None,
        ),
        signals=acoustic_signals,
    )
    speech = SpeechRiskInputs(
        status=validated.speech.status,
        evidence=(
            _evidence_reference(validated.case_id, RiskEvidenceKind.SPEECH)
            if validated.speech.status is RiskInputStatus.PRESENT
            else None
        ),
        segment_count=validated.speech.segment_count,
        accepted_transcript_count=validated.speech.accepted_transcript_count,
        review_required_transcript_count=(
            validated.speech.review_required_transcript_count
        ),
        max_vad_score=(0.9 if validated.speech.segment_count else None),
    )
    language_signals = tuple(
        LanguageRiskSignal(
            finding_id=f"{validated.case_id}-finding-{index:02d}",
            segment_id=f"{validated.case_id}-segment-{index:02d}",
            category=category,
            reason_codes=_LANGUAGE_REASONS[category],
            start_sample=40_000 + index * 2_000,
            end_sample=41_000 + index * 2_000,
            rule_match_count=(
                0 if category is LanguageCategory.NO_CONCERNING_MATCH else 1
            ),
        )
        for index, category in enumerate(validated.language.categories)
    )
    language = LanguageRiskInputs(
        status=validated.language.status,
        evidence=(
            _evidence_reference(validated.case_id, RiskEvidenceKind.LANGUAGE)
            if validated.language.status is RiskInputStatus.PRESENT
            else None
        ),
        finding_count=len(language_signals),
        signals=language_signals,
    )
    return RiskInputSet(
        source=RiskSource(
            clip_id=f"{validated.case_id}-clip",
            consent_id="risk-validation-consent-v1",
            processing_scope=validated.processing_scope,
            sample_rate_hz=16_000,
            num_samples=160_000,
        ),
        acoustic=acoustic,
        speech=speech,
        language=language,
    )


def _observed_outcome(assessment: RiskAssessmentDocument) -> RiskValidationOutcome:
    return RiskValidationOutcome(
        score=assessment.score,
        severity=assessment.severity,
        human_review_required=assessment.human_review_required,
        reason_codes=assessment.reason_codes,
        missing_branches=assessment.missing_data.missing_branches,
    )


def validate_risk_policy(
    suite: LoadedRiskValidationSuite,
    rule_set: LoadedRiskRuleSet,
    *,
    now: datetime | None = None,
) -> RiskValidationReport:
    """Run a pinned scenario suite through the public deterministic scorer."""

    if not isinstance(suite, LoadedRiskValidationSuite):
        raise RiskValidationError(
            "invalid_suite", "A loaded validation suite is required."
        )
    if not isinstance(rule_set, LoadedRiskRuleSet):
        raise RiskValidationError(
            "invalid_rule_set", "A loaded risk rule set is required."
        )
    try:
        validated_suite = RiskValidationSuite.model_validate(
            suite.suite.model_dump(mode="python")
        )
    except Exception as error:
        raise RiskValidationError(
            "invalid_suite", "Risk validation suite integrity validation failed."
        ) from error
    if (
        not re.fullmatch(r"[a-f0-9]{64}", suite.artifact_sha256)
        or not isinstance(suite.artifact_size_bytes, int)
        or isinstance(suite.artifact_size_bytes, bool)
        or not 0 < suite.artifact_size_bytes <= MAX_RISK_VALIDATION_BYTES
    ):
        raise RiskValidationError(
            "invalid_suite", "Risk validation suite provenance is invalid."
        )
    if validated_suite.rule_set_sha256 != rule_set.artifact_sha256:
        raise RiskValidationError(
            "rule_set_mismatch",
            "Risk validation suite targets a different rule artifact.",
        )
    created_at = now if now is not None else datetime.now(UTC)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise RiskValidationError(
            "invalid_time", "Risk validation time must be timezone-aware."
        )
    created_at = created_at.astimezone(UTC)

    results = []
    for case in validated_suite.cases:
        assessment = score_risk(
            build_risk_validation_inputs(case),
            rule_set=rule_set,
            now=created_at,
        )
        observed = _observed_outcome(assessment)
        results.append(
            RiskValidationCaseResult(
                case_id=case.case_id,
                coverage=case.coverage,
                expected=case.expected,
                observed=observed,
                passed=case.expected == observed,
            )
        )
    case_results = tuple(results)
    passed_count = sum(result.passed for result in case_results)
    thresholds = RiskPolicyThresholdSnapshot(
        acoustic_high_confidence_threshold=(
            rule_set.rule_set.acoustic.high_confidence_threshold
        ),
        human_review_minimum_score=rule_set.rule_set.human_review.minimum_score,
        acoustic_max_points=rule_set.rule_set.acoustic.max_points,
        speech_max_points=rule_set.rule_set.speech.max_points,
        language_max_points=rule_set.rule_set.language.max_points,
        total_max_score=rule_set.rule_set.total_max_score,
        severity_bands=default_risk_severity_bands(),
    )
    identity_payload = {
        "suite_id": validated_suite.suite_id,
        "suite_version": validated_suite.suite_version,
        "suite_sha256": suite.artifact_sha256,
        "rule_set": rule_set.as_descriptor().model_dump(mode="json"),
        "thresholds": thresholds.model_dump(mode="json"),
        "cases": [result.model_dump(mode="json") for result in case_results],
    }
    return RiskValidationReport(
        validation_id=f"a6-4-{_canonical_hash(identity_payload)}",
        created_at=created_at,
        decision_status=(
            "validated_for_deterministic_v1_scenarios"
            if passed_count == len(case_results)
            else "validation_failed"
        ),
        suite_id=validated_suite.suite_id,
        suite_version=validated_suite.suite_version,
        suite_sha256=suite.artifact_sha256,
        rule_set=rule_set.as_descriptor(),
        thresholds=thresholds,
        coverage=tuple(RiskValidationCoverage),
        scenario_count=len(case_results),
        passed_count=passed_count,
        failed_count=len(case_results) - passed_count,
        cases=case_results,
    )


def validate_builtin_risk_policy(
    *,
    now: datetime | None = None,
) -> RiskValidationReport:
    """Validate the exact bundled suite against the exact bundled rules."""

    return validate_risk_policy(
        load_builtin_risk_validation_suite(),
        load_builtin_risk_rule_set(),
        now=now,
    )


def save_risk_validation_report(
    report: RiskValidationReport,
    destination: Path,
) -> None:
    """Write a validated report without replacing an existing result."""

    if not isinstance(report, RiskValidationReport):
        raise TypeError("risk validation report has an invalid type")
    validated = RiskValidationReport.model_validate(report.model_dump(mode="python"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(validated.model_dump_json(indent=2) + "\n")
    except FileExistsError as error:
        raise RiskValidationError(
            "output_exists", "Risk validation output already exists."
        ) from error
