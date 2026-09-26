"""A7.4 end-to-end acceptance validation for deterministic consensus."""

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

from audio_sentinel.consensus_contracts import (
    BranchAgreement,
    CONSENSUS_POLICY_VERSION,
    ConsensusOutcome,
    ConsensusPolicy,
    ConsensusReasonCode,
)
from audio_sentinel.consensus_rules import (
    AgreementRuleSetDescriptor,
    LoadedAgreementRuleSet,
    evaluate_evidence_agreement,
    load_builtin_agreement_rule_set,
)
from audio_sentinel.consensus_service import decide_consensus
from audio_sentinel.contracts import EventLabel, ProcessingScope, RiskLevel
from audio_sentinel.language_contracts import LanguageCategory, LanguageReasonCode
from audio_sentinel.preparation import Identifier
from audio_sentinel.risk_contracts import (
    AcousticRiskInputs,
    AcousticRiskSignal,
    LanguageRiskInputs,
    LanguageRiskSignal,
    RiskEvidenceKind,
    RiskEvidenceReference,
    RiskInputSet,
    RiskInputStatus,
    RiskRuleSetDescriptor,
    RiskSource,
    SpeechRiskInputs,
)
from audio_sentinel.risk_scoring import (
    LoadedRiskRuleSet,
    load_builtin_risk_rule_set,
    score_risk,
)
from audio_sentinel.risk_validation import (
    AcousticValidationInput,
    LanguageValidationInput,
    SpeechValidationInput,
)


CONSENSUS_VALIDATION_SCHEMA_VERSION = "1.0"
BUILTIN_CONSENSUS_VALIDATION_SUITE_ID = (
    "audio-sentinel-consensus-validation-v1"
)
BUILTIN_CONSENSUS_VALIDATION_SUITE_VERSION = "1.0.0"
BUILTIN_CONSENSUS_VALIDATION_RESOURCE = (
    "consensus-validation-scenarios-v1.json"
)
BUILTIN_CONSENSUS_VALIDATION_SHA256 = (
    "bef7dcd275b2f21b5c7e07de9337ce60d4789f8182f1b1e01feb9ab375d7bf63"
)
MAX_CONSENSUS_VALIDATION_BYTES = 196_608


class ConsensusValidationError(RuntimeError):
    """Stable validation failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ConsensusValidationRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class ConsensusValidationCoverage(str, Enum):
    OUTCOME_NO_ACTION = "outcome_no_action"
    OUTCOME_LOG = "outcome_log"
    OUTCOME_REVIEW = "outcome_review"
    OUTCOME_ALERT = "outcome_alert"
    MULTI_BRANCH_ALERT = "multi_branch_alert"
    SUPPORT_THRESHOLD_BELOW = "support_threshold_below"
    SUPPORT_THRESHOLD_AT = "support_threshold_at"
    SINGLE_BRANCH_ALERT_BLOCK = "single_branch_alert_block"
    ACOUSTIC_NO_SPEECH_CONFLICT = "acoustic_no_speech_conflict"
    SPEECH_ABSENCE_CONFLICT = "speech_absence_conflict"
    NON_THREATENING_LANGUAGE_CONFLICT = (
        "non_threatening_language_conflict"
    )
    LANGUAGE_WITHOUT_TRANSCRIPT_CONFLICT = (
        "language_without_transcript_conflict"
    )
    UNCERTAINTY_BLOCKS_ALERT = "uncertainty_blocks_alert"
    MISSING_EVIDENCE_REVIEW = "missing_evidence_review"
    CONSENT_LIMITED_REVIEW = "consent_limited_review"
    SAFE_LANGUAGE_NEUTRAL = "safe_language_neutral"


class ConsensusBranchValidationOutcome(ConsensusValidationRecord):
    kind: RiskEvidenceKind
    input_status: RiskInputStatus
    agreement: BranchAgreement


class ConsensusValidationOutcome(ConsensusValidationRecord):
    risk_score: float = Field(ge=0, le=100)
    risk_severity: RiskLevel
    risk_review_required: bool
    branches: tuple[ConsensusBranchValidationOutcome, ...] = Field(
        min_length=3,
        max_length=3,
    )
    supporting_branches: tuple[RiskEvidenceKind, ...] = ()
    conflicting_branches: tuple[RiskEvidenceKind, ...] = ()
    outcome: ConsensusOutcome
    reason_codes: tuple[ConsensusReasonCode, ...] = Field(min_length=1)
    review_required: bool
    alert_candidate: bool

    @model_validator(mode="after")
    def validate_outcome(self) -> "ConsensusValidationOutcome":
        if tuple(branch.kind for branch in self.branches) != tuple(
            RiskEvidenceKind
        ):
            raise ValueError(
                "validation branches must use complete canonical ordering"
            )
        expected_support = tuple(
            branch.kind
            for branch in self.branches
            if branch.agreement is BranchAgreement.SUPPORTS_RISK
        )
        expected_conflicts = tuple(
            branch.kind
            for branch in self.branches
            if branch.agreement is BranchAgreement.CONFLICTS_RISK
        )
        if self.supporting_branches != expected_support:
            raise ValueError(
                "validation supporting branches differ from branch states"
            )
        if self.conflicting_branches != expected_conflicts:
            raise ValueError(
                "validation conflicting branches differ from branch states"
            )
        if self.review_required is not (
            self.outcome in {ConsensusOutcome.REVIEW, ConsensusOutcome.ALERT}
        ):
            raise ValueError(
                "validation review flag must match review or alert outcome"
            )
        if self.alert_candidate is not (
            self.outcome is ConsensusOutcome.ALERT
        ):
            raise ValueError(
                "validation alert flag must match the alert outcome"
            )
        return self


class ConsensusValidationCase(ConsensusValidationRecord):
    case_id: Identifier
    description: str = Field(min_length=1, max_length=256)
    coverage: tuple[ConsensusValidationCoverage, ...] = Field(min_length=1)
    processing_scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH
    acoustic: AcousticValidationInput = Field(
        default_factory=AcousticValidationInput
    )
    speech: SpeechValidationInput = Field(
        default_factory=SpeechValidationInput
    )
    language: LanguageValidationInput = Field(
        default_factory=LanguageValidationInput
    )
    expected: ConsensusValidationOutcome

    @model_validator(mode="after")
    def validate_case(self) -> "ConsensusValidationCase":
        if len(set(self.coverage)) != len(self.coverage):
            raise ValueError("validation case coverage markers must be unique")
        if self.processing_scope is ProcessingScope.NONE:
            raise ValueError(
                "consensus validation requires an authorized processing scope"
            )
        if self.processing_scope is ProcessingScope.ACOUSTIC_ONLY and (
            self.speech.status is not RiskInputStatus.NOT_PERMITTED
            or self.language.status is not RiskInputStatus.NOT_PERMITTED
        ):
            raise ValueError(
                "acoustic-only validation requires not-permitted branches"
            )
        if (
            self.language.status is RiskInputStatus.PRESENT
            and self.speech.status is not RiskInputStatus.PRESENT
        ):
            raise ValueError(
                "present validation language requires present speech"
            )
        if (
            self.language.status is RiskInputStatus.NO_ACCEPTED_TEXT
            and self.speech.accepted_transcript_count
        ):
            raise ValueError(
                "no accepted text conflicts with accepted transcript count"
            )
        return self


class ConsensusValidationSuite(ConsensusValidationRecord):
    schema_version: Literal["1.0"] = CONSENSUS_VALIDATION_SCHEMA_VERSION
    suite_id: Identifier
    suite_version: str = Field(min_length=1, max_length=128)
    risk_rule_set_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    agreement_rule_set_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    consensus_policy_version: Literal["1.0"] = CONSENSUS_POLICY_VERSION
    cases: tuple[ConsensusValidationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_suite(self) -> "ConsensusValidationSuite":
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("consensus validation case IDs must be unique")
        covered = {marker for case in self.cases for marker in case.coverage}
        if covered != set(ConsensusValidationCoverage):
            raise ValueError(
                "consensus validation suite must cover every required behavior"
            )
        return self


@dataclass(frozen=True)
class LoadedConsensusValidationSuite:
    suite: ConsensusValidationSuite
    artifact_sha256: str
    artifact_size_bytes: int


class ConsensusValidationCaseResult(ConsensusValidationRecord):
    case_id: Identifier
    coverage: tuple[ConsensusValidationCoverage, ...]
    expected: ConsensusValidationOutcome
    observed: ConsensusValidationOutcome
    passed: bool

    @model_validator(mode="after")
    def validate_result(self) -> "ConsensusValidationCaseResult":
        if self.passed is not (self.expected == self.observed):
            raise ValueError("validation pass flag differs from compared outcomes")
        return self


class ConsensusValidationReport(ConsensusValidationRecord):
    schema_version: Literal["1.0"] = CONSENSUS_VALIDATION_SCHEMA_VERSION
    document_type: Literal["consensus_pipeline_validation"] = (
        "consensus_pipeline_validation"
    )
    validation_id: Identifier
    created_at: datetime
    decision_status: Literal[
        "validated_for_deterministic_v1_scenarios",
        "validation_failed",
    ]
    suite_id: Identifier
    suite_version: str
    suite_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    risk_rule_set: RiskRuleSetDescriptor
    agreement_rule_set: AgreementRuleSetDescriptor
    consensus_policy: ConsensusPolicy
    coverage: tuple[ConsensusValidationCoverage, ...]
    scenario_count: int = Field(gt=0, strict=True)
    passed_count: int = Field(ge=0, strict=True)
    failed_count: int = Field(ge=0, strict=True)
    cases: tuple[ConsensusValidationCaseResult, ...]
    limitation: Literal[
        "synthetic_acceptance_validation_not_empirical_incident_accuracy"
    ] = "synthetic_acceptance_validation_not_empirical_incident_accuracy"

    @model_validator(mode="after")
    def validate_report(self) -> "ConsensusValidationReport":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("consensus validation time must be timezone-aware")
        if self.scenario_count != len(self.cases):
            raise ValueError(
                "validation scenario count differs from case results"
            )
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
        if self.coverage != tuple(ConsensusValidationCoverage):
            raise ValueError(
                "validation report coverage must use canonical order"
            )
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


def load_consensus_validation_suite_bytes(
    raw: bytes,
    *,
    expected_sha256: str | None = None,
) -> LoadedConsensusValidationSuite:
    """Validate bounded suite JSON and optionally pin its exact bytes."""

    if not isinstance(raw, bytes):
        raise TypeError("consensus validation suite must be bytes")
    if not raw:
        raise ValueError("consensus validation suite must not be empty")
    if len(raw) > MAX_CONSENSUS_VALIDATION_BYTES:
        raise ValueError("consensus validation suite exceeds the byte limit")
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and artifact_sha256 != expected_sha256:
        raise ValueError("consensus validation suite checksum mismatch")
    try:
        suite = ConsensusValidationSuite.model_validate_json(raw)
    except Exception as error:
        raise ValueError(
            "consensus validation suite failed contract validation"
        ) from error
    return LoadedConsensusValidationSuite(
        suite=suite,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=len(raw),
    )


def load_builtin_consensus_validation_suite() -> (
    LoadedConsensusValidationSuite
):
    """Load the pinned A7.4 acceptance suite without network access."""

    raw = (
        files("audio_sentinel")
        .joinpath(f"resources/{BUILTIN_CONSENSUS_VALIDATION_RESOURCE}")
        .read_bytes()
    )
    loaded = load_consensus_validation_suite_bytes(
        raw,
        expected_sha256=BUILTIN_CONSENSUS_VALIDATION_SHA256,
    )
    if (
        loaded.suite.suite_id
        != BUILTIN_CONSENSUS_VALIDATION_SUITE_ID
        or loaded.suite.suite_version
        != BUILTIN_CONSENSUS_VALIDATION_SUITE_VERSION
    ):
        raise ValueError(
            "bundled consensus validation identity differs from specification"
        )
    return loaded


def _evidence_reference(
    case_id: str,
    kind: RiskEvidenceKind,
) -> RiskEvidenceReference:
    return RiskEvidenceReference(
        kind=kind,
        document_type=_DOCUMENT_TYPES[kind],
        evidence_id=f"{case_id}-{kind.value}",
        evidence_sha256=_canonical_hash(
            {"case_id": case_id, "kind": kind.value}
        ),
    )


def build_consensus_validation_inputs(
    case: ConsensusValidationCase,
) -> RiskInputSet:
    """Expand one compact case through the public Phase 6 input contract."""

    if not isinstance(case, ConsensusValidationCase):
        raise TypeError("consensus validation case has an invalid type")
    validated = ConsensusValidationCase.model_validate(
        case.model_dump(mode="python")
    )
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
        accepted_transcript_count=(
            validated.speech.accepted_transcript_count
        ),
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
                0
                if category is LanguageCategory.NO_CONCERNING_MATCH
                else 1
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
            consent_id="consensus-validation-consent-v1",
            processing_scope=validated.processing_scope,
            sample_rate_hz=16_000,
            num_samples=160_000,
        ),
        acoustic=acoustic,
        speech=speech,
        language=language,
    )


def _observed_outcome(assessment, agreement, decision) -> (
    ConsensusValidationOutcome
):
    return ConsensusValidationOutcome(
        risk_score=assessment.score,
        risk_severity=assessment.severity,
        risk_review_required=assessment.human_review_required,
        branches=tuple(
            ConsensusBranchValidationOutcome(
                kind=branch.kind,
                input_status=branch.input_status,
                agreement=branch.agreement,
            )
            for branch in agreement.branches
        ),
        supporting_branches=agreement.supporting_branches,
        conflicting_branches=agreement.conflicting_branches,
        outcome=decision.outcome,
        reason_codes=decision.reason_codes,
        review_required=decision.review_required,
        alert_candidate=decision.alert_candidate,
    )


def validate_consensus_pipeline(
    suite: LoadedConsensusValidationSuite,
    risk_rule_set: LoadedRiskRuleSet,
    agreement_rule_set: LoadedAgreementRuleSet,
    *,
    policy: ConsensusPolicy | None = None,
    now: datetime | None = None,
) -> ConsensusValidationReport:
    """Run pinned cases through scorer, agreement, and final verification."""

    if not isinstance(suite, LoadedConsensusValidationSuite):
        raise ConsensusValidationError(
            "invalid_suite", "A loaded consensus validation suite is required."
        )
    if not isinstance(risk_rule_set, LoadedRiskRuleSet):
        raise ConsensusValidationError(
            "invalid_risk_rule_set", "A loaded risk rule set is required."
        )
    if not isinstance(agreement_rule_set, LoadedAgreementRuleSet):
        raise ConsensusValidationError(
            "invalid_agreement_rule_set",
            "A loaded agreement rule set is required.",
        )
    try:
        validated_suite = ConsensusValidationSuite.model_validate(
            suite.suite.model_dump(mode="python")
        )
        validated_policy = ConsensusPolicy.model_validate(
            (policy or ConsensusPolicy()).model_dump(mode="python")
        )
    except Exception as error:
        raise ConsensusValidationError(
            "invalid_suite", "Consensus validation integrity check failed."
        ) from error
    if (
        not re.fullmatch(r"[a-f0-9]{64}", suite.artifact_sha256)
        or not isinstance(suite.artifact_size_bytes, int)
        or isinstance(suite.artifact_size_bytes, bool)
        or not 0 < suite.artifact_size_bytes <= MAX_CONSENSUS_VALIDATION_BYTES
    ):
        raise ConsensusValidationError(
            "invalid_suite", "Consensus validation provenance is invalid."
        )
    if validated_suite.risk_rule_set_sha256 != risk_rule_set.artifact_sha256:
        raise ConsensusValidationError(
            "risk_rule_set_mismatch",
            "Consensus validation targets a different risk rule artifact.",
        )
    if (
        validated_suite.agreement_rule_set_sha256
        != agreement_rule_set.artifact_sha256
    ):
        raise ConsensusValidationError(
            "agreement_rule_set_mismatch",
            "Consensus validation targets a different agreement artifact.",
        )
    if (
        validated_suite.consensus_policy_version
        != validated_policy.policy_version
    ):
        raise ConsensusValidationError(
            "policy_mismatch",
            "Consensus validation targets a different policy version.",
        )
    created_at = now if now is not None else datetime.now(UTC)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ConsensusValidationError(
            "invalid_time", "Consensus validation time must be timezone-aware."
        )
    created_at = created_at.astimezone(UTC)

    results = []
    for case in validated_suite.cases:
        assessment = score_risk(
            build_consensus_validation_inputs(case),
            rule_set=risk_rule_set,
            now=created_at,
        )
        agreement = evaluate_evidence_agreement(
            assessment,
            rule_set=agreement_rule_set,
            now=created_at,
        )
        decision = decide_consensus(
            assessment,
            agreement,
            policy=validated_policy,
            trusted_rule_set=agreement_rule_set,
            now=created_at,
        )
        observed = _observed_outcome(assessment, agreement, decision)
        results.append(
            ConsensusValidationCaseResult(
                case_id=case.case_id,
                coverage=case.coverage,
                expected=case.expected,
                observed=observed,
                passed=case.expected == observed,
            )
        )
    case_results = tuple(results)
    passed_count = sum(result.passed for result in case_results)
    identity_payload = {
        "suite_id": validated_suite.suite_id,
        "suite_version": validated_suite.suite_version,
        "suite_sha256": suite.artifact_sha256,
        "risk_rule_set": risk_rule_set.as_descriptor().model_dump(mode="json"),
        "agreement_rule_set": agreement_rule_set.as_descriptor().model_dump(
            mode="json"
        ),
        "consensus_policy": validated_policy.model_dump(mode="json"),
        "cases": [
            result.model_dump(mode="json") for result in case_results
        ],
    }
    return ConsensusValidationReport(
        validation_id=f"a7-4-{_canonical_hash(identity_payload)}",
        created_at=created_at,
        decision_status=(
            "validated_for_deterministic_v1_scenarios"
            if passed_count == len(case_results)
            else "validation_failed"
        ),
        suite_id=validated_suite.suite_id,
        suite_version=validated_suite.suite_version,
        suite_sha256=suite.artifact_sha256,
        risk_rule_set=risk_rule_set.as_descriptor(),
        agreement_rule_set=agreement_rule_set.as_descriptor(),
        consensus_policy=validated_policy,
        coverage=tuple(ConsensusValidationCoverage),
        scenario_count=len(case_results),
        passed_count=passed_count,
        failed_count=len(case_results) - passed_count,
        cases=case_results,
    )


def validate_builtin_consensus_pipeline(
    *,
    now: datetime | None = None,
) -> ConsensusValidationReport:
    """Validate the exact bundled suite against all bundled Phase 6-7 rules."""

    return validate_consensus_pipeline(
        load_builtin_consensus_validation_suite(),
        load_builtin_risk_rule_set(),
        load_builtin_agreement_rule_set(),
        now=now,
    )


def save_consensus_validation_report(
    report: ConsensusValidationReport,
    destination: Path,
) -> None:
    """Write a validated report without replacing an existing result."""

    if not isinstance(report, ConsensusValidationReport):
        raise TypeError("consensus validation report has an invalid type")
    validated = ConsensusValidationReport.model_validate(
        report.model_dump(mode="python")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(validated.model_dump_json(indent=2) + "\n")
    except FileExistsError as error:
        raise ConsensusValidationError(
            "output_exists", "Consensus validation output already exists."
        ) from error
