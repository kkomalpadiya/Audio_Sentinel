"""B7.1 deterministic evidence-agreement and conflict-detection rules."""

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
    ConsensusBranchState,
)
from audio_sentinel.contracts import EventLabel
from audio_sentinel.language_contracts import LanguageCategory
from audio_sentinel.preparation import Identifier
from audio_sentinel.risk_contracts import (
    RiskAssessmentDocument,
    RiskEvidenceKind,
    RiskInputStatus,
)


AGREEMENT_RULE_SCHEMA_VERSION = "1.0"
AGREEMENT_RULE_FORMAT_VERSION = "1.0"
AGREEMENT_EVALUATION_SCHEMA_VERSION = "1.0"
BUILTIN_AGREEMENT_RULE_SET_ID = "audio-sentinel-consensus-agreement-v1"
BUILTIN_AGREEMENT_RULE_SET_VERSION = "1.0.0"
BUILTIN_AGREEMENT_RULE_RESOURCE = "consensus-agreement-rules-v1.json"
BUILTIN_AGREEMENT_RULE_SET_SHA256 = (
    "f2c913f921157f39a0e84b59c600e53f91c43828a9da9b83b84acaaf57d10d79"
)
MAX_AGREEMENT_RULE_SET_BYTES = 65_536


_EXPECTED_ACOUSTIC_SUPPORT_LABELS = tuple(
    label
    for label in EventLabel
    if label
    not in {
        EventLabel.AMBIENT,
        EventLabel.NO_SPEECH,
        EventLabel.SPEECH_PRESENT,
        EventLabel.NON_THREATENING_SPEECH,
    }
)
_EXPECTED_SPEECH_LIKE_LABELS = (
    EventLabel.SPEECH_PRESENT,
    EventLabel.NON_THREATENING_SPEECH,
    EventLabel.DISTRESS_SPEECH,
    EventLabel.THREATENING_SPEECH,
    EventLabel.WEAPON_REFERENCE,
)
_EXPECTED_LANGUAGE_SUPPORT_CATEGORIES = (
    LanguageCategory.DISTRESS,
    LanguageCategory.THREAT,
    LanguageCategory.WEAPON_REFERENCE,
)
_EXPECTED_LANGUAGE_NEUTRAL_CATEGORIES = (
    LanguageCategory.NO_CONCERNING_MATCH,
    LanguageCategory.AMBIGUOUS,
    LanguageCategory.CONTEXT_SUPPRESSED,
)


class AgreementReasonCode(str, Enum):
    """Portable explanations for one branch agreement classification."""

    BRANCH_MISSING = "branch_missing"
    BRANCH_NOT_PERMITTED = "branch_not_permitted"
    BRANCH_NOT_APPLICABLE = "branch_not_applicable"
    NO_RISK_SIGNAL = "no_risk_signal"
    LOW_CONFIDENCE_RISK_SIGNAL = "low_confidence_risk_signal"
    HIGH_CONFIDENCE_RISK_SIGNAL = "high_confidence_risk_signal"
    SPEECH_ABSENT = "speech_absent"
    SPEECH_PRESENT_NEUTRAL = "speech_present_neutral"
    SAFE_LANGUAGE_CONTEXT = "safe_language_context"
    AMBIGUOUS_LANGUAGE = "ambiguous_language"
    ACTIVE_LANGUAGE_RISK = "active_language_risk"
    ACOUSTIC_NO_SPEECH_CONFLICT = "acoustic_no_speech_conflict"
    ACOUSTIC_SPEECH_ABSENCE_CONFLICT = "acoustic_speech_absence_conflict"
    NON_THREATENING_LANGUAGE_CONFLICT = "non_threatening_language_conflict"
    LANGUAGE_WITHOUT_ACCEPTED_TRANSCRIPT_CONFLICT = (
        "language_without_accepted_transcript_conflict"
    )


_REASON_ORDER = {reason: index for index, reason in enumerate(AgreementReasonCode)}
_KIND_ORDER = {kind: index for index, kind in enumerate(RiskEvidenceKind)}
_CONFLICT_REASONS = frozenset(
    {
        AgreementReasonCode.ACOUSTIC_NO_SPEECH_CONFLICT,
        AgreementReasonCode.ACOUSTIC_SPEECH_ABSENCE_CONFLICT,
        AgreementReasonCode.NON_THREATENING_LANGUAGE_CONFLICT,
        AgreementReasonCode.LANGUAGE_WITHOUT_ACCEPTED_TRANSCRIPT_CONFLICT,
    }
)


class ConsensusAgreementError(RuntimeError):
    """Stable evaluation failure code plus a safe, non-sensitive message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class AgreementRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class AcousticSupportRules(AgreementRecord):
    minimum_peak_score: float = Field(ge=0, le=1)
    support_labels: tuple[EventLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_labels(self) -> "AcousticSupportRules":
        if self.support_labels != _EXPECTED_ACOUSTIC_SUPPORT_LABELS:
            raise ValueError("acoustic support labels must match the v1 risk taxonomy")
        return self


class SpeechConsistencyRules(AgreementRecord):
    minimum_peak_score: float = Field(ge=0, le=1)
    no_speech_label: Literal[EventLabel.NO_SPEECH] = EventLabel.NO_SPEECH
    speech_like_labels: tuple[EventLabel, ...] = Field(min_length=1)
    non_threatening_speech_label: Literal[EventLabel.NON_THREATENING_SPEECH] = (
        EventLabel.NON_THREATENING_SPEECH
    )

    @model_validator(mode="after")
    def validate_labels(self) -> "SpeechConsistencyRules":
        if self.speech_like_labels != _EXPECTED_SPEECH_LIKE_LABELS:
            raise ValueError("speech-like labels must match the v1 speech taxonomy")
        return self


class LanguageSupportRules(AgreementRecord):
    support_categories: tuple[LanguageCategory, ...] = Field(min_length=1)
    neutral_categories: tuple[LanguageCategory, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_categories(self) -> "LanguageSupportRules":
        if self.support_categories != _EXPECTED_LANGUAGE_SUPPORT_CATEGORIES:
            raise ValueError("language support categories must match the v1 taxonomy")
        if self.neutral_categories != _EXPECTED_LANGUAGE_NEUTRAL_CATEGORIES:
            raise ValueError("language neutral categories must match the v1 taxonomy")
        if set(self.support_categories) & set(self.neutral_categories):
            raise ValueError("language support and neutral categories must be disjoint")
        if set(self.support_categories) | set(self.neutral_categories) != set(
            LanguageCategory
        ):
            raise ValueError("language agreement rules must cover every category")
        return self


class ConflictRules(AgreementRecord):
    no_speech_vs_vad_speech: Literal[True] = True
    speech_signal_vs_zero_vad_segments: Literal[True] = True
    non_threatening_speech_vs_active_language: Literal[True] = True
    language_findings_require_accepted_transcript: Literal[True] = True


class AgreementRuleSet(AgreementRecord):
    """Versioned branch-support and contradiction rules for B7.1."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/consensus-agreement-rule-set.schema.json"
        },
    )

    schema_version: Literal["1.0"] = AGREEMENT_RULE_SCHEMA_VERSION
    rule_format_version: Literal["1.0"] = AGREEMENT_RULE_FORMAT_VERSION
    rule_set_id: Identifier
    rule_set_version: str = Field(min_length=1, max_length=128)
    acoustic_support: AcousticSupportRules
    speech_consistency: SpeechConsistencyRules
    language_support: LanguageSupportRules
    conflicts: ConflictRules


class AgreementRuleSetDescriptor(AgreementRecord):
    rule_set_id: Identifier
    rule_set_version: str = Field(min_length=1, max_length=128)
    rule_format_version: Literal["1.0"] = AGREEMENT_RULE_FORMAT_VERSION
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class LoadedAgreementRuleSet:
    """Validated rule artifact plus exact byte-level provenance."""

    rule_set: AgreementRuleSet
    artifact_sha256: str
    artifact_size_bytes: int

    def as_descriptor(self) -> AgreementRuleSetDescriptor:
        return AgreementRuleSetDescriptor(
            rule_set_id=self.rule_set.rule_set_id,
            rule_set_version=self.rule_set.rule_set_version,
            rule_format_version=self.rule_set.rule_format_version,
            artifact_sha256=self.artifact_sha256,
        )


class BranchAgreementEvaluation(AgreementRecord):
    kind: RiskEvidenceKind
    input_status: RiskInputStatus
    agreement: BranchAgreement
    reason_codes: tuple[AgreementReasonCode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evaluation(self) -> "BranchAgreementEvaluation":
        _validate_reason_tuple(self.reason_codes)
        ConsensusBranchState(
            kind=self.kind,
            input_status=self.input_status,
            agreement=self.agreement,
        )
        has_conflict_reason = bool(set(self.reason_codes) & _CONFLICT_REASONS)
        required_status_reason = {
            RiskInputStatus.MISSING: AgreementReasonCode.BRANCH_MISSING,
            RiskInputStatus.NOT_PERMITTED: AgreementReasonCode.BRANCH_NOT_PERMITTED,
            RiskInputStatus.NOT_APPLICABLE: AgreementReasonCode.BRANCH_NOT_APPLICABLE,
            RiskInputStatus.NO_ACCEPTED_TEXT: AgreementReasonCode.BRANCH_NOT_APPLICABLE,
        }.get(self.input_status)
        if required_status_reason is not None and self.reason_codes != (
            required_status_reason,
        ):
            raise ValueError("non-present branch requires its exact status reason")
        if self.agreement is BranchAgreement.CONFLICTS_RISK:
            if not has_conflict_reason:
                raise ValueError("conflicting branch requires a conflict reason")
        elif has_conflict_reason:
            raise ValueError("conflict reasons require a conflicting branch state")
        if self.agreement is BranchAgreement.SUPPORTS_RISK and not set(
            self.reason_codes
        ) & {
            AgreementReasonCode.HIGH_CONFIDENCE_RISK_SIGNAL,
            AgreementReasonCode.ACTIVE_LANGUAGE_RISK,
        }:
            raise ValueError("supporting branch requires an explicit support reason")
        return self

    def as_branch_state(self) -> ConsensusBranchState:
        return ConsensusBranchState(
            kind=self.kind,
            input_status=self.input_status,
            agreement=self.agreement,
        )


class ConsensusAgreementEvaluation(AgreementRecord):
    """Deterministic B7.1 result consumed by the later decision service."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/consensus-agreement.schema.json"
        },
    )

    schema_version: Literal["1.0"] = AGREEMENT_EVALUATION_SCHEMA_VERSION
    document_type: Literal["consensus_agreement"] = "consensus_agreement"
    evaluation_id: Identifier
    created_at: datetime
    assessment_id: Identifier
    assessment_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rule_set: AgreementRuleSetDescriptor
    branches: tuple[BranchAgreementEvaluation, ...] = Field(min_length=3, max_length=3)
    supporting_branches: tuple[RiskEvidenceKind, ...]
    conflicting_branches: tuple[RiskEvidenceKind, ...]
    has_conflict: bool

    @model_validator(mode="after")
    def validate_document(self) -> "ConsensusAgreementEvaluation":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if tuple(branch.kind for branch in self.branches) != tuple(RiskEvidenceKind):
            raise ValueError("agreement branches must use complete canonical ordering")
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
            raise ValueError("supporting_branches must match branch evaluations")
        if self.conflicting_branches != expected_conflicts:
            raise ValueError("conflicting_branches must match branch evaluations")
        if self.has_conflict is not bool(expected_conflicts):
            raise ValueError("has_conflict must match conflicting branches")
        if self.evaluation_id != _evaluation_id(
            assessment_id=self.assessment_id,
            assessment_sha256=self.assessment_sha256,
            rule_set=self.rule_set,
            branches=self.branches,
        ):
            raise ValueError("evaluation_id must match canonical agreement content")
        return self

    @property
    def branch_states(self) -> tuple[ConsensusBranchState, ...]:
        return tuple(branch.as_branch_state() for branch in self.branches)


def load_agreement_rule_set_bytes(
    raw: bytes,
    *,
    expected_sha256: str | None = None,
) -> LoadedAgreementRuleSet:
    """Validate bounded JSON rules and optionally require their exact digest."""

    if not isinstance(raw, bytes):
        raise TypeError("agreement rule artifact must be bytes")
    if not raw:
        raise ValueError("agreement rule artifact must not be empty")
    if len(raw) > MAX_AGREEMENT_RULE_SET_BYTES:
        raise ValueError("agreement rule artifact exceeds the byte limit")
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and artifact_sha256 != expected_sha256:
        raise ValueError("agreement rule artifact checksum mismatch")
    try:
        rule_set = AgreementRuleSet.model_validate_json(raw)
    except Exception as error:
        raise ValueError("agreement rule artifact failed contract validation") from error
    return LoadedAgreementRuleSet(
        rule_set=rule_set,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=len(raw),
    )


def load_builtin_agreement_rule_set() -> LoadedAgreementRuleSet:
    """Load the pinned local B7.1 rules without network or model runtimes."""

    raw = (
        files("audio_sentinel")
        .joinpath(f"resources/{BUILTIN_AGREEMENT_RULE_RESOURCE}")
        .read_bytes()
    )
    loaded = load_agreement_rule_set_bytes(
        raw,
        expected_sha256=BUILTIN_AGREEMENT_RULE_SET_SHA256,
    )
    if (
        loaded.rule_set.rule_set_id != BUILTIN_AGREEMENT_RULE_SET_ID
        or loaded.rule_set.rule_set_version != BUILTIN_AGREEMENT_RULE_SET_VERSION
    ):
        raise ValueError("bundled agreement rule identity differs from its pin")
    return loaded


def risk_assessment_sha256(assessment: RiskAssessmentDocument) -> str:
    """Return the canonical hash used to pin one validated Phase 6 assessment."""

    validated = _validate_assessment(assessment)
    return _canonical_hash(validated.model_dump(mode="json"))


def evaluate_evidence_agreement(
    assessment: RiskAssessmentDocument,
    *,
    rule_set: LoadedAgreementRuleSet | None = None,
    now: datetime | None = None,
) -> ConsensusAgreementEvaluation:
    """Classify support and conflicts without choosing a consensus outcome."""

    validated = _validate_assessment(assessment)
    loaded = _validate_loaded_rules(
        rule_set if rule_set is not None else load_builtin_agreement_rule_set()
    )
    created_at = _clock(now)
    rules = loaded.rule_set
    inputs = validated.inputs

    working: dict[RiskEvidenceKind, dict[str, object]] = {
        RiskEvidenceKind.ACOUSTIC: _base_acoustic_result(validated, rules),
        RiskEvidenceKind.SPEECH: _base_speech_result(validated),
        RiskEvidenceKind.LANGUAGE: _base_language_result(validated, rules),
    }

    _apply_cross_branch_conflicts(validated, rules, working)

    branches = tuple(
        BranchAgreementEvaluation(
            kind=kind,
            input_status=_branch_status(validated, kind),
            agreement=working[kind]["agreement"],
            reason_codes=tuple(
                sorted(working[kind]["reasons"], key=_REASON_ORDER.__getitem__)
            ),
        )
        for kind in RiskEvidenceKind
    )
    supporting = tuple(
        branch.kind
        for branch in branches
        if branch.agreement is BranchAgreement.SUPPORTS_RISK
    )
    conflicting = tuple(
        branch.kind
        for branch in branches
        if branch.agreement is BranchAgreement.CONFLICTS_RISK
    )
    assessment_hash = _canonical_hash(validated.model_dump(mode="json"))
    descriptor = loaded.as_descriptor()
    return ConsensusAgreementEvaluation(
        evaluation_id=_evaluation_id(
            assessment_id=validated.assessment_id,
            assessment_sha256=assessment_hash,
            rule_set=descriptor,
            branches=branches,
        ),
        created_at=created_at,
        assessment_id=validated.assessment_id,
        assessment_sha256=assessment_hash,
        rule_set=descriptor,
        branches=branches,
        supporting_branches=supporting,
        conflicting_branches=conflicting,
        has_conflict=bool(conflicting),
    )


def _base_acoustic_result(
    assessment: RiskAssessmentDocument,
    rules: AgreementRuleSet,
) -> dict[str, object]:
    branch = assessment.inputs.acoustic
    non_present = _non_present_result(branch.status)
    if non_present is not None:
        return non_present
    support_labels = set(rules.acoustic_support.support_labels)
    support = [signal for signal in branch.signals if signal.label in support_labels]
    if any(
        signal.peak_score >= rules.acoustic_support.minimum_peak_score
        for signal in support
    ):
        return _working(
            BranchAgreement.SUPPORTS_RISK,
            AgreementReasonCode.HIGH_CONFIDENCE_RISK_SIGNAL,
        )
    if support:
        return _working(
            BranchAgreement.NEUTRAL,
            AgreementReasonCode.LOW_CONFIDENCE_RISK_SIGNAL,
        )
    return _working(BranchAgreement.NEUTRAL, AgreementReasonCode.NO_RISK_SIGNAL)


def _base_speech_result(assessment: RiskAssessmentDocument) -> dict[str, object]:
    branch = assessment.inputs.speech
    non_present = _non_present_result(branch.status)
    if non_present is not None:
        return non_present
    reason = (
        AgreementReasonCode.SPEECH_PRESENT_NEUTRAL
        if branch.segment_count > 0
        else AgreementReasonCode.SPEECH_ABSENT
    )
    return _working(BranchAgreement.NEUTRAL, reason)


def _base_language_result(
    assessment: RiskAssessmentDocument,
    rules: AgreementRuleSet,
) -> dict[str, object]:
    branch = assessment.inputs.language
    non_present = _non_present_result(branch.status)
    if non_present is not None:
        return non_present
    categories = {signal.category for signal in branch.signals}
    if categories & set(rules.language_support.support_categories):
        return _working(
            BranchAgreement.SUPPORTS_RISK,
            AgreementReasonCode.ACTIVE_LANGUAGE_RISK,
        )
    if LanguageCategory.AMBIGUOUS in categories:
        return _working(
            BranchAgreement.NEUTRAL,
            AgreementReasonCode.AMBIGUOUS_LANGUAGE,
        )
    return _working(
        BranchAgreement.NEUTRAL,
        AgreementReasonCode.SAFE_LANGUAGE_CONTEXT,
    )


def _apply_cross_branch_conflicts(
    assessment: RiskAssessmentDocument,
    rules: AgreementRuleSet,
    working: dict[RiskEvidenceKind, dict[str, object]],
) -> None:
    inputs = assessment.inputs
    acoustic = inputs.acoustic
    speech = inputs.speech
    language = inputs.language
    threshold = rules.speech_consistency.minimum_peak_score
    high_labels = {
        signal.label
        for signal in acoustic.signals
        if signal.peak_score >= threshold
    }
    language_categories = {signal.category for signal in language.signals}
    active_language = bool(
        language_categories & set(rules.language_support.support_categories)
    )

    if (
        rules.conflicts.no_speech_vs_vad_speech
        and rules.speech_consistency.no_speech_label in high_labels
        and speech.status is RiskInputStatus.PRESENT
        and speech.segment_count > 0
    ):
        _mark_conflict(
            working,
            (RiskEvidenceKind.ACOUSTIC, RiskEvidenceKind.SPEECH),
            AgreementReasonCode.ACOUSTIC_NO_SPEECH_CONFLICT,
        )
        if language.status is RiskInputStatus.PRESENT and language.finding_count > 0:
            _mark_conflict(
                working,
                (RiskEvidenceKind.LANGUAGE,),
                AgreementReasonCode.ACOUSTIC_NO_SPEECH_CONFLICT,
            )

    if (
        rules.conflicts.speech_signal_vs_zero_vad_segments
        and high_labels & set(rules.speech_consistency.speech_like_labels)
        and speech.status is RiskInputStatus.PRESENT
        and speech.segment_count == 0
    ):
        _mark_conflict(
            working,
            (RiskEvidenceKind.ACOUSTIC, RiskEvidenceKind.SPEECH),
            AgreementReasonCode.ACOUSTIC_SPEECH_ABSENCE_CONFLICT,
        )

    if (
        rules.conflicts.non_threatening_speech_vs_active_language
        and rules.speech_consistency.non_threatening_speech_label in high_labels
        and active_language
    ):
        _mark_conflict(
            working,
            (RiskEvidenceKind.ACOUSTIC, RiskEvidenceKind.LANGUAGE),
            AgreementReasonCode.NON_THREATENING_LANGUAGE_CONFLICT,
        )

    if (
        rules.conflicts.language_findings_require_accepted_transcript
        and language.status is RiskInputStatus.PRESENT
        and language.finding_count > 0
        and speech.status is RiskInputStatus.PRESENT
        and speech.accepted_transcript_count == 0
    ):
        _mark_conflict(
            working,
            (RiskEvidenceKind.SPEECH, RiskEvidenceKind.LANGUAGE),
            AgreementReasonCode.LANGUAGE_WITHOUT_ACCEPTED_TRANSCRIPT_CONFLICT,
        )


def _non_present_result(status: RiskInputStatus) -> dict[str, object] | None:
    mapping = {
        RiskInputStatus.MISSING: (
            BranchAgreement.UNAVAILABLE,
            AgreementReasonCode.BRANCH_MISSING,
        ),
        RiskInputStatus.NOT_PERMITTED: (
            BranchAgreement.NOT_PERMITTED,
            AgreementReasonCode.BRANCH_NOT_PERMITTED,
        ),
        RiskInputStatus.NOT_APPLICABLE: (
            BranchAgreement.NOT_APPLICABLE,
            AgreementReasonCode.BRANCH_NOT_APPLICABLE,
        ),
        RiskInputStatus.NO_ACCEPTED_TEXT: (
            BranchAgreement.NOT_APPLICABLE,
            AgreementReasonCode.BRANCH_NOT_APPLICABLE,
        ),
    }
    if status is RiskInputStatus.PRESENT:
        return None
    agreement, reason = mapping[status]
    return _working(agreement, reason)


def _working(
    agreement: BranchAgreement,
    reason: AgreementReasonCode,
) -> dict[str, object]:
    return {"agreement": agreement, "reasons": {reason}}


def _mark_conflict(
    working: dict[RiskEvidenceKind, dict[str, object]],
    kinds: tuple[RiskEvidenceKind, ...],
    reason: AgreementReasonCode,
) -> None:
    for kind in kinds:
        current = working[kind]
        if current["agreement"] in {
            BranchAgreement.UNAVAILABLE,
            BranchAgreement.NOT_PERMITTED,
            BranchAgreement.NOT_APPLICABLE,
        }:
            continue
        current["agreement"] = BranchAgreement.CONFLICTS_RISK
        current["reasons"].add(reason)


def _branch_status(
    assessment: RiskAssessmentDocument,
    kind: RiskEvidenceKind,
) -> RiskInputStatus:
    return {
        RiskEvidenceKind.ACOUSTIC: assessment.inputs.acoustic.status,
        RiskEvidenceKind.SPEECH: assessment.inputs.speech.status,
        RiskEvidenceKind.LANGUAGE: assessment.inputs.language.status,
    }[kind]


def _validate_reason_tuple(reasons: tuple[AgreementReasonCode, ...]) -> None:
    if len(set(reasons)) != len(reasons):
        raise ValueError("agreement reason_codes must be unique")
    if list(reasons) != sorted(reasons, key=_REASON_ORDER.__getitem__):
        raise ValueError("agreement reason_codes must use canonical ordering")


def _validate_assessment(assessment: RiskAssessmentDocument) -> RiskAssessmentDocument:
    if not isinstance(assessment, RiskAssessmentDocument):
        raise ConsensusAgreementError(
            "invalid_assessment", "A validated Phase 6 risk assessment is required."
        )
    try:
        validated = RiskAssessmentDocument.model_validate(
            assessment.model_dump(mode="python")
        )
    except Exception as error:
        raise ConsensusAgreementError(
            "invalid_assessment", "Risk assessment failed integrity validation."
        ) from error
    if validated != assessment:
        raise ConsensusAgreementError(
            "invalid_assessment", "Risk assessment integrity validation failed."
        )
    return validated


def _validate_loaded_rules(loaded: LoadedAgreementRuleSet) -> LoadedAgreementRuleSet:
    if not isinstance(loaded, LoadedAgreementRuleSet):
        raise ValueError("loaded agreement rule set has an invalid type")
    validated = AgreementRuleSet.model_validate(loaded.rule_set.model_dump(mode="python"))
    if (
        validated != loaded.rule_set
        or not re.fullmatch(r"[a-f0-9]{64}", loaded.artifact_sha256)
        or not isinstance(loaded.artifact_size_bytes, int)
        or isinstance(loaded.artifact_size_bytes, bool)
        or loaded.artifact_size_bytes <= 0
        or loaded.artifact_size_bytes > MAX_AGREEMENT_RULE_SET_BYTES
    ):
        raise ValueError("loaded agreement rule set failed integrity validation")
    return LoadedAgreementRuleSet(
        rule_set=validated,
        artifact_sha256=loaded.artifact_sha256,
        artifact_size_bytes=loaded.artifact_size_bytes,
    )


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ConsensusAgreementError(
            "invalid_time", "Agreement evaluation time must be timezone-aware."
        )
    return value.astimezone(UTC)


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _evaluation_id(
    *,
    assessment_id: str,
    assessment_sha256: str,
    rule_set: AgreementRuleSetDescriptor,
    branches: tuple[BranchAgreementEvaluation, ...],
) -> str:
    digest = _canonical_hash(
        {
            "assessment_id": assessment_id,
            "assessment_sha256": assessment_sha256,
            "rule_set": rule_set.model_dump(mode="json"),
            "branches": [branch.model_dump(mode="json") for branch in branches],
        }
    )
    return f"agreement-{digest}"


def agreement_schema_documents() -> dict[str, dict[str, object]]:
    """Return the B7.1 public evaluation and rule-set schemas."""

    return {
        "consensus-agreement.schema.json": ConsensusAgreementEvaluation.model_json_schema(),
        "consensus-agreement-rule-set.schema.json": AgreementRuleSet.model_json_schema(),
    }


def write_agreement_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the B7.1 schemas for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in agreement_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
