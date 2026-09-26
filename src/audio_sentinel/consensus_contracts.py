"""A7.1 consensus policy, branch states, and final outcome contract."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.contracts import RiskLevel
from audio_sentinel.preparation import Identifier
from audio_sentinel.risk_contracts import (
    RiskEvidenceKind,
    RiskInputStatus,
    RiskReasonCode,
    severity_for_score,
)


CONSENSUS_SCHEMA_VERSION = "1.0"
CONSENSUS_POLICY_VERSION = "1.0"


class ConsensusOutcome(str, Enum):
    """The one primary action selected by the verification layer."""

    NO_ACTION = "no_action"
    LOG = "log"
    REVIEW = "review"
    ALERT = "alert"


class BranchAgreement(str, Enum):
    """How one evidence branch relates to the recorded risk assessment."""

    SUPPORTS_RISK = "supports_risk"
    NEUTRAL = "neutral"
    CONFLICTS_RISK = "conflicts_risk"
    UNAVAILABLE = "unavailable"
    NOT_PERMITTED = "not_permitted"
    NOT_APPLICABLE = "not_applicable"


class ConsensusReasonCode(str, Enum):
    """Portable explanations for the selected consensus outcome."""

    NO_RISK_EVIDENCE = "no_risk_evidence"
    LOW_RISK_LOGGED = "low_risk_logged"
    SCORE_REVIEW_REQUIRED = "score_review_required"
    RISK_UNCERTAINTY_REVIEW_REQUIRED = "risk_uncertainty_review_required"
    MISSING_EVIDENCE_REVIEW_REQUIRED = "missing_evidence_review_required"
    EVIDENCE_CONFLICT_REVIEW_REQUIRED = "evidence_conflict_review_required"
    INSUFFICIENT_ALERT_AGREEMENT = "insufficient_alert_agreement"
    CONSENT_LIMITED_EVIDENCE = "consent_limited_evidence"
    CRITICAL_RISK = "critical_risk"
    MULTI_BRANCH_ALERT_AGREEMENT = "multi_branch_alert_agreement"
    ALERT_CANDIDATE = "alert_candidate"


_REASON_ORDER = {reason: index for index, reason in enumerate(ConsensusReasonCode)}
_RISK_REASON_ORDER = {reason: index for index, reason in enumerate(RiskReasonCode)}
_KIND_ORDER = {kind: index for index, kind in enumerate(RiskEvidenceKind)}
_ALERT_BLOCKING_RISK_REASONS = frozenset(
    {
        RiskReasonCode.TRANSCRIPT_REVIEW_REQUIRED,
        RiskReasonCode.LANGUAGE_AMBIGUOUS,
        RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED,
    }
)


class ConsensusContractRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class ConsensusPolicy(ConsensusContractRecord):
    """Fixed v1 action thresholds and alert safety gates."""

    policy_version: Literal["1.0"] = CONSENSUS_POLICY_VERSION
    no_action_score: Literal[0] = 0
    log_max_score: Literal[24] = 24
    review_min_score: Literal[25] = 25
    alert_min_score: Literal[75] = 75
    alert_min_supporting_branches: Literal[2] = 2
    alert_support_eligible_branches: tuple[RiskEvidenceKind, ...] = (
        RiskEvidenceKind.ACOUSTIC,
        RiskEvidenceKind.LANGUAGE,
    )
    conflicts_block_alert: Literal[True] = True
    missing_evidence_blocks_alert: Literal[True] = True
    uncertainty_blocks_alert: Literal[True] = True
    alert_requires_critical_severity: Literal[True] = True
    alert_is_local_candidate_only: Literal[True] = True
    outcome_precedence: tuple[ConsensusOutcome, ...] = (
        ConsensusOutcome.ALERT,
        ConsensusOutcome.REVIEW,
        ConsensusOutcome.LOG,
        ConsensusOutcome.NO_ACTION,
    )

    @model_validator(mode="after")
    def validate_policy(self) -> "ConsensusPolicy":
        expected_branches = (
            RiskEvidenceKind.ACOUSTIC,
            RiskEvidenceKind.LANGUAGE,
        )
        if self.alert_support_eligible_branches != expected_branches:
            raise ValueError("v1 alert support branches must be acoustic and language")
        expected_precedence = (
            ConsensusOutcome.ALERT,
            ConsensusOutcome.REVIEW,
            ConsensusOutcome.LOG,
            ConsensusOutcome.NO_ACTION,
        )
        if self.outcome_precedence != expected_precedence:
            raise ValueError("v1 consensus outcome precedence must match the policy")
        return self


class ConsensusRiskReference(ConsensusContractRecord):
    """Hash-pinned, privacy-minimized snapshot of one Phase 6 assessment."""

    assessment_id: Identifier
    assessment_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    clip_id: Identifier
    score: float = Field(ge=0, le=100)
    severity: RiskLevel
    human_review_required: bool
    reason_codes: tuple[RiskReasonCode, ...] = Field(min_length=1)
    missing_branches: tuple[RiskEvidenceKind, ...] = ()
    not_permitted_branches: tuple[RiskEvidenceKind, ...] = ()

    @model_validator(mode="after")
    def validate_risk_reference(self) -> "ConsensusRiskReference":
        if self.severity is not severity_for_score(self.score):
            raise ValueError("consensus risk severity must match the Phase 6 score")
        _validate_unique_ordered(
            "risk reason_codes", self.reason_codes, _RISK_REASON_ORDER
        )
        _validate_unique_ordered(
            "missing_branches", self.missing_branches, _KIND_ORDER
        )
        _validate_unique_ordered(
            "not_permitted_branches", self.not_permitted_branches, _KIND_ORDER
        )
        if set(self.missing_branches) & set(self.not_permitted_branches):
            raise ValueError("a branch cannot be both missing and not permitted")
        if self.missing_branches and not self.human_review_required:
            raise ValueError("missing risk evidence requires Phase 6 human review")
        return self


class ConsensusBranchState(ConsensusContractRecord):
    """Typed B7.1 handoff for one Phase 6 evidence branch."""

    kind: RiskEvidenceKind
    input_status: RiskInputStatus
    agreement: BranchAgreement

    @model_validator(mode="after")
    def validate_state(self) -> "ConsensusBranchState":
        if (
            self.kind is RiskEvidenceKind.SPEECH
            and self.agreement is BranchAgreement.SUPPORTS_RISK
        ):
            raise ValueError("speech presence alone cannot support a risk decision")
        if self.input_status is RiskInputStatus.PRESENT:
            if self.agreement not in {
                BranchAgreement.SUPPORTS_RISK,
                BranchAgreement.NEUTRAL,
                BranchAgreement.CONFLICTS_RISK,
            }:
                raise ValueError("present branch requires an evaluated agreement state")
        elif self.input_status is RiskInputStatus.MISSING:
            if self.agreement is not BranchAgreement.UNAVAILABLE:
                raise ValueError("missing branch must be unavailable")
        elif self.input_status is RiskInputStatus.NOT_PERMITTED:
            if self.agreement is not BranchAgreement.NOT_PERMITTED:
                raise ValueError("not-permitted branch must retain that state")
        elif self.agreement is not BranchAgreement.NOT_APPLICABLE:
            raise ValueError("non-applicable branch must retain that state")
        return self


class ConsensusDecisionDocument(ConsensusContractRecord):
    """Versioned consensus outcome; alert means a local candidate, not delivery."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/consensus-decision.schema.json"
        },
    )

    schema_version: Literal["1.0"] = CONSENSUS_SCHEMA_VERSION
    document_type: Literal["consensus_decision"] = "consensus_decision"
    decision_id: Identifier
    created_at: datetime
    policy: ConsensusPolicy = Field(default_factory=ConsensusPolicy)
    risk_assessment: ConsensusRiskReference
    branch_states: tuple[ConsensusBranchState, ...] = Field(min_length=3, max_length=3)
    outcome: ConsensusOutcome
    reason_codes: tuple[ConsensusReasonCode, ...] = Field(min_length=1)
    review_required: bool
    alert_candidate: bool

    @model_validator(mode="after")
    def validate_document(self) -> "ConsensusDecisionDocument":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        expected_kinds = tuple(RiskEvidenceKind)
        if tuple(state.kind for state in self.branch_states) != expected_kinds:
            raise ValueError("branch_states must cover every branch in canonical order")
        _validate_unique_ordered("reason_codes", self.reason_codes, _REASON_ORDER)
        self._validate_snapshot_alignment()
        self._validate_outcome()
        return self

    def _validate_snapshot_alignment(self) -> None:
        status_by_kind = {state.kind: state.input_status for state in self.branch_states}
        missing = tuple(
            kind
            for kind in RiskEvidenceKind
            if status_by_kind[kind] is RiskInputStatus.MISSING
        )
        not_permitted = tuple(
            kind
            for kind in RiskEvidenceKind
            if status_by_kind[kind] is RiskInputStatus.NOT_PERMITTED
        )
        if missing != self.risk_assessment.missing_branches:
            raise ValueError("branch states must match risk missing_branches")
        if not_permitted != self.risk_assessment.not_permitted_branches:
            raise ValueError("branch states must match risk not_permitted_branches")

    def _validate_outcome(self) -> None:
        risk = self.risk_assessment
        supporting = {
            state.kind
            for state in self.branch_states
            if state.agreement is BranchAgreement.SUPPORTS_RISK
        }
        conflicts = {
            state.kind
            for state in self.branch_states
            if state.agreement is BranchAgreement.CONFLICTS_RISK
        }
        eligible_support = supporting & set(self.policy.alert_support_eligible_branches)
        blocking_uncertainty = bool(
            set(risk.reason_codes) & _ALERT_BLOCKING_RISK_REASONS
        )
        alert_eligible = (
            risk.score >= self.policy.alert_min_score
            and risk.severity is RiskLevel.CRITICAL
            and len(eligible_support) >= self.policy.alert_min_supporting_branches
            and not conflicts
            and not risk.missing_branches
            and not blocking_uncertainty
        )

        if self.alert_candidate is not (self.outcome is ConsensusOutcome.ALERT):
            raise ValueError("alert_candidate must match the alert outcome")
        if self.review_required is not (
            self.outcome in {ConsensusOutcome.REVIEW, ConsensusOutcome.ALERT}
        ):
            raise ValueError("review_required must match review or alert outcomes")

        if self.outcome is ConsensusOutcome.ALERT:
            if not alert_eligible:
                raise ValueError("alert outcome does not satisfy every v1 safety gate")
            _require_reasons(
                self.reason_codes,
                {
                    ConsensusReasonCode.CRITICAL_RISK,
                    ConsensusReasonCode.MULTI_BRANCH_ALERT_AGREEMENT,
                    ConsensusReasonCode.ALERT_CANDIDATE,
                },
            )
            return

        if alert_eligible:
            raise ValueError("an alert-eligible decision must use the alert outcome")

        if self.outcome is ConsensusOutcome.REVIEW:
            review_triggered = (
                risk.score >= self.policy.review_min_score
                or risk.human_review_required
                or bool(conflicts)
                or bool(risk.missing_branches)
            )
            if not review_triggered:
                raise ValueError("review outcome requires a score or safety trigger")
            required_reasons: set[ConsensusReasonCode] = set()
            if risk.score >= self.policy.review_min_score:
                required_reasons.add(ConsensusReasonCode.SCORE_REVIEW_REQUIRED)
            if blocking_uncertainty:
                required_reasons.add(
                    ConsensusReasonCode.RISK_UNCERTAINTY_REVIEW_REQUIRED
                )
            if risk.missing_branches:
                required_reasons.add(
                    ConsensusReasonCode.MISSING_EVIDENCE_REVIEW_REQUIRED
                )
            if conflicts:
                required_reasons.add(
                    ConsensusReasonCode.EVIDENCE_CONFLICT_REVIEW_REQUIRED
                )
            if (
                risk.score >= self.policy.alert_min_score
                and len(eligible_support)
                < self.policy.alert_min_supporting_branches
            ):
                required_reasons.add(
                    ConsensusReasonCode.INSUFFICIENT_ALERT_AGREEMENT
                )
            _require_reasons(self.reason_codes, required_reasons)
            return

        if self.outcome is ConsensusOutcome.LOG:
            if not (0 < risk.score <= self.policy.log_max_score):
                raise ValueError("log outcome requires a low positive risk score")
            if risk.human_review_required or conflicts or risk.missing_branches:
                raise ValueError("log outcome cannot bypass a review trigger")
            _require_reasons(
                self.reason_codes, {ConsensusReasonCode.LOW_RISK_LOGGED}
            )
            return

        if risk.score != self.policy.no_action_score:
            raise ValueError("no-action outcome requires a zero risk score")
        if risk.human_review_required or conflicts or risk.missing_branches or supporting:
            raise ValueError("no-action outcome cannot bypass evidence or review triggers")
        _require_reasons(self.reason_codes, {ConsensusReasonCode.NO_RISK_EVIDENCE})


def _validate_unique_ordered(
    field_name: str,
    values: tuple[Enum, ...],
    order: dict[Enum, int],
) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")
    if list(values) != sorted(values, key=order.__getitem__):
        raise ValueError(f"{field_name} must use canonical ordering")


def _require_reasons(
    actual: tuple[ConsensusReasonCode, ...],
    required: set[ConsensusReasonCode],
) -> None:
    if not required.issubset(actual):
        raise ValueError("consensus outcome is missing required reason codes")


def consensus_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public JSON Schema for the v1 consensus decision."""

    return {
        "consensus-decision.schema.json": ConsensusDecisionDocument.model_json_schema()
    }


def write_consensus_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the consensus contract for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in consensus_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
