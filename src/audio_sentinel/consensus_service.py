"""A7.2 final verification service for one Phase 6 assessment."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json

from audio_sentinel.consensus_contracts import (
    ConsensusBranchState,
    ConsensusDecisionDocument,
    ConsensusOutcome,
    ConsensusPolicy,
    ConsensusReasonCode,
    ConsensusRiskReference,
)
from audio_sentinel.consensus_rules import (
    ConsensusAgreementEvaluation,
    LoadedAgreementRuleSet,
    evaluate_evidence_agreement,
    load_builtin_agreement_rule_set,
    risk_assessment_sha256,
)
from audio_sentinel.risk_contracts import (
    RiskAssessmentDocument,
    RiskEvidenceKind,
    RiskReasonCode,
)


_REASON_ORDER = {reason: index for index, reason in enumerate(ConsensusReasonCode)}
_ALERT_BLOCKING_RISK_REASONS = frozenset(
    {
        RiskReasonCode.TRANSCRIPT_REVIEW_REQUIRED,
        RiskReasonCode.LANGUAGE_AMBIGUOUS,
        RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED,
    }
)


class ConsensusVerificationError(RuntimeError):
    """Stable failure code plus a safe, non-sensitive message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class FinalVerificationService:
    """Verify trusted inputs and apply the fixed A7.1 consensus policy."""

    def __init__(
        self,
        *,
        policy: ConsensusPolicy | None = None,
        trusted_rule_set: LoadedAgreementRuleSet | None = None,
    ) -> None:
        self._policy = _validate_policy(policy or ConsensusPolicy())
        try:
            self._trusted_rule_set = (
                trusted_rule_set
                if trusted_rule_set is not None
                else load_builtin_agreement_rule_set()
            )
        except Exception as error:
            raise ConsensusVerificationError(
                "invalid_rule_set",
                "Trusted agreement rules failed integrity validation.",
            ) from error

    def decide(
        self,
        assessment: RiskAssessmentDocument,
        agreement: ConsensusAgreementEvaluation,
        *,
        now: datetime | None = None,
    ) -> ConsensusDecisionDocument:
        """Return exactly one verified outcome without sending an alert."""

        created_at = _clock(now)
        validated_assessment = _validate_assessment(assessment)
        validated_agreement = _validate_agreement(agreement)
        assessment_hash = risk_assessment_sha256(validated_assessment)

        if (
            validated_agreement.assessment_id != validated_assessment.assessment_id
            or validated_agreement.assessment_sha256 != assessment_hash
        ):
            raise ConsensusVerificationError(
                "assessment_mismatch",
                "Agreement evaluation does not belong to the supplied risk assessment.",
            )

        try:
            expected_agreement = evaluate_evidence_agreement(
                validated_assessment,
                rule_set=self._trusted_rule_set,
                now=validated_agreement.created_at,
            )
        except Exception as error:
            raise ConsensusVerificationError(
                "invalid_rule_set",
                "Trusted agreement rules failed integrity validation.",
            ) from error
        if expected_agreement != validated_agreement:
            raise ConsensusVerificationError(
                "untrusted_agreement",
                "Agreement evaluation does not match the trusted rule result.",
            )

        branch_states = validated_agreement.branch_states
        supporting = set(validated_agreement.supporting_branches)
        conflicts = set(validated_agreement.conflicting_branches)
        eligible_support = supporting & set(
            self._policy.alert_support_eligible_branches
        )
        blocking_uncertainty = bool(
            set(validated_assessment.reason_codes)
            & _ALERT_BLOCKING_RISK_REASONS
        )
        missing = validated_assessment.missing_data.missing_branches
        not_permitted = validated_assessment.missing_data.not_permitted_branches

        alert_eligible = (
            validated_assessment.score >= self._policy.alert_min_score
            and validated_assessment.severity.value == "critical"
            and len(eligible_support)
            >= self._policy.alert_min_supporting_branches
            and not conflicts
            and not missing
            and not blocking_uncertainty
        )

        reasons: set[ConsensusReasonCode] = set()
        if not_permitted:
            reasons.add(ConsensusReasonCode.CONSENT_LIMITED_EVIDENCE)

        if alert_eligible:
            outcome = ConsensusOutcome.ALERT
            reasons.update(
                {
                    ConsensusReasonCode.CRITICAL_RISK,
                    ConsensusReasonCode.MULTI_BRANCH_ALERT_AGREEMENT,
                    ConsensusReasonCode.ALERT_CANDIDATE,
                }
            )
        else:
            review_triggered = (
                validated_assessment.score >= self._policy.review_min_score
                or validated_assessment.human_review_required
                or bool(conflicts)
                or bool(missing)
            )
            if review_triggered:
                outcome = ConsensusOutcome.REVIEW
                if validated_assessment.score >= self._policy.review_min_score:
                    reasons.add(ConsensusReasonCode.SCORE_REVIEW_REQUIRED)
                if blocking_uncertainty:
                    reasons.add(
                        ConsensusReasonCode.RISK_UNCERTAINTY_REVIEW_REQUIRED
                    )
                if missing:
                    reasons.add(
                        ConsensusReasonCode.MISSING_EVIDENCE_REVIEW_REQUIRED
                    )
                if conflicts:
                    reasons.add(
                        ConsensusReasonCode.EVIDENCE_CONFLICT_REVIEW_REQUIRED
                    )
                if (
                    validated_assessment.score >= self._policy.alert_min_score
                    and len(eligible_support)
                    < self._policy.alert_min_supporting_branches
                ):
                    reasons.add(
                        ConsensusReasonCode.INSUFFICIENT_ALERT_AGREEMENT
                    )
                if not reasons:
                    raise ConsensusVerificationError(
                        "unexplained_review",
                        "Risk assessment requires review without a recognized safety reason.",
                    )
            elif 0 < validated_assessment.score <= self._policy.log_max_score:
                outcome = ConsensusOutcome.LOG
                reasons.add(ConsensusReasonCode.LOW_RISK_LOGGED)
            elif (
                validated_assessment.score == self._policy.no_action_score
                and not supporting
            ):
                outcome = ConsensusOutcome.NO_ACTION
                reasons.add(ConsensusReasonCode.NO_RISK_EVIDENCE)
            else:
                raise ConsensusVerificationError(
                    "inconsistent_evidence",
                    "Risk score and agreement evidence cannot produce a safe outcome.",
                )

        ordered_reasons = tuple(sorted(reasons, key=_REASON_ORDER.__getitem__))
        risk_reference = ConsensusRiskReference(
            assessment_id=validated_assessment.assessment_id,
            assessment_sha256=assessment_hash,
            clip_id=validated_assessment.inputs.source.clip_id,
            score=validated_assessment.score,
            severity=validated_assessment.severity,
            human_review_required=validated_assessment.human_review_required,
            reason_codes=validated_assessment.reason_codes,
            missing_branches=missing,
            not_permitted_branches=not_permitted,
        )
        decision_id = _decision_id(
            policy=self._policy,
            risk_reference=risk_reference,
            branch_states=branch_states,
            outcome=outcome,
            reason_codes=ordered_reasons,
        )
        return ConsensusDecisionDocument(
            decision_id=decision_id,
            created_at=created_at,
            policy=self._policy,
            risk_assessment=risk_reference,
            branch_states=branch_states,
            outcome=outcome,
            reason_codes=ordered_reasons,
            review_required=outcome
            in {ConsensusOutcome.REVIEW, ConsensusOutcome.ALERT},
            alert_candidate=outcome is ConsensusOutcome.ALERT,
        )


def decide_consensus(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
    *,
    policy: ConsensusPolicy | None = None,
    trusted_rule_set: LoadedAgreementRuleSet | None = None,
    now: datetime | None = None,
) -> ConsensusDecisionDocument:
    """Convenience entry point for the A7.2 final verification service."""

    return FinalVerificationService(
        policy=policy,
        trusted_rule_set=trusted_rule_set,
    ).decide(assessment, agreement, now=now)


def _validate_assessment(
    assessment: RiskAssessmentDocument,
) -> RiskAssessmentDocument:
    if not isinstance(assessment, RiskAssessmentDocument):
        raise ConsensusVerificationError(
            "invalid_assessment", "A validated Phase 6 risk assessment is required."
        )
    try:
        validated = RiskAssessmentDocument.model_validate(
            assessment.model_dump(mode="python")
        )
    except Exception as error:
        raise ConsensusVerificationError(
            "invalid_assessment", "Risk assessment failed integrity validation."
        ) from error
    if validated != assessment:
        raise ConsensusVerificationError(
            "invalid_assessment", "Risk assessment failed integrity validation."
        )
    return validated


def _validate_agreement(
    agreement: ConsensusAgreementEvaluation,
) -> ConsensusAgreementEvaluation:
    if not isinstance(agreement, ConsensusAgreementEvaluation):
        raise ConsensusVerificationError(
            "invalid_agreement", "A validated B7.1 agreement evaluation is required."
        )
    try:
        validated = ConsensusAgreementEvaluation.model_validate(
            agreement.model_dump(mode="python")
        )
    except Exception as error:
        raise ConsensusVerificationError(
            "invalid_agreement", "Agreement evaluation failed integrity validation."
        ) from error
    if validated != agreement:
        raise ConsensusVerificationError(
            "invalid_agreement", "Agreement evaluation failed integrity validation."
        )
    return validated


def _validate_policy(policy: ConsensusPolicy) -> ConsensusPolicy:
    if not isinstance(policy, ConsensusPolicy):
        raise ConsensusVerificationError(
            "invalid_policy", "A validated A7.1 consensus policy is required."
        )
    try:
        validated = ConsensusPolicy.model_validate(policy.model_dump(mode="python"))
    except Exception as error:
        raise ConsensusVerificationError(
            "invalid_policy", "Consensus policy failed integrity validation."
        ) from error
    if validated != policy:
        raise ConsensusVerificationError(
            "invalid_policy", "Consensus policy failed integrity validation."
        )
    return validated


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ConsensusVerificationError(
            "invalid_time", "Consensus decision time must be timezone-aware."
        )
    return value.astimezone(UTC)


def _decision_id(
    *,
    policy: ConsensusPolicy,
    risk_reference: ConsensusRiskReference,
    branch_states: tuple[ConsensusBranchState, ...],
    outcome: ConsensusOutcome,
    reason_codes: tuple[ConsensusReasonCode, ...],
) -> str:
    payload = {
        "policy": policy.model_dump(mode="json"),
        "risk_assessment": risk_reference.model_dump(mode="json"),
        "branch_states": [state.model_dump(mode="json") for state in branch_states],
        "outcome": outcome.value,
        "reason_codes": [reason.value for reason in reason_codes],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"decision-{hashlib.sha256(canonical).hexdigest()}"
