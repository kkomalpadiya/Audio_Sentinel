"""A7.2 final verification decision-service tests."""

from datetime import UTC, datetime, timedelta
import json

import pytest
from pydantic import ValidationError

from audio_sentinel.consensus_contracts import (
    ConsensusDecisionDocument,
    ConsensusOutcome,
    ConsensusReasonCode,
)
from audio_sentinel.consensus_rules import (
    ConsensusAgreementEvaluation,
    evaluate_evidence_agreement,
    load_agreement_rule_set_bytes,
    load_builtin_agreement_rule_set,
    risk_assessment_sha256,
)
from audio_sentinel.consensus_service import (
    ConsensusVerificationError,
    FinalVerificationService,
    decide_consensus,
)
from audio_sentinel.contracts import EventLabel, ProcessingScope
from audio_sentinel.language_contracts import LanguageCategory, LanguageReasonCode
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
    RiskSource,
    SpeechRiskInputs,
)
from audio_sentinel.risk_scoring import score_risk


NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


def test_clean_critical_multi_branch_support_becomes_local_alert_candidate() -> None:
    assessment = _assessment(
        acoustic=[(EventLabel.EXPLOSION, 0.95)],
        speech_segments=1,
        accepted_transcripts=1,
        language=[LanguageCategory.THREAT],
    )

    decision = _decide(assessment)

    assert assessment.score >= 75
    assert decision.outcome is ConsensusOutcome.ALERT
    assert decision.review_required is True
    assert decision.alert_candidate is True
    assert decision.reason_codes == (
        ConsensusReasonCode.CRITICAL_RISK,
        ConsensusReasonCode.MULTI_BRANCH_ALERT_AGREEMENT,
        ConsensusReasonCode.ALERT_CANDIDATE,
    )


def test_zero_risk_clean_evidence_selects_no_action() -> None:
    decision = _decide(_assessment())

    assert decision.outcome is ConsensusOutcome.NO_ACTION
    assert decision.reason_codes == (ConsensusReasonCode.NO_RISK_EVIDENCE,)
    assert decision.review_required is False
    assert decision.alert_candidate is False


def test_low_positive_clean_evidence_selects_log() -> None:
    assessment = _assessment(acoustic=[(EventLabel.SIREN, 0.4)])

    decision = _decide(assessment)

    assert 0 < assessment.score <= 24
    assert decision.outcome is ConsensusOutcome.LOG
    assert decision.reason_codes == (ConsensusReasonCode.LOW_RISK_LOGGED,)


def test_review_threshold_selects_review() -> None:
    assessment = _assessment(acoustic=[(EventLabel.GLASS_BREAK, 0.5)])

    decision = _decide(assessment)

    assert 25 <= assessment.score < 75
    assert decision.outcome is ConsensusOutcome.REVIEW
    assert ConsensusReasonCode.SCORE_REVIEW_REQUIRED in decision.reason_codes


def test_missing_evidence_cannot_be_treated_as_safe() -> None:
    assessment = _assessment(acoustic_status=RiskInputStatus.MISSING)

    decision = _decide(assessment)

    assert decision.outcome is ConsensusOutcome.REVIEW
    assert ConsensusReasonCode.MISSING_EVIDENCE_REVIEW_REQUIRED in (
        decision.reason_codes
    )


def test_conflict_forces_review_instead_of_averaging_branches() -> None:
    assessment = _assessment(
        acoustic=[(EventLabel.NO_SPEECH, 0.95)],
        speech_segments=1,
        accepted_transcripts=1,
        language=[LanguageCategory.THREAT],
    )

    decision = _decide(assessment)

    assert decision.outcome is ConsensusOutcome.REVIEW
    assert ConsensusReasonCode.EVIDENCE_CONFLICT_REVIEW_REQUIRED in (
        decision.reason_codes
    )
    assert decision.alert_candidate is False


def test_critical_score_without_two_supporting_branches_requires_review() -> None:
    assessment = _assessment(
        acoustic=[(EventLabel.EXPLOSION, 0.8)],
        speech_segments=1,
        accepted_transcripts=1,
        language=[LanguageCategory.THREAT],
    )

    decision = _decide(assessment)

    assert assessment.score >= 75
    assert decision.outcome is ConsensusOutcome.REVIEW
    assert ConsensusReasonCode.INSUFFICIENT_ALERT_AGREEMENT in (
        decision.reason_codes
    )


def test_phase_six_uncertainty_blocks_an_otherwise_eligible_alert() -> None:
    assessment = _assessment(
        acoustic=[(EventLabel.EXPLOSION, 0.95)],
        speech_segments=1,
        accepted_transcripts=1,
        review_transcripts=1,
        language=[LanguageCategory.THREAT],
    )

    decision = _decide(assessment)

    assert decision.outcome is ConsensusOutcome.REVIEW
    assert ConsensusReasonCode.RISK_UNCERTAINTY_REVIEW_REQUIRED in (
        decision.reason_codes
    )


def test_consent_limited_zero_score_remains_no_action_with_receipt() -> None:
    assessment = _assessment(scope=ProcessingScope.ACOUSTIC_ONLY)

    decision = _decide(assessment)

    assert decision.outcome is ConsensusOutcome.NO_ACTION
    assert decision.reason_codes == (
        ConsensusReasonCode.NO_RISK_EVIDENCE,
        ConsensusReasonCode.CONSENT_LIMITED_EVIDENCE,
    )


def test_decision_pins_the_exact_assessment_and_branch_states() -> None:
    assessment = _assessment(acoustic=[(EventLabel.SIREN, 0.4)])
    agreement = evaluate_evidence_agreement(assessment, now=NOW)

    decision = decide_consensus(assessment, agreement, now=NOW)

    assert decision.risk_assessment.assessment_id == assessment.assessment_id
    assert decision.risk_assessment.assessment_sha256 == risk_assessment_sha256(
        assessment
    )
    assert decision.risk_assessment.clip_id == assessment.inputs.source.clip_id
    assert decision.branch_states == agreement.branch_states


def test_mismatched_assessment_and_agreement_are_rejected() -> None:
    assessment = _assessment(acoustic=[(EventLabel.SIREN, 0.4)])
    other = _assessment(acoustic=[(EventLabel.GLASS_BREAK, 0.5)])
    agreement = evaluate_evidence_agreement(other, now=NOW)

    with pytest.raises(ConsensusVerificationError) as captured:
        decide_consensus(assessment, agreement, now=NOW)

    assert captured.value.code == "assessment_mismatch"


def test_tampered_agreement_is_rejected_during_integrity_validation() -> None:
    assessment = _assessment(acoustic=[(EventLabel.SIREN, 0.9)])
    agreement = evaluate_evidence_agreement(assessment, now=NOW)
    object.__setattr__(agreement, "assessment_sha256", "0" * 64)

    with pytest.raises(ConsensusVerificationError) as captured:
        decide_consensus(assessment, agreement, now=NOW)

    assert captured.value.code == "invalid_agreement"


def test_custom_rule_result_requires_explicitly_trusted_rule_artifact() -> None:
    assessment = _assessment(acoustic=[(EventLabel.SIREN, 0.75)])
    custom = _custom_rule_set(minimum_peak_score=0.7)
    agreement = evaluate_evidence_agreement(assessment, rule_set=custom, now=NOW)

    with pytest.raises(ConsensusVerificationError) as captured:
        decide_consensus(assessment, agreement, now=NOW)
    assert captured.value.code == "untrusted_agreement"

    decision = decide_consensus(
        assessment,
        agreement,
        trusted_rule_set=custom,
        now=NOW,
    )
    assert decision.branch_states == agreement.branch_states


def test_decision_identity_is_deterministic_and_excludes_timestamp() -> None:
    assessment = _assessment(acoustic=[(EventLabel.SIREN, 0.4)])
    agreement = evaluate_evidence_agreement(assessment, now=NOW)

    first = decide_consensus(assessment, agreement, now=NOW)
    second = decide_consensus(
        assessment, agreement, now=NOW + timedelta(minutes=5)
    )

    assert first.decision_id == second.decision_id
    assert first.created_at != second.created_at


def test_service_instance_can_be_reused_without_state_leakage() -> None:
    service = FinalVerificationService()
    first_assessment = _assessment()
    second_assessment = _assessment(acoustic=[(EventLabel.SIREN, 0.4)])

    first = service.decide(
        first_assessment,
        evaluate_evidence_agreement(first_assessment, now=NOW),
        now=NOW,
    )
    second = service.decide(
        second_assessment,
        evaluate_evidence_agreement(second_assessment, now=NOW),
        now=NOW,
    )

    assert first.outcome is ConsensusOutcome.NO_ACTION
    assert second.outcome is ConsensusOutcome.LOG


def test_wrong_assessment_type_fails_with_stable_safe_code() -> None:
    with pytest.raises(ConsensusVerificationError) as captured:
        decide_consensus(None, None, now=NOW)  # type: ignore[arg-type]
    assert captured.value.code == "invalid_assessment"


def test_wrong_agreement_type_fails_with_stable_safe_code() -> None:
    with pytest.raises(ConsensusVerificationError) as captured:
        decide_consensus(_assessment(), None, now=NOW)  # type: ignore[arg-type]
    assert captured.value.code == "invalid_agreement"


def test_naive_decision_time_is_rejected() -> None:
    assessment = _assessment()
    agreement = evaluate_evidence_agreement(assessment, now=NOW)

    with pytest.raises(ConsensusVerificationError) as captured:
        decide_consensus(
            assessment,
            agreement,
            now=datetime(2026, 9, 26, 12),
        )

    assert captured.value.code == "invalid_time"


def test_decision_is_portable_immutable_and_privacy_minimized() -> None:
    decision = _decide(
        _assessment(
            speech_segments=1,
            accepted_transcripts=1,
            language=[LanguageCategory.NO_CONCERNING_MATCH],
        )
    )
    serialized = decision.model_dump_json()

    assert ConsensusDecisionDocument.model_validate_json(serialized) == decision
    with pytest.raises(ValidationError, match="frozen"):
        decision.outcome = ConsensusOutcome.ALERT  # type: ignore[misc]
    for forbidden in (
        "transcript_text",
        "matched_text",
        "audio_bytes",
        "speaker",
        "audio_path",
        "phone_number",
    ):
        assert forbidden not in serialized


def _decide(assessment: RiskAssessmentDocument) -> ConsensusDecisionDocument:
    agreement = evaluate_evidence_agreement(assessment, now=NOW)
    return decide_consensus(assessment, agreement, now=NOW)


def _assessment(
    *,
    acoustic: list[tuple[EventLabel, float]] | None = None,
    acoustic_status: RiskInputStatus = RiskInputStatus.PRESENT,
    speech_segments: int = 0,
    accepted_transcripts: int = 0,
    review_transcripts: int = 0,
    language: list[LanguageCategory] | None = None,
    scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH,
) -> RiskAssessmentDocument:
    acoustic = acoustic or []
    language = language or []
    speech_status = RiskInputStatus.PRESENT
    language_status = (
        RiskInputStatus.PRESENT if language else RiskInputStatus.NO_ACCEPTED_TEXT
    )
    if scope is ProcessingScope.ACOUSTIC_ONLY:
        speech_status = RiskInputStatus.NOT_PERMITTED
        language_status = RiskInputStatus.NOT_PERMITTED

    acoustic_signals = tuple(
        AcousticRiskSignal(
            label=label,
            start_sample=index * 16_000,
            end_sample=(index + 1) * 16_000,
            start_seconds=float(index),
            end_seconds=float(index + 1),
            peak_score=peak,
            source_event_index=index,
        )
        for index, (label, peak) in enumerate(acoustic)
    )
    language_signals = tuple(
        LanguageRiskSignal(
            finding_id=f"finding-{index:03d}",
            segment_id=f"segment-{index:03d}",
            category=category,
            reason_codes=_language_reasons(category),
            start_sample=(index + 5) * 16_000,
            end_sample=(index + 6) * 16_000,
            rule_match_count=(
                0 if category is LanguageCategory.NO_CONCERNING_MATCH else 1
            ),
        )
        for index, category in enumerate(language)
    )
    inputs = RiskInputSet(
        source=RiskSource(
            clip_id="clip-final-verification-test",
            consent_id="consent-final-verification-test",
            processing_scope=scope,
            sample_rate_hz=16_000,
            num_samples=160_000,
        ),
        acoustic=AcousticRiskInputs(
            status=acoustic_status,
            evidence=(
                _evidence(RiskEvidenceKind.ACOUSTIC)
                if acoustic_status is RiskInputStatus.PRESENT
                else None
            ),
            event_count=(
                len(acoustic_signals)
                if acoustic_status is RiskInputStatus.PRESENT
                else 0
            ),
            max_peak_score=(
                max((signal.peak_score for signal in acoustic_signals), default=None)
                if acoustic_status is RiskInputStatus.PRESENT
                else None
            ),
            signals=(
                acoustic_signals
                if acoustic_status is RiskInputStatus.PRESENT
                else ()
            ),
        ),
        speech=SpeechRiskInputs(
            status=speech_status,
            evidence=(
                _evidence(RiskEvidenceKind.SPEECH)
                if speech_status is RiskInputStatus.PRESENT
                else None
            ),
            segment_count=(
                speech_segments if speech_status is RiskInputStatus.PRESENT else 0
            ),
            accepted_transcript_count=(
                accepted_transcripts
                if speech_status is RiskInputStatus.PRESENT
                else 0
            ),
            review_required_transcript_count=(
                review_transcripts
                if speech_status is RiskInputStatus.PRESENT
                else 0
            ),
            max_vad_score=(
                0.9
                if speech_status is RiskInputStatus.PRESENT and speech_segments > 0
                else None
            ),
        ),
        language=LanguageRiskInputs(
            status=language_status,
            evidence=(
                _evidence(RiskEvidenceKind.LANGUAGE)
                if language_status is RiskInputStatus.PRESENT
                else None
            ),
            finding_count=(
                len(language_signals)
                if language_status is RiskInputStatus.PRESENT
                else 0
            ),
            signals=(
                language_signals
                if language_status is RiskInputStatus.PRESENT
                else ()
            ),
        ),
    )
    return score_risk(inputs, now=NOW)


def _evidence(kind: RiskEvidenceKind) -> RiskEvidenceReference:
    return RiskEvidenceReference(
        kind=kind,
        document_type={
            RiskEvidenceKind.ACOUSTIC: "acoustic_event_candidates",
            RiskEvidenceKind.SPEECH: "speech_evidence_candidates",
            RiskEvidenceKind.LANGUAGE: "language_evidence",
        }[kind],
        evidence_id=f"{kind.value}-evidence-final-test",
        evidence_sha256={
            RiskEvidenceKind.ACOUSTIC: "a" * 64,
            RiskEvidenceKind.SPEECH: "b" * 64,
            RiskEvidenceKind.LANGUAGE: "c" * 64,
        }[kind],
    )


def _language_reasons(
    category: LanguageCategory,
) -> tuple[LanguageReasonCode, ...]:
    if category is LanguageCategory.NO_CONCERNING_MATCH:
        return (LanguageReasonCode.NO_RULE_MATCH,)
    return (LanguageReasonCode.PHRASE_MATCH,)


def _custom_rule_set(*, minimum_peak_score: float):
    loaded = load_builtin_agreement_rule_set()
    data = loaded.rule_set.model_dump(mode="json")
    data["rule_set_id"] = "custom-final-verification-rules"
    data["rule_set_version"] = "test-1"
    data["acoustic_support"]["minimum_peak_score"] = minimum_peak_score
    raw = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    return load_agreement_rule_set_bytes(raw)
