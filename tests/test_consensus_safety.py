"""B7.2 disagreement, low-confidence, and false-alert safety matrix."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from audio_sentinel.consensus_contracts import (
    BranchAgreement,
    ConsensusOutcome,
    ConsensusReasonCode,
)
from audio_sentinel.consensus_rules import (
    AgreementReasonCode,
    evaluate_evidence_agreement,
)
from audio_sentinel.consensus_service import decide_consensus
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


@dataclass(frozen=True)
class SafetyScenario:
    case_id: str
    acoustic: tuple[tuple[EventLabel, float], ...] = ()
    acoustic_status: RiskInputStatus = RiskInputStatus.PRESENT
    speech_segments: int = 0
    accepted_transcripts: int = 0
    review_transcripts: int = 0
    language: tuple[LanguageCategory, ...] = ()
    scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH
    expected_score: float = 0
    expected_outcome: ConsensusOutcome = ConsensusOutcome.NO_ACTION
    expected_support: tuple[RiskEvidenceKind, ...] = ()
    expected_conflicts: tuple[RiskEvidenceKind, ...] = ()
    expected_reasons: tuple[ConsensusReasonCode, ...] = ()


SAFETY_SCENARIOS = (
    SafetyScenario(
        case_id="clean-zero-no-action",
        expected_reasons=(ConsensusReasonCode.NO_RISK_EVIDENCE,),
    ),
    SafetyScenario(
        case_id="low-positive-log",
        acoustic=((EventLabel.SIREN, 0.84),),
        expected_score=10,
        expected_outcome=ConsensusOutcome.LOG,
        expected_reasons=(ConsensusReasonCode.LOW_RISK_LOGGED,),
    ),
    SafetyScenario(
        case_id="medium-low-confidence-review",
        acoustic=((EventLabel.GLASS_BREAK, 0.84),),
        expected_score=25,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_reasons=(ConsensusReasonCode.SCORE_REVIEW_REQUIRED,),
    ),
    SafetyScenario(
        case_id="critical-below-support-threshold",
        acoustic=((EventLabel.EXPLOSION, 0.849999),),
        speech_segments=1,
        accepted_transcripts=1,
        language=(LanguageCategory.THREAT,),
        expected_score=82,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_support=(RiskEvidenceKind.LANGUAGE,),
        expected_reasons=(
            ConsensusReasonCode.SCORE_REVIEW_REQUIRED,
            ConsensusReasonCode.INSUFFICIENT_ALERT_AGREEMENT,
        ),
    ),
    SafetyScenario(
        case_id="critical-at-support-threshold",
        acoustic=((EventLabel.EXPLOSION, 0.85),),
        speech_segments=1,
        accepted_transcripts=1,
        language=(LanguageCategory.THREAT,),
        expected_score=92,
        expected_outcome=ConsensusOutcome.ALERT,
        expected_support=(
            RiskEvidenceKind.ACOUSTIC,
            RiskEvidenceKind.LANGUAGE,
        ),
        expected_reasons=(
            ConsensusReasonCode.CRITICAL_RISK,
            ConsensusReasonCode.MULTI_BRANCH_ALERT_AGREEMENT,
            ConsensusReasonCode.ALERT_CANDIDATE,
        ),
    ),
    SafetyScenario(
        case_id="low-confidence-no-speech-does-not-conflict",
        acoustic=((EventLabel.NO_SPEECH, 0.849999),),
        speech_segments=1,
        accepted_transcripts=1,
        language=(LanguageCategory.THREAT,),
        expected_score=42,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_support=(RiskEvidenceKind.LANGUAGE,),
        expected_reasons=(ConsensusReasonCode.SCORE_REVIEW_REQUIRED,),
    ),
    SafetyScenario(
        case_id="no-speech-vs-vad-and-language-conflict",
        acoustic=((EventLabel.NO_SPEECH, 0.85),),
        speech_segments=1,
        accepted_transcripts=1,
        language=(LanguageCategory.THREAT,),
        expected_score=42,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_conflicts=tuple(RiskEvidenceKind),
        expected_reasons=(
            ConsensusReasonCode.SCORE_REVIEW_REQUIRED,
            ConsensusReasonCode.EVIDENCE_CONFLICT_REVIEW_REQUIRED,
        ),
    ),
    SafetyScenario(
        case_id="acoustic-speech-vs-zero-vad-conflict",
        acoustic=((EventLabel.THREATENING_SPEECH, 0.85),),
        expected_score=50,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_conflicts=(
            RiskEvidenceKind.ACOUSTIC,
            RiskEvidenceKind.SPEECH,
        ),
        expected_reasons=(
            ConsensusReasonCode.SCORE_REVIEW_REQUIRED,
            ConsensusReasonCode.EVIDENCE_CONFLICT_REVIEW_REQUIRED,
        ),
    ),
    SafetyScenario(
        case_id="non-threatening-speech-vs-active-language-conflict",
        acoustic=((EventLabel.NON_THREATENING_SPEECH, 0.85),),
        speech_segments=1,
        accepted_transcripts=1,
        language=(LanguageCategory.THREAT,),
        expected_score=42,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_conflicts=(
            RiskEvidenceKind.ACOUSTIC,
            RiskEvidenceKind.LANGUAGE,
        ),
        expected_reasons=(
            ConsensusReasonCode.SCORE_REVIEW_REQUIRED,
            ConsensusReasonCode.EVIDENCE_CONFLICT_REVIEW_REQUIRED,
        ),
    ),
    SafetyScenario(
        case_id="language-without-accepted-transcript-conflict",
        acoustic=((EventLabel.EXPLOSION, 0.85),),
        speech_segments=1,
        language=(LanguageCategory.THREAT,),
        expected_score=92,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_support=(RiskEvidenceKind.ACOUSTIC,),
        expected_conflicts=(
            RiskEvidenceKind.SPEECH,
            RiskEvidenceKind.LANGUAGE,
        ),
        expected_reasons=(
            ConsensusReasonCode.SCORE_REVIEW_REQUIRED,
            ConsensusReasonCode.EVIDENCE_CONFLICT_REVIEW_REQUIRED,
            ConsensusReasonCode.INSUFFICIENT_ALERT_AGREEMENT,
        ),
    ),
    SafetyScenario(
        case_id="transcript-review-blocks-clean-critical-alert",
        acoustic=((EventLabel.EXPLOSION, 0.85),),
        speech_segments=1,
        accepted_transcripts=1,
        review_transcripts=1,
        language=(LanguageCategory.THREAT,),
        expected_score=97,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_support=(
            RiskEvidenceKind.ACOUSTIC,
            RiskEvidenceKind.LANGUAGE,
        ),
        expected_reasons=(
            ConsensusReasonCode.SCORE_REVIEW_REQUIRED,
            ConsensusReasonCode.RISK_UNCERTAINTY_REVIEW_REQUIRED,
        ),
    ),
    SafetyScenario(
        case_id="ambiguous-language-blocks-alert",
        acoustic=((EventLabel.EXPLOSION, 0.85),),
        speech_segments=1,
        accepted_transcripts=1,
        language=(LanguageCategory.AMBIGUOUS,),
        expected_score=57,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_support=(RiskEvidenceKind.ACOUSTIC,),
        expected_reasons=(
            ConsensusReasonCode.SCORE_REVIEW_REQUIRED,
            ConsensusReasonCode.RISK_UNCERTAINTY_REVIEW_REQUIRED,
        ),
    ),
    SafetyScenario(
        case_id="missing-acoustic-evidence-forces-review",
        acoustic_status=RiskInputStatus.MISSING,
        speech_segments=1,
        accepted_transcripts=1,
        language=(LanguageCategory.THREAT,),
        expected_score=42,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_support=(RiskEvidenceKind.LANGUAGE,),
        expected_reasons=(
            ConsensusReasonCode.SCORE_REVIEW_REQUIRED,
            ConsensusReasonCode.RISK_UNCERTAINTY_REVIEW_REQUIRED,
            ConsensusReasonCode.MISSING_EVIDENCE_REVIEW_REQUIRED,
        ),
    ),
    SafetyScenario(
        case_id="consent-limited-high-acoustic-review",
        acoustic=((EventLabel.EXPLOSION, 0.85),),
        scope=ProcessingScope.ACOUSTIC_ONLY,
        expected_score=50,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_support=(RiskEvidenceKind.ACOUSTIC,),
        expected_reasons=(
            ConsensusReasonCode.SCORE_REVIEW_REQUIRED,
            ConsensusReasonCode.CONSENT_LIMITED_EVIDENCE,
        ),
    ),
    SafetyScenario(
        case_id="safe-language-does-not-add-alert-support",
        acoustic=((EventLabel.GUNSHOT, 0.85),),
        speech_segments=1,
        accepted_transcripts=1,
        language=(LanguageCategory.NO_CONCERNING_MATCH,),
        expected_score=57,
        expected_outcome=ConsensusOutcome.REVIEW,
        expected_support=(RiskEvidenceKind.ACOUSTIC,),
        expected_reasons=(ConsensusReasonCode.SCORE_REVIEW_REQUIRED,),
    ),
)


@pytest.mark.parametrize(
    "scenario",
    SAFETY_SCENARIOS,
    ids=lambda scenario: scenario.case_id,
)
def test_consensus_safety_matrix(scenario: SafetyScenario) -> None:
    assessment = _assessment(
        acoustic=scenario.acoustic,
        acoustic_status=scenario.acoustic_status,
        speech_segments=scenario.speech_segments,
        accepted_transcripts=scenario.accepted_transcripts,
        review_transcripts=scenario.review_transcripts,
        language=scenario.language,
        scope=scenario.scope,
    )
    agreement = evaluate_evidence_agreement(assessment, now=NOW)
    decision = decide_consensus(assessment, agreement, now=NOW)

    assert assessment.score == scenario.expected_score
    assert agreement.supporting_branches == scenario.expected_support
    assert agreement.conflicting_branches == scenario.expected_conflicts
    assert decision.outcome is scenario.expected_outcome
    assert decision.reason_codes == scenario.expected_reasons
    assert decision.alert_candidate is (
        scenario.expected_outcome is ConsensusOutcome.ALERT
    )
    assert decision.review_required is (
        scenario.expected_outcome
        in {ConsensusOutcome.REVIEW, ConsensusOutcome.ALERT}
    )


@pytest.mark.parametrize(
    "peak,expected_agreement,expected_reason",
    [
        (
            0.0,
            BranchAgreement.NEUTRAL,
            AgreementReasonCode.LOW_CONFIDENCE_RISK_SIGNAL,
        ),
        (
            0.849999,
            BranchAgreement.NEUTRAL,
            AgreementReasonCode.LOW_CONFIDENCE_RISK_SIGNAL,
        ),
        (
            0.85,
            BranchAgreement.SUPPORTS_RISK,
            AgreementReasonCode.HIGH_CONFIDENCE_RISK_SIGNAL,
        ),
        (
            1.0,
            BranchAgreement.SUPPORTS_RISK,
            AgreementReasonCode.HIGH_CONFIDENCE_RISK_SIGNAL,
        ),
    ],
)
def test_acoustic_support_boundary_cannot_round_up_low_confidence(
    peak: float,
    expected_agreement: BranchAgreement,
    expected_reason: AgreementReasonCode,
) -> None:
    assessment = _assessment(acoustic=((EventLabel.EXPLOSION, peak),))

    acoustic = evaluate_evidence_agreement(assessment, now=NOW).branches[0]

    assert acoustic.agreement is expected_agreement
    assert acoustic.reason_codes == (expected_reason,)


@pytest.mark.parametrize(
    "category,expected_agreement",
    [
        (LanguageCategory.NO_CONCERNING_MATCH, BranchAgreement.NEUTRAL),
        (LanguageCategory.AMBIGUOUS, BranchAgreement.NEUTRAL),
        (LanguageCategory.CONTEXT_SUPPRESSED, BranchAgreement.NEUTRAL),
        (LanguageCategory.DISTRESS, BranchAgreement.SUPPORTS_RISK),
        (LanguageCategory.THREAT, BranchAgreement.SUPPORTS_RISK),
        (LanguageCategory.WEAPON_REFERENCE, BranchAgreement.SUPPORTS_RISK),
    ],
)
def test_language_taxonomy_cannot_promote_safe_or_uncertain_context(
    category: LanguageCategory,
    expected_agreement: BranchAgreement,
) -> None:
    assessment = _assessment(
        speech_segments=1,
        accepted_transcripts=1,
        language=(category,),
    )

    language = evaluate_evidence_agreement(assessment, now=NOW).branches[2]

    assert language.agreement is expected_agreement


def test_speech_presence_alone_never_counts_as_alert_support() -> None:
    assessment = _assessment(speech_segments=1)
    agreement = evaluate_evidence_agreement(assessment, now=NOW)
    decision = decide_consensus(assessment, agreement, now=NOW)

    assert agreement.branches[1].agreement is BranchAgreement.NEUTRAL
    assert RiskEvidenceKind.SPEECH not in agreement.supporting_branches
    assert decision.alert_candidate is False


def test_every_false_alert_scenario_has_an_explicit_safety_explanation() -> None:
    false_alert_cases = tuple(
        scenario
        for scenario in SAFETY_SCENARIOS
        if scenario.expected_score >= 75
        and scenario.expected_outcome is not ConsensusOutcome.ALERT
    )

    assert {scenario.case_id for scenario in false_alert_cases} == {
        "critical-below-support-threshold",
        "language-without-accepted-transcript-conflict",
        "transcript-review-blocks-clean-critical-alert",
    }
    for scenario in false_alert_cases:
        assert set(scenario.expected_reasons) & {
            ConsensusReasonCode.INSUFFICIENT_ALERT_AGREEMENT,
            ConsensusReasonCode.EVIDENCE_CONFLICT_REVIEW_REQUIRED,
            ConsensusReasonCode.RISK_UNCERTAINTY_REVIEW_REQUIRED,
        }


def _assessment(
    *,
    acoustic: tuple[tuple[EventLabel, float], ...] = (),
    acoustic_status: RiskInputStatus = RiskInputStatus.PRESENT,
    speech_segments: int = 0,
    accepted_transcripts: int = 0,
    review_transcripts: int = 0,
    language: tuple[LanguageCategory, ...] = (),
    scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH,
) -> RiskAssessmentDocument:
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
            finding_id=f"finding-safety-{index:03d}",
            segment_id=f"segment-safety-{index:03d}",
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
            clip_id="clip-consensus-safety",
            consent_id="consent-consensus-safety",
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
        evidence_id=f"{kind.value}-evidence-safety",
        evidence_sha256={
            RiskEvidenceKind.ACOUSTIC: "d" * 64,
            RiskEvidenceKind.SPEECH: "e" * 64,
            RiskEvidenceKind.LANGUAGE: "f" * 64,
        }[kind],
    )


def _language_reasons(
    category: LanguageCategory,
) -> tuple[LanguageReasonCode, ...]:
    if category is LanguageCategory.NO_CONCERNING_MATCH:
        return (LanguageReasonCode.NO_RULE_MATCH,)
    if category is LanguageCategory.AMBIGUOUS:
        return (
            LanguageReasonCode.PHRASE_MATCH,
            LanguageReasonCode.INSUFFICIENT_CONTEXT,
        )
    if category is LanguageCategory.CONTEXT_SUPPRESSED:
        return (
            LanguageReasonCode.PHRASE_MATCH,
            LanguageReasonCode.EXPLICIT_NEGATION,
        )
    return (LanguageReasonCode.PHRASE_MATCH,)
