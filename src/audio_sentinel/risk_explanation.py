"""Evidence-based, deterministic explanation of one verified final report.

This module answers "Is this audio dangerous?" using only the persisted, hash-verified
final report produced by the existing pipeline. It never sends audio or text to an
external service, never invents a confidence value, and never states that a
recording definitely contains an emergency.

The per-branch point breakdown re-runs the *existing* private scoring functions in
``risk_scoring`` against the report's recorded inputs and the pinned built-in rule
set. If that recomputation does not reproduce the recorded score exactly, the
breakdown is withheld rather than shown with numbers that might be wrong.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from audio_sentinel.consensus_contracts import (
    BranchAgreement,
    ConsensusOutcome,
    ConsensusReasonCode,
)
from audio_sentinel.contracts import EventLabel
from audio_sentinel.final_report import FinalReportDocument
from audio_sentinel.language_contracts import LanguageCategory
from audio_sentinel.risk_contracts import (
    RiskEvidenceKind,
    RiskInputStatus,
    RiskReasonCode,
)
from audio_sentinel import risk_scoring


EXPLANATION_VERSION = "1.0"
SCORE_TOLERANCE = 1e-6

Verdict = Literal[
    "no_indicators_detected",
    "low_indicators",
    "possible_risk_review_recommended",
    "elevated_risk_alert_candidate",
    "inconclusive",
]


class ExplanationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, protected_namespaces=())


class AcousticEventEvidence(ExplanationRecord):
    label: str
    display_name: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    peak_score: float = Field(ge=0, le=1)
    rule_points: float = Field(ge=0)
    high_confidence: bool


class LanguageFindingEvidence(ExplanationRecord):
    category: str
    display_name: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    reason_codes: tuple[str, ...]
    rule_points: float = Field(ge=0)


class BranchContribution(ExplanationRecord):
    branch: Literal["acoustic", "speech", "language"]
    input_status: str
    agreement: str
    points: float | None = Field(default=None, ge=0)
    max_points: float = Field(ge=0)
    summary: str


class ScoreBreakdown(ExplanationRecord):
    reproduced: bool
    total_before_cap: float | None = None
    total_max_score: float
    rule_set_id: str
    rule_set_version: str


class RiskExplanation(ExplanationRecord):
    """Structured answer; ``answer`` is the one-paragraph human summary."""

    explanation_version: Literal["1.0"] = EXPLANATION_VERSION
    report_id: str
    question: Literal["Is this audio dangerous?"] = "Is this audio dangerous?"
    verdict: Verdict
    answer: str
    risk_score: float = Field(ge=0, le=100)
    score_range: tuple[int, int] = (0, 100)
    risk_level: str
    outcome: str
    human_review_recommended: bool
    alert_candidate: bool
    acoustic_events: tuple[AcousticEventEvidence, ...]
    speech_summary: str
    language_findings: tuple[LanguageFindingEvidence, ...]
    contributions: tuple[BranchContribution, ...]
    breakdown: ScoreBreakdown
    risk_reasons: tuple[str, ...]
    decision_reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    model_confidence: None = None
    notification_delivery: Literal["not_sent"] = "not_sent"


EVENT_NAMES: dict[EventLabel, str] = {
    EventLabel.AMBIENT: "Ambient sound",
    EventLabel.NO_SPEECH: "No speech",
    EventLabel.SPEECH_PRESENT: "Speech present",
    EventLabel.NON_THREATENING_SPEECH: "Non-threatening speech",
    EventLabel.SIREN: "Siren",
    EventLabel.SMOKE_ALARM: "Smoke / fire alarm",
    EventLabel.GLASS_BREAK: "Glass breaking",
    EventLabel.CROWD_PANIC: "Crowd panic / screaming",
    EventLabel.DISTRESS_SPEECH: "Distress vocalisation",
    EventLabel.THREATENING_SPEECH: "Threatening speech",
    EventLabel.WEAPON_REFERENCE: "Weapon reference",
    EventLabel.GUNSHOT: "Gunshot-like sound",
    EventLabel.EXPLOSION: "Explosion-like sound",
}

LANGUAGE_NAMES: dict[LanguageCategory, str] = {
    LanguageCategory.NO_CONCERNING_MATCH: "No concerning language",
    LanguageCategory.DISTRESS: "Distress language",
    LanguageCategory.THREAT: "Threatening language",
    LanguageCategory.WEAPON_REFERENCE: "Weapon reference in speech",
    LanguageCategory.AMBIGUOUS: "Ambiguous language",
    LanguageCategory.CONTEXT_SUPPRESSED: "Concerning words in a negated, quoted or hypothetical context",
}

RISK_REASON_TEXT: dict[RiskReasonCode, str] = {
    RiskReasonCode.NO_RISK_EVIDENCE: "No scored risk evidence was found.",
    RiskReasonCode.ACOUSTIC_EVENT_CANDIDATE: "The acoustic model detected at least one risk-weighted sound event.",
    RiskReasonCode.ACOUSTIC_HIGH_CONFIDENCE: "At least one risk-weighted sound event reached the rule set's high-confidence threshold.",
    RiskReasonCode.SPEECH_PRESENT: "Voice activity detection found speech segments.",
    RiskReasonCode.TRANSCRIPT_REVIEW_REQUIRED: "Some transcripts were unreliable and need human review.",
    RiskReasonCode.LANGUAGE_DISTRESS: "Transcribed speech matched distress language rules.",
    RiskReasonCode.LANGUAGE_THREAT: "Transcribed speech matched threat language rules.",
    RiskReasonCode.LANGUAGE_WEAPON_REFERENCE: "Transcribed speech referred to a weapon.",
    RiskReasonCode.LANGUAGE_AMBIGUOUS: "Transcribed speech was ambiguous.",
    RiskReasonCode.LANGUAGE_CONTEXT_SUPPRESSED: "Concerning words appeared only in a negated, quoted or hypothetical context.",
    RiskReasonCode.MISSING_ACOUSTIC_EVIDENCE: "Acoustic evidence is missing.",
    RiskReasonCode.MISSING_SPEECH_EVIDENCE: "Speech evidence is missing.",
    RiskReasonCode.MISSING_LANGUAGE_EVIDENCE: "Language evidence is missing.",
    RiskReasonCode.SPEECH_NOT_PERMITTED: "Speech analysis was not permitted by the selected consent scope.",
    RiskReasonCode.LANGUAGE_NOT_APPLICABLE: "Language analysis did not apply (not permitted or no accepted transcript).",
    RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED: "Required evidence is missing, so human review is required.",
}

DECISION_REASON_TEXT: dict[ConsensusReasonCode, str] = {
    ConsensusReasonCode.NO_RISK_EVIDENCE: "No risk evidence: no action selected.",
    ConsensusReasonCode.LOW_RISK_LOGGED: "Low score: the result is logged only.",
    ConsensusReasonCode.SCORE_REVIEW_REQUIRED: "The score reached the human-review band (25 or more).",
    ConsensusReasonCode.RISK_UNCERTAINTY_REVIEW_REQUIRED: "Uncertain evidence (for example unreliable transcripts or ambiguous language) requires review.",
    ConsensusReasonCode.MISSING_EVIDENCE_REVIEW_REQUIRED: "Missing evidence requires review.",
    ConsensusReasonCode.EVIDENCE_CONFLICT_REVIEW_REQUIRED: "The evidence branches disagree, so review is required.",
    ConsensusReasonCode.INSUFFICIENT_ALERT_AGREEMENT: "Not enough independent branches agreed to raise an alert candidate.",
    ConsensusReasonCode.CONSENT_LIMITED_EVIDENCE: "Consent scope limited the evidence that could be analysed.",
    ConsensusReasonCode.CRITICAL_RISK: "The score is in the critical band (75 or more).",
    ConsensusReasonCode.MULTI_BRANCH_ALERT_AGREEMENT: "Acoustic and language evidence independently supported the risk.",
    ConsensusReasonCode.ALERT_CANDIDATE: "A local alert candidate was created for human review (no notification sent).",
}

BASE_LIMITATIONS: tuple[str, ...] = (
    "This is an automated screening result, not a determination that an emergency occurred.",
    "The acoustic threshold and scoring rules are experimental and not calibrated for production use.",
    "The models can miss events and can confuse similar sounds (for example fireworks and gunshots).",
    "No model confidence is reported for the overall assessment; peak scores are per-event model outputs only.",
)


def _seconds(sample: int, sample_rate_hz: int) -> float:
    return round(sample / sample_rate_hz, 3)


def _builtin_rules_match(report: FinalReportDocument) -> risk_scoring.LoadedRiskRuleSet | None:
    try:
        loaded = risk_scoring.load_builtin_risk_rule_set()
    except Exception:
        return None
    if loaded.as_descriptor() != report.risk_assessment.scoring_policy.rule_set:
        return None
    return loaded


def explain_report(
    report: FinalReportDocument,
    *,
    silent_input: bool = False,
) -> RiskExplanation:
    """Build the evidence-based answer for one verified final report."""

    assessment = report.risk_assessment
    inputs = assessment.inputs
    decision = report.decision
    rate = inputs.source.sample_rate_hz
    loaded = _builtin_rules_match(report)
    rules = None if loaded is None else loaded.rule_set

    # --- Reproduce the existing scoring per branch (no new thresholds) ---------
    acoustic_pts = speech_pts = language_pts = None
    reproduced = False
    total_before_cap = None
    if rules is not None:
        acoustic_pts, _ = risk_scoring._acoustic_score(inputs.acoustic, rules.acoustic)
        speech_pts, _ = risk_scoring._speech_score(inputs.speech, rules.speech)
        language_pts, _ = risk_scoring._language_score(inputs.language, rules.language)
        total_before_cap = round(acoustic_pts + speech_pts + language_pts, 6)
        reproduced = (
            abs(min(rules.total_max_score, total_before_cap) - assessment.score)
            <= SCORE_TOLERANCE
        )
        if not reproduced:
            acoustic_pts = speech_pts = language_pts = total_before_cap = None

    label_points = (
        {}
        if rules is None
        else {item.label: item.points for item in rules.acoustic.label_scores}
    )
    category_points = (
        {}
        if rules is None
        else {item.category: item.points for item in rules.language.category_scores}
    )
    high_threshold = None if rules is None else rules.acoustic.high_confidence_threshold

    events = tuple(
        AcousticEventEvidence(
            label=signal.label.value,
            display_name=EVENT_NAMES.get(signal.label, signal.label.value),
            start_seconds=round(signal.start_seconds, 3),
            end_seconds=round(signal.end_seconds, 3),
            peak_score=round(signal.peak_score, 4),
            rule_points=float(label_points.get(signal.label, 0)),
            high_confidence=(
                high_threshold is not None and signal.peak_score >= high_threshold
            ),
        )
        for signal in sorted(
            inputs.acoustic.signals,
            key=lambda s: (-label_points.get(s.label, 0), -s.peak_score, s.start_sample),
        )
    )
    findings = tuple(
        LanguageFindingEvidence(
            category=signal.category.value,
            display_name=LANGUAGE_NAMES.get(signal.category, signal.category.value),
            start_seconds=_seconds(signal.start_sample, rate),
            end_seconds=_seconds(signal.end_sample, rate),
            reason_codes=tuple(code.value for code in signal.reason_codes),
            rule_points=float(category_points.get(signal.category, 0)),
        )
        for signal in inputs.language.signals
    )

    speech = inputs.speech
    if speech.status is RiskInputStatus.NOT_PERMITTED:
        speech_summary = "Speech analysis was not run because the selected consent scope is acoustic-only."
    elif speech.status is not RiskInputStatus.PRESENT:
        speech_summary = f"Speech evidence status: {speech.status.value}."
    elif speech.segment_count == 0:
        speech_summary = "No speech segments were detected."
    else:
        speech_summary = (
            f"{speech.segment_count} speech segment(s) detected; "
            f"{speech.accepted_transcript_count} transcript(s) accepted, "
            f"{speech.review_required_transcript_count} need review. "
            "Transcript text is intentionally not shown."
        )

    states = {state.kind: state for state in decision.branch_states}

    def contribution(kind: RiskEvidenceKind, points: float | None, max_points: float, summary: str) -> BranchContribution:
        state = states[kind]
        return BranchContribution(
            branch=kind.value,
            input_status=state.input_status.value,
            agreement=state.agreement.value,
            points=points,
            max_points=max_points,
            summary=summary,
        )

    top = next((e for e in events if e.rule_points > 0), None)
    acoustic_summary = (
        "No risk-weighted sound events were detected."
        if top is None
        else (
            f"Strongest weighted event: {top.display_name} "
            f"({top.rule_points:g} points, peak score {top.peak_score:.2f})"
            + (" plus the high-confidence bonus." if any(e.high_confidence and e.rule_points > 0 for e in events) else ".")
        )
    )
    language_summary = (
        "No language findings."
        if not findings
        else ", ".join(f"{f.display_name} (+{f.rule_points:g})" for f in findings) + "."
    )
    contributions = (
        contribution(RiskEvidenceKind.ACOUSTIC, acoustic_pts, 0 if rules is None else rules.acoustic.max_points, acoustic_summary),
        contribution(RiskEvidenceKind.SPEECH, speech_pts, 0 if rules is None else rules.speech.max_points, speech_summary),
        contribution(RiskEvidenceKind.LANGUAGE, language_pts, 0 if rules is None else rules.language.max_points, language_summary),
    )

    # --- Verdict ----------------------------------------------------------------
    missing = bool(assessment.missing_data.missing_branches)
    outcome = decision.outcome
    if silent_input or missing:
        verdict: Verdict = "inconclusive"
    elif outcome is ConsensusOutcome.ALERT:
        verdict = "elevated_risk_alert_candidate"
    elif outcome is ConsensusOutcome.REVIEW:
        verdict = "possible_risk_review_recommended"
    elif outcome is ConsensusOutcome.LOG:
        verdict = "low_indicators"
    else:
        verdict = "no_indicators_detected"

    score_text = f"{assessment.score:g}/100 ({assessment.severity.value})"
    evidence_bits = [e.display_name for e in events if e.rule_points > 0]
    evidence_bits += [f.display_name for f in findings if f.rule_points > 0]
    evidence_text = ", ".join(dict.fromkeys(evidence_bits)) or "no risk-weighted indicators"

    if verdict == "inconclusive":
        cause = (
            "the recording appears to be silent or nearly silent"
            if silent_input
            else "required evidence is missing"
        )
        answer = (
            f"Inconclusive. The system scored this recording {score_text}, but {cause}, "
            "so the result cannot be relied on. A person should listen to the recording."
        )
    elif verdict == "no_indicators_detected":
        answer = (
            f"No risk indicators were detected (score {score_text}). This does not prove the "
            "recording is safe; it means the models found nothing the scoring rules weight as risky."
        )
    elif verdict == "low_indicators":
        answer = (
            f"Only weak indicators were found (score {score_text}): {evidence_text}. "
            "The result is logged but does not reach the human-review band."
        )
    elif verdict == "possible_risk_review_recommended":
        answer = (
            f"Possible risk. The recording scored {score_text} based on: {evidence_text}. "
            "Human review is recommended before drawing any conclusion."
        )
    else:
        answer = (
            f"Elevated risk indicators. The recording scored {score_text} and independent "
            f"evidence branches agreed ({evidence_text}). A local alert candidate was created "
            "for human review. This is not a confirmed emergency and no notification was sent."
        )

    limitations = list(BASE_LIMITATIONS)
    if states[RiskEvidenceKind.SPEECH].agreement is BranchAgreement.NOT_PERMITTED:
        limitations.append("Speech and language were not analysed (acoustic-only scope), so spoken threats or distress cannot be detected.")
    if silent_input:
        limitations.append("The recording level is below the pipeline's silence floor; quiet or empty audio produces no evidence.")
    if not reproduced:
        limitations.append("The per-branch point breakdown could not be reproduced exactly, so it is not shown.")
    if speech.status is RiskInputStatus.PRESENT:
        limitations.append("Transcription (Whisper tiny.en) is English-only and may mis-hear words in noisy audio.")

    return RiskExplanation(
        report_id=report.report_id,
        verdict=verdict,
        answer=answer,
        risk_score=assessment.score,
        risk_level=assessment.severity.value,
        outcome=outcome.value,
        human_review_recommended=bool(decision.review_required or assessment.human_review_required or verdict == "inconclusive"),
        alert_candidate=decision.alert_candidate,
        acoustic_events=events,
        speech_summary=speech_summary,
        language_findings=findings,
        contributions=contributions,
        breakdown=ScoreBreakdown(
            reproduced=reproduced,
            total_before_cap=total_before_cap,
            total_max_score=100 if rules is None else rules.total_max_score,
            rule_set_id=assessment.scoring_policy.rule_set.rule_set_id,
            rule_set_version=assessment.scoring_policy.rule_set.rule_set_version,
        ),
        risk_reasons=tuple(RISK_REASON_TEXT.get(code, code.value) for code in assessment.reason_codes),
        decision_reasons=tuple(DECISION_REASON_TEXT.get(code, code.value) for code in decision.reason_codes),
        limitations=tuple(limitations),
    )


__all__ = ["RiskExplanation", "explain_report", "EXPLANATION_VERSION"]
