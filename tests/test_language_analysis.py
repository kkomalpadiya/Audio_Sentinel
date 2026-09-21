"""A5.2 tests for deterministic accepted-transcript analysis."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
import hashlib
import json

import pytest
from pydantic import ValidationError

from audio_sentinel import language_analysis as analysis
from audio_sentinel.language_contracts import (
    LanguageCategory,
    LanguageReasonCode,
    LanguageRuleKind,
)
from audio_sentinel.language_rules import load_builtin_language_rule_set
from audio_sentinel.speech_contracts import (
    SpeechEvidenceDocument,
    SpeechEvidenceSource,
    SpeechModelDescriptor,
    SpeechReliabilityPolicy,
    SpeechSegmentEvidence,
    SpeechWindowReference,
    TranscriptCandidate,
    TranscriptConfidenceKind,
)
from audio_sentinel.speech_transcription import (
    DownstreamTranscript,
    SpeechOrchestrationSettings,
    SpeechTranscriptionResult,
)


NOW = datetime(2026, 9, 21, 12, tzinfo=UTC)


def speech_result(
    texts: tuple[str, ...] = ("Please help me",),
    *,
    scores: tuple[float, ...] | None = None,
) -> SpeechTranscriptionResult:
    scores = scores or tuple(0.9 for _ in texts)
    policy = SpeechReliabilityPolicy()
    num_samples = max(16_000, len(texts) * 4_000)
    source = SpeechEvidenceSource(
        preparation_manifest_path="prepared/language/manifest.json",
        preparation_manifest_sha256="1" * 64,
        raw_audio_sha256="2" * 64,
        clip_id="language-clip",
        consent_id="language-consent",
        sample_rate_hz=16_000,
        num_samples=num_samples,
    )
    window = SpeechWindowReference(
        window_id="window-0000",
        audio_path="prepared/language/window-0000.wav",
        window_audio_sha256="3" * 64,
        start_sample=0,
        end_sample=num_samples,
        padding_samples=0,
    )
    segments = []
    downstream = []
    for index, (text, score) in enumerate(zip(texts, scores, strict=True)):
        start = index * 4_000
        end = start + 2_000
        candidate = TranscriptCandidate(
            text=text,
            confidence_score=score,
            confidence_kind=TranscriptConfidenceKind.DERIVED_SCORE,
            language="en",
            language_confidence=0.99,
        )
        segment = SpeechSegmentEvidence(
            segment_id=f"speech-{index:04d}",
            source_window_ids=(window.window_id,),
            start_sample=start,
            end_sample=end,
            start_seconds=start / 16_000,
            end_seconds=end / 16_000,
            vad_score=0.9,
            transcript=candidate,
            assessment=policy.assess(candidate),
        )
        segments.append(segment)
        if segment.assessment.downstream_text_allowed:
            downstream.append(
                DownstreamTranscript(
                    segment_id=segment.segment_id,
                    start_sample=start,
                    end_sample=end,
                    start_seconds=start / 16_000,
                    end_seconds=end / 16_000,
                    text=text,
                    confidence_score=score,
                    confidence_kind=candidate.confidence_kind,
                )
            )
    evidence = SpeechEvidenceDocument(
        evidence_id="0" * 64,
        created_at=NOW,
        source=source,
        reliability_policy=policy,
        vad_model=SpeechModelDescriptor(
            model_id="test-vad",
            model_version="1",
            artifact_sha256="4" * 64,
            runtime_distribution="test-runtime",
            runtime_version="1",
        ),
        transcription_model=SpeechModelDescriptor(
            model_id="test-asr",
            model_version="1",
            artifact_sha256="5" * 64,
            runtime_distribution="test-runtime",
            runtime_version="1",
        ),
        input_window_count=1,
        segment_count=len(segments),
        transcribed_segment_count=len(segments),
        windows=(window,),
        segments=tuple(segments),
    )
    evidence = evidence.model_copy(
        update={
            "evidence_id": analysis._canonical_hash(
                analysis._speech_identity_payload(evidence)
            )
        }
    )
    return SpeechTranscriptionResult(
        evidence=evidence,
        settings=SpeechOrchestrationSettings(),
        transcriptions=(),
        downstream_transcripts=tuple(downstream),
    )


def analyze(text: str):
    return analysis.analyze_accepted_transcripts(speech_result((text,)), now=NOW)


@pytest.mark.parametrize(
    ("text", "category", "rule_id", "reason"),
    [
        ("HELP", LanguageCategory.DISTRESS, "distress-keyword-help", LanguageReasonCode.KEYWORD_MATCH),
        ("There is a firearm", LanguageCategory.WEAPON_REFERENCE, "weapon-keyword-firearm", LanguageReasonCode.KEYWORD_MATCH),
        ("I will hurt you", LanguageCategory.THREAT, "threat-phrase-i-will-hurt-you", LanguageReasonCode.PHRASE_MATCH),
        ("ＣＡＬＬ ＡＮ ＡＭＢＵＬＡＮＣＥ", LanguageCategory.DISTRESS, "distress-phrase-call-an-ambulance", LanguageReasonCode.PHRASE_MATCH),
    ],
)
def test_matches_normalized_keywords_and_phrases(text, category, rule_id, reason):
    finding = analyze(text).evidence.analyses[0].findings[0]

    assert finding.category is category
    assert finding.reason_codes == (reason,)
    assert finding.matches[0].rule_id == rule_id


def test_phrase_match_wins_over_contained_keyword_match():
    findings = analyze("Please help me").evidence.analyses[0].findings

    assert len(findings) == 1
    assert findings[0].matches[0].rule_id == "distress-phrase-please-help-me"


def test_distinct_repeated_matches_remain_distinct_and_ordered():
    findings = analyze("help then help").evidence.analyses[0].findings

    assert len(findings) == 2
    assert [item.matches[0].start_character for item in findings] == [0, 10]
    assert findings[0].finding_id != findings[1].finding_id


def test_no_match_is_explicit_and_does_not_mean_safe():
    finding = analyze("ordinary conversation").evidence.analyses[0].findings[0]

    assert finding.category is LanguageCategory.NO_CONCERNING_MATCH
    assert finding.reason_codes == (LanguageReasonCode.NO_RULE_MATCH,)
    assert finding.matches == ()


@pytest.mark.parametrize(
    ("text", "active_rule", "negation_rule"),
    [
        ("do not shoot", "threat-keyword-shoot", "negation-do-not"),
        ("I don’t have a gun", "weapon-phrase-have-a-gun", "negation-dont"),
        ("never going to stab you", "threat-phrase-going-to-stab-you", "negation-never"),
        ("without any dangerous weapon", "weapon-keyword-weapon", "negation-without"),
    ],
)
def test_preceding_negation_suppresses_with_both_exact_matches(
    text, active_rule, negation_rule
):
    finding = analyze(text).evidence.analyses[0].findings[0]

    assert finding.category is LanguageCategory.CONTEXT_SUPPRESSED
    assert finding.reason_codes[-1] is LanguageReasonCode.EXPLICIT_NEGATION
    assert {item.rule_id for item in finding.matches} == {active_rule, negation_rule}
    assert [item.start_character for item in finding.matches] == sorted(
        item.start_character for item in finding.matches
    )


def test_negation_inside_a_distress_phrase_does_not_suppress_that_phrase():
    finding = analyze("I cannot breathe").evidence.analyses[0].findings[0]

    assert finding.category is LanguageCategory.DISTRESS
    assert finding.matches[0].rule_id == "distress-phrase-i-cannot-breathe"


@pytest.mark.parametrize(
    "text",
    [
        "not one two three four gun",
        "not. there is a gun",
        "gun not",
    ],
)
def test_distant_new_sentence_or_following_negation_does_not_suppress(text):
    finding = analyze(text).evidence.analyses[0].findings[-1]

    assert finding.category is LanguageCategory.WEAPON_REFERENCE


def test_nearest_and_longest_negation_is_selected_deterministically():
    finding = analyze("did not shoot").evidence.analyses[0].findings[0]

    assert {item.rule_id for item in finding.matches} == {
        "negation-did-not",
        "threat-keyword-shoot",
    }


def test_match_span_and_hash_refer_to_original_text_not_normalized_text():
    text = "Alert: ＧＵＮ!"
    match = analyze(text).evidence.analyses[0].findings[0].matches[0]

    assert (match.start_character, match.end_character) == (7, 10)
    assert match.matched_text_sha256 == hashlib.sha256("ＧＵＮ".encode()).hexdigest()


def test_phrase_span_includes_original_interior_punctuation():
    text = "Please, help me!"
    match = analyze(text).evidence.analyses[0].findings[0].matches[0]

    assert text[match.start_character : match.end_character] == "Please, help me"


def test_multiple_categories_and_suppression_can_coexist():
    findings = analyze(
        "help but do not shoot and later found a knife"
    ).evidence.analyses[0].findings

    assert [item.category for item in findings] == [
        LanguageCategory.DISTRESS,
        LanguageCategory.CONTEXT_SUPPRESSED,
        LanguageCategory.WEAPON_REFERENCE,
    ]


def test_only_phase_four_accepted_handoff_is_analyzed():
    result = analysis.analyze_accepted_transcripts(
        speech_result(("help", "gun", "kill"), scores=(0.9, 0.7, 0.49)), now=NOW
    )

    assert result.evidence.input_transcript_count == 1
    assert result.evidence.analyses[0].transcript.segment_id == "speech-0000"


def test_empty_accepted_inventory_is_valid_but_has_no_no_match_finding():
    result = analysis.analyze_accepted_transcripts(
        speech_result(("help",), scores=(0.7,)), now=NOW
    )

    assert result.evidence.input_transcript_count == 0
    assert result.evidence.finding_count == 0
    assert result.evidence.analyses == ()


def test_output_provenance_pins_speech_and_rule_artifacts():
    speech = speech_result(("help",))
    result = analysis.analyze_accepted_transcripts(speech, now=NOW)
    loaded = load_builtin_language_rule_set()

    assert result.evidence.source.speech_evidence_id == speech.evidence.evidence_id
    assert result.evidence.source.speech_evidence_sha256 == analysis._canonical_hash(
        speech.evidence.model_dump(mode="json")
    )
    assert result.evidence.rule_set == loaded.as_descriptor()


def test_evidence_identity_is_stable_across_creation_times():
    speech = speech_result(("help",))
    first = analysis.analyze_accepted_transcripts(speech, now=NOW)
    second = analysis.analyze_accepted_transcripts(
        speech, now=datetime(2026, 9, 22, tzinfo=UTC)
    )

    assert first.evidence.evidence_id == second.evidence.evidence_id
    assert first.evidence.created_at != second.evidence.created_at


def test_summary_is_json_ready_and_does_not_repeat_transcript_text():
    secret = "please help me private phrase"
    result = analyze(secret)
    document = json.dumps(result.to_summary(), sort_keys=True)

    assert json.loads(document)["evidence"]["document_type"] == "language_evidence"
    assert secret not in document
    assert "risk" not in document and "alert" not in document


def test_result_settings_and_evidence_are_immutable():
    result = analyze("help")

    with pytest.raises(FrozenInstanceError):
        result.settings = analysis.LanguageAnalysisSettings(max_findings=2)
    with pytest.raises(ValidationError):
        result.evidence.finding_count = 99


@pytest.mark.parametrize(
    "damage",
    ["evidence_id", "missing_handoff", "changed_text", "extra_handoff", "list_handoff"],
)
def test_rejects_tampered_phase_four_input(damage):
    speech = speech_result(("help",))
    if damage == "evidence_id":
        speech = replace(
            speech,
            evidence=speech.evidence.model_copy(update={"evidence_id": "f" * 64}),
        )
    elif damage == "missing_handoff":
        speech = replace(speech, downstream_transcripts=())
    elif damage == "changed_text":
        changed = replace(speech.downstream_transcripts[0], text="gun")
        speech = replace(speech, downstream_transcripts=(changed,))
    elif damage == "extra_handoff":
        speech = replace(
            speech,
            downstream_transcripts=(
                *speech.downstream_transcripts,
                speech.downstream_transcripts[0],
            ),
        )
    else:
        speech = replace(speech, downstream_transcripts=list(speech.downstream_transcripts))

    with pytest.raises(analysis.LanguageAnalysisError) as error:
        analysis.analyze_accepted_transcripts(speech, now=NOW)

    assert error.value.code == "invalid_speech_input"


@pytest.mark.parametrize(
    ("settings", "code"),
    [
        (analysis.LanguageAnalysisSettings(max_transcripts=1), "too_many_transcripts"),
        (analysis.LanguageAnalysisSettings(max_total_text_bytes=1), "input_too_large"),
        (analysis.LanguageAnalysisSettings(max_total_tokens=1), "input_too_large"),
        (analysis.LanguageAnalysisSettings(max_rule_checks=1), "analysis_too_large"),
        (analysis.LanguageAnalysisSettings(max_findings=1), "analysis_too_large"),
    ],
)
def test_resource_limits_fail_without_partial_output(settings, code):
    speech = speech_result(("help", "gun"))

    with pytest.raises(analysis.LanguageAnalysisError) as error:
        analysis.analyze_accepted_transcripts(speech, settings=settings, now=NOW)

    assert error.value.code == code


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_transcripts": 0},
        {"max_total_text_bytes": True},
        {"max_total_tokens": 0},
        {"max_rule_checks": 0},
        {"max_findings": 0},
        {"unexpected": 1},
    ],
)
def test_invalid_analysis_settings_are_rejected(overrides):
    with pytest.raises(ValidationError):
        analysis.LanguageAnalysisSettings(**overrides)


def test_naive_creation_time_is_rejected_before_analysis():
    with pytest.raises(analysis.LanguageAnalysisError) as error:
        analysis.analyze_accepted_transcripts(
            speech_result(("private transcript",)), now=datetime(2026, 9, 21)
        )

    assert error.value.code == "invalid_time"
    assert "private" not in str(error.value)


def test_invalid_rule_set_failure_does_not_disclose_transcript():
    loaded = replace(load_builtin_language_rule_set(), artifact_sha256="bad")

    with pytest.raises(analysis.LanguageAnalysisError) as error:
        analysis.analyze_accepted_transcripts(
            speech_result(("private transcript",)), rule_set=loaded, now=NOW
        )

    assert error.value.code == "invalid_rule_set"
    assert "private" not in str(error.value)


def test_rule_kind_is_preserved_in_portable_match():
    match = analyze("I will kill you").evidence.analyses[0].findings[0].matches[0]

    assert match.rule_kind is LanguageRuleKind.PHRASE
