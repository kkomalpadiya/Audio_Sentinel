import copy
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.speech_contracts import (
    SpeechEvidenceDocument,
    SpeechReliabilityPolicy,
    SpeechReliabilityReason,
    TranscriptCandidate,
    TranscriptConfidenceKind,
    TranscriptReliability,
    speech_schema_documents,
    write_speech_schemas,
)


@pytest.fixture
def speech_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (project_root / "docs" / "examples" / "speech-evidence.json").read_text(
            encoding="utf-8"
        )
    )


def transcript(score: float) -> TranscriptCandidate:
    return TranscriptCandidate(
        text="Please call for help.",
        confidence_score=score,
        confidence_kind=TranscriptConfidenceKind.DERIVED_SCORE,
        language="en",
        language_confidence=0.9,
    )


def test_example_validates_and_round_trips(speech_data: dict[str, object]) -> None:
    document = SpeechEvidenceDocument.model_validate(speech_data)

    assert SpeechEvidenceDocument.model_validate_json(document.model_dump_json()) == document
    assert document.source.processing_scope == "acoustic_and_speech"
    assert document.segment_count == 2
    assert document.transcribed_segment_count == 2
    assert document.segments[0].assessment.downstream_text_allowed is True
    assert document.segments[1].assessment.human_review_required is True


@pytest.mark.parametrize(
    "score,reliability,reason,allowed,review",
    [
        (0.0, TranscriptReliability.REJECTED_LOW_CONFIDENCE,
         SpeechReliabilityReason.BELOW_REVIEW_THRESHOLD, False, False),
        (0.499999, TranscriptReliability.REJECTED_LOW_CONFIDENCE,
         SpeechReliabilityReason.BELOW_REVIEW_THRESHOLD, False, False),
        (0.5, TranscriptReliability.REVIEW_REQUIRED,
         SpeechReliabilityReason.BELOW_ACCEPTANCE_THRESHOLD, False, True),
        (0.799999, TranscriptReliability.REVIEW_REQUIRED,
         SpeechReliabilityReason.BELOW_ACCEPTANCE_THRESHOLD, False, True),
        (0.8, TranscriptReliability.ACCEPTED,
         SpeechReliabilityReason.MEETS_ACCEPTANCE_THRESHOLD, True, False),
        (1.0, TranscriptReliability.ACCEPTED,
         SpeechReliabilityReason.MEETS_ACCEPTANCE_THRESHOLD, True, False),
    ],
)
def test_reliability_boundaries_are_explicit(
    score: float,
    reliability: TranscriptReliability,
    reason: SpeechReliabilityReason,
    allowed: bool,
    review: bool,
) -> None:
    assessment = SpeechReliabilityPolicy().assess(transcript(score))

    assert assessment.reliability is reliability
    assert assessment.reason_codes == (reason,)
    assert assessment.downstream_text_allowed is allowed
    assert assessment.human_review_required is review


def test_missing_transcript_is_not_silently_treated_as_safe() -> None:
    assessment = SpeechReliabilityPolicy().assess(None)

    assert assessment.reliability is TranscriptReliability.NOT_TRANSCRIBED
    assert assessment.reason_codes == (SpeechReliabilityReason.TRANSCRIPT_NOT_ATTEMPTED,)
    assert assessment.downstream_text_allowed is False
    assert assessment.human_review_required is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"transcript_review_threshold": 0.8, "transcript_acceptance_threshold": 0.8},
        {"transcript_review_threshold": 0.9, "transcript_acceptance_threshold": 0.8},
        {"vad_speech_threshold": -0.1},
        {"vad_speech_threshold": 1.1},
        {"transcript_review_threshold": float("nan")},
        {"transcript_acceptance_threshold": float("inf")},
        {"policy_version": "2.0"},
        {"unexpected": True},
    ],
)
def test_invalid_reliability_policies_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SpeechReliabilityPolicy(**overrides)


def test_document_rejects_assessment_that_disagrees_with_policy(
    speech_data: dict[str, object],
) -> None:
    segment = speech_data["segments"][1]
    segment["assessment"] = {
        "reliability": "accepted",
        "reason_codes": ["meets_acceptance_threshold"],
        "downstream_text_allowed": True,
        "human_review_required": False,
    }

    with pytest.raises(ValidationError, match="assessment does not match"):
        SpeechEvidenceDocument.model_validate(speech_data)


def test_document_rejects_vad_score_below_recorded_threshold(
    speech_data: dict[str, object],
) -> None:
    speech_data["segments"][0]["vad_score"] = 0.599999

    with pytest.raises(ValidationError, match="below vad_speech_threshold"):
        SpeechEvidenceDocument.model_validate(speech_data)


def test_vad_threshold_boundary_is_inclusive(speech_data: dict[str, object]) -> None:
    speech_data["segments"][0]["vad_score"] = 0.6

    assert SpeechEvidenceDocument.model_validate(speech_data).segments[0].vad_score == 0.6


def test_vad_only_document_can_record_not_transcribed_segments(
    speech_data: dict[str, object],
) -> None:
    speech_data["transcription_model"] = None
    speech_data["transcribed_segment_count"] = 0
    for segment in speech_data["segments"]:
        segment["transcript"] = None
        segment["assessment"] = {
            "reliability": "not_transcribed",
            "reason_codes": ["transcript_not_attempted"],
            "downstream_text_allowed": False,
            "human_review_required": False,
        }

    document = SpeechEvidenceDocument.model_validate(speech_data)

    assert document.transcription_model is None
    assert all(segment.transcript is None for segment in document.segments)


def test_transcribed_segment_requires_model_provenance(speech_data: dict[str, object]) -> None:
    speech_data["transcription_model"] = None

    with pytest.raises(ValidationError, match="transcription_model provenance"):
        SpeechEvidenceDocument.model_validate(speech_data)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("input_window_count", 1, "window inventory"),
        ("segment_count", 1, "segment inventory"),
        ("transcribed_segment_count", 1, "segments with transcripts"),
        ("created_at", "2026-09-19T12:00:00", "timezone-aware"),
    ],
)
def test_document_summary_fields_are_verified(
    speech_data: dict[str, object], field: str, value: object, message: str
) -> None:
    speech_data[field] = value

    with pytest.raises(ValidationError, match=message):
        SpeechEvidenceDocument.model_validate(speech_data)


def test_window_and_segment_inventories_require_deterministic_unique_order(
    speech_data: dict[str, object],
) -> None:
    reversed_windows = copy.deepcopy(speech_data)
    reversed_windows["windows"].reverse()
    with pytest.raises(ValidationError, match="windows must use deterministic"):
        SpeechEvidenceDocument.model_validate(reversed_windows)

    duplicate_windows = copy.deepcopy(speech_data)
    duplicate_windows["windows"][1]["window_id"] = "window-0000"
    with pytest.raises(ValidationError, match="window IDs must be unique"):
        SpeechEvidenceDocument.model_validate(duplicate_windows)

    reversed_segments = copy.deepcopy(speech_data)
    reversed_segments["segments"].reverse()
    with pytest.raises(ValidationError, match="segments must use deterministic"):
        SpeechEvidenceDocument.model_validate(reversed_segments)

    duplicate_segments = copy.deepcopy(speech_data)
    duplicate_segments["segments"][1]["segment_id"] = "speech-0000"
    with pytest.raises(ValidationError, match="segment IDs must be unique"):
        SpeechEvidenceDocument.model_validate(duplicate_segments)


def test_segment_spans_are_sample_exact_bounded_and_non_overlapping(
    speech_data: dict[str, object],
) -> None:
    wrong_seconds = copy.deepcopy(speech_data)
    wrong_seconds["segments"][0]["start_seconds"] = 0.126
    with pytest.raises(ValidationError, match="seconds must equal sample offsets"):
        SpeechEvidenceDocument.model_validate(wrong_seconds)

    beyond_source = copy.deepcopy(speech_data)
    beyond_source["segments"][1]["end_sample"] = 32001
    beyond_source["segments"][1]["end_seconds"] = 32001 / 16000
    with pytest.raises(ValidationError, match="cannot exceed"):
        SpeechEvidenceDocument.model_validate(beyond_source)

    overlapping = copy.deepcopy(speech_data)
    overlapping["segments"][1]["start_sample"] = 9999
    overlapping["segments"][1]["start_seconds"] = 9999 / 16000
    with pytest.raises(ValidationError, match="must not overlap"):
        SpeechEvidenceDocument.model_validate(overlapping)


def test_segment_source_window_links_are_checked(speech_data: dict[str, object]) -> None:
    unknown = copy.deepcopy(speech_data)
    unknown["segments"][0]["source_window_ids"] = ["window-missing"]
    with pytest.raises(ValidationError, match="unknown source window"):
        SpeechEvidenceDocument.model_validate(unknown)

    duplicate = copy.deepcopy(speech_data)
    duplicate["segments"][0]["source_window_ids"] = ["window-0000", "window-0000"]
    with pytest.raises(ValidationError, match="source_window_ids must be unique"):
        SpeechEvidenceDocument.model_validate(duplicate)

    no_intersection = copy.deepcopy(speech_data)
    no_intersection["segments"][0]["source_window_ids"] = ["window-0001"]
    with pytest.raises(ValidationError, match="must overlap"):
        SpeechEvidenceDocument.model_validate(no_intersection)

    reversed_references = copy.deepcopy(speech_data)
    reversed_references["segments"][1]["source_window_ids"].reverse()
    with pytest.raises(ValidationError, match="deterministic temporal ordering"):
        SpeechEvidenceDocument.model_validate(reversed_references)

    coverage_gap = copy.deepcopy(speech_data)
    coverage_gap["windows"][0]["end_sample"] = 19_000
    coverage_gap["windows"][1]["start_sample"] = 20_000
    with pytest.raises(ValidationError, match="cover the complete segment"):
        SpeechEvidenceDocument.model_validate(coverage_gap)


def test_source_requires_speech_consent_and_safe_manifest_path(
    speech_data: dict[str, object],
) -> None:
    wrong_scope = copy.deepcopy(speech_data)
    wrong_scope["source"]["processing_scope"] = "acoustic_only"
    with pytest.raises(ValidationError):
        SpeechEvidenceDocument.model_validate(wrong_scope)

    traversal = copy.deepcopy(speech_data)
    traversal["source"]["preparation_manifest_path"] = "../manifest.json"
    with pytest.raises(ValidationError, match="relative POSIX"):
        SpeechEvidenceDocument.model_validate(traversal)

    wrong_suffix = copy.deepcopy(speech_data)
    wrong_suffix["source"]["preparation_manifest_path"] = "prepared/manifest.txt"
    with pytest.raises(ValidationError, match="must end in .json"):
        SpeechEvidenceDocument.model_validate(wrong_suffix)


@pytest.mark.parametrize(
    "updates",
    [
        {"text": "   "},
        {"text": " leading"},
        {"text": "bad\rline"},
        {"confidence_score": -0.1},
        {"confidence_score": float("nan")},
        {"language": "en", "language_confidence": None},
        {"language": None, "language_confidence": 0.9},
        {"language": "english", "language_confidence": 0.9},
        {"confidence_kind": "probability"},
        {"extra": "not allowed"},
    ],
)
def test_invalid_transcript_candidates_are_rejected(updates: dict[str, object]) -> None:
    values: dict[str, object] = {
        "text": "Please call for help.",
        "confidence_score": 0.9,
        "confidence_kind": "model_score",
        "language": "en",
        "language_confidence": 0.9,
    }
    values.update(updates)

    with pytest.raises(ValidationError):
        TranscriptCandidate.model_validate(values)


def test_records_are_immutable(speech_data: dict[str, object]) -> None:
    document = SpeechEvidenceDocument.model_validate(speech_data)

    with pytest.raises(ValidationError):
        document.segment_count = 99


def test_schema_is_portable_versioned_and_has_no_risk_decision() -> None:
    schema = speech_schema_documents()["speech-evidence.schema.json"]

    assert schema["$id"].endswith("/v1/speech-evidence.schema.json")
    assert schema["additionalProperties"] is False
    assert "risk_score" not in schema["properties"]
    assert "risk_level" not in schema["properties"]
    assert "language_category" not in schema["properties"]
    assert schema["properties"]["document_type"]["const"] == "speech_evidence_candidates"


def test_schema_export_matches_checked_in_contract(tmp_path: Path, project_root: Path) -> None:
    exported = write_speech_schemas(tmp_path)
    checked_in = json.loads(
        (project_root / "docs" / "schemas" / "v1" / "speech-evidence.schema.json")
        .read_text(encoding="utf-8")
    )

    assert set(exported) == {"speech-evidence.schema.json"}
    assert json.loads(exported["speech-evidence.schema.json"].read_text(encoding="utf-8")) == checked_in
    assert checked_in == speech_schema_documents()["speech-evidence.schema.json"]


def test_checked_in_example_has_no_raw_audio_or_personal_identity_fields(
    speech_data: dict[str, object],
) -> None:
    serialized = json.dumps(speech_data)

    assert "audio_bytes" not in serialized
    assert "speaker_name" not in serialized
    assert "speaker_embedding" not in serialized
    assert "risk_score" not in serialized
    assert datetime.fromisoformat(speech_data["created_at"].replace("Z", "+00:00")).tzinfo is UTC
