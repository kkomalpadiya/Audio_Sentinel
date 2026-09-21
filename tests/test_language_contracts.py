"""A5.1 contract tests for accepted-only, explainable language evidence."""

import copy
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.language_contracts import (
    AcceptedTranscriptReference,
    LanguageCategory,
    LanguageEvidenceDocument,
    LanguageFinding,
    LanguageReasonCode,
    LanguageRuleKind,
    LanguageRuleMatch,
    LanguageTranscriptAnalysis,
    language_schema_documents,
    write_language_schemas,
)


@pytest.fixture
def language_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (project_root / "docs" / "examples" / "language-evidence.json").read_text(
            encoding="utf-8"
        )
    )


def _match(kind: str = "keyword", start: int = 0, end: int = 4) -> dict[str, object]:
    return {
        "rule_id": f"{kind}-rule-001",
        "rule_kind": kind,
        "start_character": start,
        "end_character": end,
        "matched_text_sha256": "a" * 64,
    }


def _finding(
    category: str,
    reasons: list[str],
    matches: list[dict[str, object]] | None = None,
    finding_id: str = "finding-test-001",
) -> LanguageFinding:
    return LanguageFinding.model_validate(
        {
            "finding_id": finding_id,
            "category": category,
            "reason_codes": reasons,
            "matches": [] if matches is None else matches,
        }
    )


def test_example_validates_and_round_trips(language_data: dict[str, object]) -> None:
    document = LanguageEvidenceDocument.model_validate(language_data)

    assert LanguageEvidenceDocument.model_validate_json(document.model_dump_json()) == document
    assert document.source.processing_scope == "acoustic_and_speech"
    assert document.analysis_method == "versioned_rules"
    assert document.analyses[0].findings[0].category is LanguageCategory.DISTRESS
    assert (
        document.analyses[1].findings[0].category
        is LanguageCategory.CONTEXT_SUPPRESSED
    )


def test_category_and_reason_taxonomies_are_closed_and_risk_free() -> None:
    assert [category.value for category in LanguageCategory] == [
        "no_concerning_match",
        "distress",
        "threat",
        "weapon_reference",
        "ambiguous",
        "context_suppressed",
    ]
    assert [reason.value for reason in LanguageReasonCode] == [
        "no_rule_match",
        "keyword_match",
        "phrase_match",
        "explicit_negation",
        "hypothetical_or_conditional",
        "quoted_or_reported_speech",
        "insufficient_context",
        "conflicting_signals",
    ]
    forbidden = {"risk", "severity", "incident", "alert", "speaker"}
    assert not any(word in item.value for item in LanguageCategory for word in forbidden)


@pytest.mark.parametrize(
    "overrides",
    [
        {"speech_evidence_sha256": "A" * 64},
        {"speech_evidence_sha256": "a" * 63},
        {"processing_scope": "acoustic_only"},
        {"sample_rate_hz": 7999},
        {"num_samples": 0},
        {"unexpected": True},
    ],
)
def test_source_rejects_invalid_provenance(
    language_data: dict[str, object], overrides: dict[str, object]
) -> None:
    language_data["source"].update(overrides)

    with pytest.raises(ValidationError):
        LanguageEvidenceDocument.model_validate(language_data)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rule_set_id": "x"},
        {"rule_set_version": ""},
        {"rule_format_version": "2.0"},
        {"artifact_sha256": "z" * 64},
        {"unexpected": True},
    ],
)
def test_rule_set_descriptor_is_versioned_and_closed(
    language_data: dict[str, object], overrides: dict[str, object]
) -> None:
    language_data["rule_set"].update(overrides)

    with pytest.raises(ValidationError):
        LanguageEvidenceDocument.model_validate(language_data)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("transcript_reliability", "review_required", "accepted"),
        ("downstream_text_allowed", False, "True"),
        ("start_sample", -1, "greater than or equal"),
        ("end_sample", 2000, "greater than start_sample"),
        ("start_seconds", -0.1, "greater than or equal"),
        ("end_seconds", 0.125, "greater than start_seconds"),
        ("transcript_character_count", 0, "greater than 0"),
        ("transcript_character_count", 20_001, "less than or equal"),
        ("transcript_utf8_bytes", 20, "below character count"),
        ("confidence_score", float("nan"), "finite"),
        ("confidence_score", 1.01, "less than or equal"),
    ],
)
def test_transcript_reference_enforces_phase4_handoff_and_bounds(
    language_data: dict[str, object], field: str, value: object, message: str
) -> None:
    transcript = language_data["analyses"][0]["transcript"]
    transcript[field] = value

    with pytest.raises(ValidationError, match=message):
        LanguageEvidenceDocument.model_validate(language_data)


@pytest.mark.parametrize(
    "category,reasons,matches",
    [
        ("distress", ["keyword_match"], [_match("keyword")]),
        ("distress", ["phrase_match"], [_match("phrase")]),
        ("threat", ["keyword_match"], [_match("keyword")]),
        ("weapon_reference", ["phrase_match"], [_match("phrase")]),
        (
            "ambiguous",
            ["keyword_match", "insufficient_context"],
            [_match("keyword")],
        ),
        (
            "ambiguous",
            ["phrase_match", "conflicting_signals"],
            [_match("phrase")],
        ),
        (
            "context_suppressed",
            ["keyword_match", "explicit_negation"],
            [_match("negation", 0, 2), _match("keyword", 3, 7)],
        ),
        (
            "context_suppressed",
            ["phrase_match", "hypothetical_or_conditional"],
            [_match("phrase")],
        ),
        (
            "context_suppressed",
            ["phrase_match", "quoted_or_reported_speech"],
            [_match("phrase")],
        ),
        ("no_concerning_match", ["no_rule_match"], []),
    ],
)
def test_valid_category_explanations_are_accepted(
    category: str, reasons: list[str], matches: list[dict[str, object]]
) -> None:
    finding = _finding(category, reasons, matches)

    assert finding.category.value == category


@pytest.mark.parametrize(
    "category,reasons,matches,message",
    [
        ("no_concerning_match", ["keyword_match"], [_match()], "requires only"),
        ("no_concerning_match", ["no_rule_match"], [_match()], "requires only"),
        ("distress", ["no_rule_match"], [], "reserved"),
        ("distress", ["keyword_match"], [], "keyword rule match"),
        ("threat", ["keyword_match", "insufficient_context"], [_match()], "only match"),
        ("ambiguous", ["keyword_match"], [_match()], "ambiguity reason"),
        (
            "ambiguous",
            ["keyword_match", "explicit_negation"],
            [_match()],
            "ambiguity reason",
        ),
        ("context_suppressed", ["phrase_match"], [_match("phrase")], "suppression"),
        (
            "context_suppressed",
            ["keyword_match", "explicit_negation"],
            [_match("keyword")],
            "negation rule match",
        ),
    ],
)
def test_incompatible_category_explanations_are_rejected(
    category: str,
    reasons: list[str],
    matches: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _finding(category, reasons, matches)


def test_reason_codes_must_be_unique_and_canonically_ordered() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _finding("distress", ["keyword_match", "keyword_match"], [_match()])
    with pytest.raises(ValidationError, match="canonical ordering"):
        _finding(
            "context_suppressed",
            ["explicit_negation", "keyword_match"],
            [_match("negation", 0, 2), _match("keyword", 3, 7)],
        )


def test_rule_match_spans_and_order_are_checked() -> None:
    with pytest.raises(ValidationError, match="greater than start_character"):
        LanguageRuleMatch.model_validate(_match(start=4, end=4))
    with pytest.raises(ValidationError, match="deterministic ordering"):
        _finding(
            "distress",
            ["keyword_match"],
            [_match("keyword", 6, 9), _match("keyword", 0, 4)],
        )


def test_match_span_cannot_exceed_transcript(language_data: dict[str, object]) -> None:
    language_data["analyses"][0]["findings"][0]["matches"][0]["end_character"] = 22

    with pytest.raises(ValidationError, match="cannot exceed"):
        LanguageEvidenceDocument.model_validate(language_data)


def test_no_concerning_match_must_be_the_only_finding(
    language_data: dict[str, object],
) -> None:
    language_data["analyses"][0]["findings"].append(
        {
            "finding_id": "language-finding-0002",
            "category": "no_concerning_match",
            "reason_codes": ["no_rule_match"],
            "matches": [],
        }
    )
    language_data["finding_count"] = 3

    with pytest.raises(ValidationError, match="must be the only"):
        LanguageEvidenceDocument.model_validate(language_data)


def test_finding_ids_and_finding_order_are_deterministic(
    language_data: dict[str, object],
) -> None:
    duplicate = copy.deepcopy(language_data)
    first = duplicate["analyses"][0]["findings"][0]
    second = copy.deepcopy(first)
    duplicate["analyses"][0]["findings"].append(second)
    duplicate["finding_count"] = 3
    with pytest.raises(ValidationError, match="finding IDs must be unique"):
        LanguageEvidenceDocument.model_validate(duplicate)

    reversed_data = copy.deepcopy(language_data)
    earlier = {
        "finding_id": "language-finding-0002",
        "category": "distress",
        "reason_codes": ["keyword_match"],
        "matches": [_match("keyword", 0, 4)],
    }
    reversed_data["analyses"][0]["findings"].append(earlier)
    reversed_data["finding_count"] = 3
    with pytest.raises(ValidationError, match="findings must use deterministic"):
        LanguageEvidenceDocument.model_validate(reversed_data)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("input_transcript_count", 1, "accepted transcript inventory"),
        ("analyzed_transcript_count", 1, "analysis inventory"),
        ("finding_count", 1, "finding inventory"),
        ("created_at", "2026-09-21T12:00:00", "timezone-aware"),
        ("schema_version", "2.0", "1.0"),
        ("document_type", "risk_assessment", "language_evidence"),
        ("analysis_method", "model", "versioned_rules"),
    ],
)
def test_document_summary_and_version_fields_are_verified(
    language_data: dict[str, object], field: str, value: object, message: str
) -> None:
    language_data[field] = value

    with pytest.raises(ValidationError, match=message):
        LanguageEvidenceDocument.model_validate(language_data)


def test_empty_accepted_transcript_inventory_is_valid(
    language_data: dict[str, object],
) -> None:
    language_data["input_transcript_count"] = 0
    language_data["analyzed_transcript_count"] = 0
    language_data["finding_count"] = 0
    language_data["analyses"] = []

    document = LanguageEvidenceDocument.model_validate(language_data)

    assert document.analyses == ()


def test_transcript_inventory_requires_unique_temporal_order(
    language_data: dict[str, object],
) -> None:
    reversed_data = copy.deepcopy(language_data)
    reversed_data["analyses"].reverse()
    with pytest.raises(ValidationError, match="deterministic temporal"):
        LanguageEvidenceDocument.model_validate(reversed_data)

    duplicate = copy.deepcopy(language_data)
    duplicate["analyses"][1]["transcript"]["segment_id"] = "speech-0000"
    with pytest.raises(ValidationError, match="segment IDs must be unique"):
        LanguageEvidenceDocument.model_validate(duplicate)


def test_transcript_spans_are_sample_exact_bounded_and_non_overlapping(
    language_data: dict[str, object],
) -> None:
    wrong_seconds = copy.deepcopy(language_data)
    wrong_seconds["analyses"][0]["transcript"]["start_seconds"] = 0.126
    with pytest.raises(ValidationError, match="seconds must equal"):
        LanguageEvidenceDocument.model_validate(wrong_seconds)

    beyond_source = copy.deepcopy(language_data)
    beyond_source["analyses"][1]["transcript"]["end_sample"] = 32001
    beyond_source["analyses"][1]["transcript"]["end_seconds"] = 32001 / 16000
    with pytest.raises(ValidationError, match="cannot exceed"):
        LanguageEvidenceDocument.model_validate(beyond_source)

    overlapping = copy.deepcopy(language_data)
    overlapping["analyses"][1]["transcript"]["start_sample"] = 9999
    overlapping["analyses"][1]["transcript"]["start_seconds"] = 9999 / 16000
    with pytest.raises(ValidationError, match="must not overlap"):
        LanguageEvidenceDocument.model_validate(overlapping)


def test_models_are_frozen_and_reject_extra_fields(
    language_data: dict[str, object],
) -> None:
    document = LanguageEvidenceDocument.model_validate(language_data)
    with pytest.raises(ValidationError):
        document.finding_count = 99

    language_data["risk_score"] = 100
    with pytest.raises(ValidationError):
        LanguageEvidenceDocument.model_validate(language_data)


def test_serialized_evidence_does_not_copy_transcript_or_risk_fields(
    language_data: dict[str, object],
) -> None:
    serialized = LanguageEvidenceDocument.model_validate(language_data).model_dump_json()

    assert "Please call for help" not in serialized
    assert '"risk_score"' not in serialized
    assert '"severity"' not in serialized
    assert '"incident"' not in serialized
    assert '"alert"' not in serialized
    assert '"speaker"' not in serialized


def test_checked_in_schema_matches_generated_contract(project_root: Path) -> None:
    expected = language_schema_documents()["language-evidence.schema.json"]
    checked_in = json.loads(
        (project_root / "docs" / "schemas" / "v1" / "language-evidence.schema.json")
        .read_text(encoding="utf-8")
    )

    assert checked_in == expected
    assert checked_in["$id"].endswith("/language-evidence.schema.json")
    assert "risk_score" not in json.dumps(checked_in)


def test_schema_export_writes_portable_json(tmp_path: Path) -> None:
    exported = write_language_schemas(tmp_path)
    destination = exported["language-evidence.schema.json"]

    assert destination == tmp_path / "language-evidence.schema.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        language_schema_documents()["language-evidence.schema.json"]
    )


def test_public_records_keep_enum_and_datetime_types(
    language_data: dict[str, object],
) -> None:
    document = LanguageEvidenceDocument.model_validate(language_data)

    assert isinstance(document.created_at, datetime)
    assert document.created_at.tzinfo is not None
    assert document.created_at.astimezone(UTC).utcoffset().total_seconds() == 0
    assert document.analyses[0].transcript.confidence_kind.value == "derived_score"
    assert document.analyses[0].findings[0].matches[0].rule_kind is LanguageRuleKind.PHRASE


def test_standalone_public_types_validate() -> None:
    transcript = AcceptedTranscriptReference(
        segment_id="speech-test-001",
        start_sample=0,
        end_sample=1600,
        start_seconds=0,
        end_seconds=0.1,
        transcript_sha256="b" * 64,
        transcript_character_count=4,
        transcript_utf8_bytes=4,
        confidence_score=0.9,
        confidence_kind="model_score",
    )
    finding = _finding("distress", ["keyword_match"], [_match()])

    analysis = LanguageTranscriptAnalysis(transcript=transcript, findings=(finding,))

    assert analysis.transcript.downstream_text_allowed is True
