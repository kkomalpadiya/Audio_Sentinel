"""A6.1 contract tests for risk inputs, score bands, and missing-data policy."""

import copy
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.contracts import RiskLevel
from audio_sentinel.risk_contracts import (
    RiskAssessmentDocument,
    RiskEvidenceReference,
    RiskInputStatus,
    RiskMissingDataAssessment,
    RiskMissingDataPolicy,
    RiskReasonCode,
    RiskRuleSetDescriptor,
    RiskScoringPolicy,
    default_risk_severity_bands,
    risk_schema_documents,
    severity_for_score,
    write_risk_schemas,
)


@pytest.fixture
def risk_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (project_root / "docs" / "examples" / "risk-assessment.json").read_text(
            encoding="utf-8"
        )
    )


def test_example_validates_and_round_trips(risk_data: dict[str, object]) -> None:
    document = RiskAssessmentDocument.model_validate(risk_data)

    assert RiskAssessmentDocument.model_validate_json(document.model_dump_json()) == document
    assert document.document_type == "risk_assessment"
    assert document.score == 82
    assert document.severity is RiskLevel.CRITICAL
    assert document.inputs.acoustic.status is RiskInputStatus.PRESENT
    assert document.inputs.language.signals[0].category.value == "distress"
    assert document.human_review_required is True


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, RiskLevel.NONE),
        (0.1, RiskLevel.LOW),
        (24, RiskLevel.LOW),
        (25, RiskLevel.MEDIUM),
        (49, RiskLevel.MEDIUM),
        (50, RiskLevel.HIGH),
        (74, RiskLevel.HIGH),
        (75, RiskLevel.CRITICAL),
        (100, RiskLevel.CRITICAL),
    ],
)
def test_score_boundaries_are_explicit(score: float, expected: RiskLevel) -> None:
    assert severity_for_score(score) is expected


@pytest.mark.parametrize("score", [-0.01, 100.01, float("nan"), float("inf")])
def test_invalid_scores_are_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        severity_for_score(score)


def test_default_policy_pins_v1_score_range_and_bands() -> None:
    policy = RiskScoringPolicy(
        rule_set=RiskRuleSetDescriptor(
            rule_set_id="test-risk-rules",
            rule_set_version="1.0.0",
            artifact_sha256="a" * 64,
        )
    )

    assert policy.score_min == 0
    assert policy.score_max == 100
    assert policy.severity_bands == default_risk_severity_bands()
    assert [band.risk_level for band in policy.severity_bands] == list(RiskLevel)
    assert policy.missing_data_policy.required_branches == ("acoustic",)


def test_policy_rejects_changed_or_reordered_severity_bands() -> None:
    policy = RiskScoringPolicy(
        rule_set=RiskRuleSetDescriptor(
            rule_set_id="test-risk-rules",
            rule_set_version="1.0.0",
            artifact_sha256="a" * 64,
        )
    )
    payload = policy.model_dump(mode="json")
    payload["severity_bands"][1]["max_score"] = 25
    with pytest.raises(ValidationError, match="documented defaults"):
        RiskScoringPolicy.model_validate(payload)

    payload = policy.model_dump(mode="json")
    payload["severity_bands"].reverse()
    with pytest.raises(ValidationError, match="canonical"):
        RiskScoringPolicy.model_validate(payload)


def test_evidence_reference_kind_must_match_document_type() -> None:
    with pytest.raises(ValidationError, match="document_type"):
        RiskEvidenceReference(
            kind="language",
            document_type="speech_evidence_candidates",
            evidence_id="language-example-001",
            evidence_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "branch,path,count_field",
    [
        ("acoustic", ["event_count"], "event_count"),
        ("speech", ["segment_count"], "segment_count"),
        ("language", ["finding_count"], "finding_count"),
    ],
)
def test_non_present_branches_cannot_carry_scoring_content(
    risk_data: dict[str, object],
    branch: str,
    path: list[str],
    count_field: str,
) -> None:
    risk_data["inputs"][branch]["status"] = "missing"
    risk_data["inputs"][branch]["evidence"] = None
    risk_data["inputs"][branch][count_field] = 1

    with pytest.raises(ValidationError, match="non-present"):
        RiskAssessmentDocument.model_validate(risk_data)


def test_present_branch_requires_matching_evidence_reference(
    risk_data: dict[str, object],
) -> None:
    risk_data["inputs"]["acoustic"]["evidence"] = None

    with pytest.raises(ValidationError, match="requires evidence"):
        RiskAssessmentDocument.model_validate(risk_data)


def test_acoustic_counts_peak_and_order_are_validated(
    risk_data: dict[str, object],
) -> None:
    wrong_count = copy.deepcopy(risk_data)
    wrong_count["inputs"]["acoustic"]["event_count"] = 2
    with pytest.raises(ValidationError, match="event_count"):
        RiskAssessmentDocument.model_validate(wrong_count)

    wrong_peak = copy.deepcopy(risk_data)
    wrong_peak["inputs"]["acoustic"]["max_peak_score"] = 0.5
    with pytest.raises(ValidationError, match="max_peak_score"):
        RiskAssessmentDocument.model_validate(wrong_peak)

    wrong_seconds = copy.deepcopy(risk_data)
    wrong_seconds["inputs"]["acoustic"]["signals"][0]["end_seconds"] = 1
    with pytest.raises(ValidationError, match="seconds must equal"):
        RiskAssessmentDocument.model_validate(wrong_seconds)


def test_speech_counts_are_bounded(risk_data: dict[str, object]) -> None:
    risk_data["inputs"]["speech"]["accepted_transcript_count"] = 3

    with pytest.raises(ValidationError, match="cannot exceed segment_count"):
        RiskAssessmentDocument.model_validate(risk_data)


def test_language_signals_require_unique_ids_and_speech_provenance(
    risk_data: dict[str, object],
) -> None:
    duplicate = copy.deepcopy(risk_data)
    duplicate["inputs"]["language"]["signals"].append(
        copy.deepcopy(duplicate["inputs"]["language"]["signals"][0])
    )
    duplicate["inputs"]["language"]["finding_count"] = 2
    with pytest.raises(ValidationError, match="finding IDs must be unique"):
        RiskAssessmentDocument.model_validate(duplicate)

    no_speech = copy.deepcopy(risk_data)
    no_speech["inputs"]["speech"] = {
        "status": "missing",
        "evidence": None,
        "segment_count": 0,
        "accepted_transcript_count": 0,
        "review_required_transcript_count": 0,
        "max_vad_score": None,
    }
    no_speech["missing_data"] = {
        "missing_branches": ["speech"],
        "not_permitted_branches": [],
        "review_required": True,
        "reason_codes": ["missing_speech_evidence", "missing_data_review_required"],
    }
    with pytest.raises(ValidationError, match="language input requires present speech"):
        RiskAssessmentDocument.model_validate(no_speech)


def test_acoustic_only_consent_blocks_speech_and_language_presence(
    risk_data: dict[str, object],
) -> None:
    risk_data["inputs"]["source"]["processing_scope"] = "acoustic_only"

    with pytest.raises(ValidationError, match="speech input cannot be present"):
        RiskAssessmentDocument.model_validate(risk_data)


def test_missing_data_policy_records_review_instead_of_treating_absence_as_safe(
    risk_data: dict[str, object],
) -> None:
    risk_data["inputs"]["language"] = {
        "status": "missing",
        "evidence": None,
        "finding_count": 0,
        "signals": [],
    }
    risk_data["score"] = 24
    risk_data["severity"] = "low"
    risk_data["reason_codes"] = [
        "acoustic_event_candidate",
        "acoustic_high_confidence",
        "speech_present",
        "transcript_review_required",
        "missing_language_evidence",
        "missing_data_review_required",
    ]
    risk_data["missing_data"] = {
        "missing_branches": ["language"],
        "not_permitted_branches": [],
        "review_required": True,
        "reason_codes": ["missing_language_evidence", "missing_data_review_required"],
    }

    document = RiskAssessmentDocument.model_validate(risk_data)

    assert document.missing_data.review_required is True
    assert document.human_review_required is True
    assert document.missing_data.missing_branches == ("language",)


def test_missing_data_summary_must_match_branch_statuses(
    risk_data: dict[str, object],
) -> None:
    risk_data["missing_data"]["missing_branches"] = ["language"]

    with pytest.raises(ValidationError, match="missing_branches"):
        RiskAssessmentDocument.model_validate(risk_data)


def test_review_required_missing_data_must_set_human_review(
    risk_data: dict[str, object],
) -> None:
    risk_data["inputs"]["acoustic"] = {
        "status": "missing",
        "evidence": None,
        "event_count": 0,
        "max_peak_score": None,
        "signals": [],
    }
    risk_data["score"] = 0
    risk_data["severity"] = "none"
    risk_data["reason_codes"] = [
        "missing_acoustic_evidence",
        "missing_data_review_required",
    ]
    risk_data["missing_data"] = {
        "missing_branches": ["acoustic"],
        "not_permitted_branches": [],
        "review_required": True,
        "reason_codes": ["missing_acoustic_evidence", "missing_data_review_required"],
    }
    risk_data["human_review_required"] = False

    with pytest.raises(ValidationError, match="human_review_required"):
        RiskAssessmentDocument.model_validate(risk_data)


def test_reason_codes_must_be_unique_and_canonically_ordered(
    risk_data: dict[str, object],
) -> None:
    duplicated = copy.deepcopy(risk_data)
    duplicated["reason_codes"].append("language_distress")
    with pytest.raises(ValidationError, match="unique"):
        RiskAssessmentDocument.model_validate(duplicated)

    reordered = copy.deepcopy(risk_data)
    reordered["reason_codes"] = list(reversed(reordered["reason_codes"]))
    with pytest.raises(ValidationError, match="canonical"):
        RiskAssessmentDocument.model_validate(reordered)


def test_document_rejects_wrong_severity_and_naive_time(
    risk_data: dict[str, object],
) -> None:
    wrong_severity = copy.deepcopy(risk_data)
    wrong_severity["severity"] = "high"
    with pytest.raises(ValidationError, match="score band"):
        RiskAssessmentDocument.model_validate(wrong_severity)

    naive_time = copy.deepcopy(risk_data)
    naive_time["created_at"] = "2026-09-25T12:00:00"
    with pytest.raises(ValidationError, match="timezone-aware"):
        RiskAssessmentDocument.model_validate(naive_time)


def test_risk_contract_is_not_consensus_or_alert_output(
    risk_data: dict[str, object],
) -> None:
    serialized = RiskAssessmentDocument.model_validate(risk_data).model_dump_json()

    assert '"consensus"' not in serialized
    assert '"approved"' not in serialized
    assert '"alert_id"' not in serialized
    assert "Please call for help" not in serialized
    assert "audio_bytes" not in serialized
    assert "speaker" not in serialized


def test_missing_data_policy_rejects_duplicate_required_branches() -> None:
    with pytest.raises(ValidationError, match="unique"):
        RiskMissingDataPolicy(required_branches=("acoustic", "acoustic"))


def test_missing_data_assessment_rejects_inconsistent_review_state() -> None:
    with pytest.raises(ValidationError, match="review_required"):
        RiskMissingDataAssessment(
            missing_branches=("language",),
            not_permitted_branches=(),
            review_required=False,
            reason_codes=(RiskReasonCode.MISSING_LANGUAGE_EVIDENCE,),
        )


def test_checked_in_schema_matches_generated_contract(project_root: Path) -> None:
    checked_in = json.loads(
        (project_root / "docs" / "schemas" / "v1" / "risk-assessment.schema.json")
        .read_text(encoding="utf-8")
    )

    assert checked_in == risk_schema_documents()["risk-assessment.schema.json"]
    assert checked_in["$id"].endswith("/risk-assessment.schema.json")


def test_schema_export_writes_portable_json(tmp_path: Path) -> None:
    exported = write_risk_schemas(tmp_path)
    destination = exported["risk-assessment.schema.json"]

    assert destination == tmp_path / "risk-assessment.schema.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        risk_schema_documents()["risk-assessment.schema.json"]
    )


def test_public_records_keep_datetime_and_enum_types(
    risk_data: dict[str, object],
) -> None:
    document = RiskAssessmentDocument.model_validate(risk_data)

    assert isinstance(document.created_at, datetime)
    assert document.created_at.tzinfo is not None
    assert document.created_at.astimezone(UTC).utcoffset().total_seconds() == 0
    assert document.reason_codes[0] is RiskReasonCode.ACOUSTIC_EVENT_CANDIDATE
