"""A7.1 consensus policy and outcome contract tests."""

import copy
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.consensus_contracts import (
    BranchAgreement,
    ConsensusDecisionDocument,
    ConsensusOutcome,
    ConsensusPolicy,
    ConsensusReasonCode,
    consensus_schema_documents,
    write_consensus_schemas,
)
from audio_sentinel.risk_contracts import RiskReasonCode


@pytest.fixture
def consensus_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (project_root / "docs" / "examples" / "consensus-decision.json").read_text(
            encoding="utf-8"
        )
    )


def test_example_validates_and_round_trips(consensus_data: dict[str, object]) -> None:
    document = ConsensusDecisionDocument.model_validate(consensus_data)

    assert ConsensusDecisionDocument.model_validate_json(document.model_dump_json()) == document
    assert document.outcome is ConsensusOutcome.ALERT
    assert document.alert_candidate is True
    assert document.review_required is True
    assert document.branch_states[0].agreement is BranchAgreement.SUPPORTS_RISK


def test_v1_policy_pins_thresholds_and_safety_gates() -> None:
    policy = ConsensusPolicy()

    assert policy.no_action_score == 0
    assert policy.log_max_score == 24
    assert policy.review_min_score == 25
    assert policy.alert_min_score == 75
    assert policy.alert_min_supporting_branches == 2
    assert policy.alert_support_eligible_branches == ("acoustic", "language")
    assert policy.outcome_precedence == ("alert", "review", "log", "no_action")
    assert policy.alert_is_local_candidate_only is True


def test_policy_rejects_changed_support_branches_or_precedence() -> None:
    with pytest.raises(ValidationError, match="acoustic and language"):
        ConsensusPolicy(alert_support_eligible_branches=("acoustic", "speech"))

    with pytest.raises(ValidationError, match="precedence"):
        ConsensusPolicy(
            outcome_precedence=("review", "alert", "log", "no_action")
        )


@pytest.mark.parametrize(
    "status,agreement,message",
    [
        ("present", "unavailable", "evaluated"),
        ("missing", "neutral", "unavailable"),
        ("not_permitted", "not_applicable", "retain"),
        ("no_accepted_text", "neutral", "non-applicable"),
    ],
)
def test_branch_state_must_match_input_status(
    consensus_data: dict[str, object], status: str, agreement: str, message: str
) -> None:
    consensus_data["branch_states"][0]["input_status"] = status
    consensus_data["branch_states"][0]["agreement"] = agreement

    with pytest.raises(ValidationError, match=message):
        ConsensusDecisionDocument.model_validate(consensus_data)


def test_branch_inventory_must_be_complete_and_canonical(
    consensus_data: dict[str, object],
) -> None:
    consensus_data["branch_states"] = list(reversed(consensus_data["branch_states"]))

    with pytest.raises(ValidationError, match="canonical order"):
        ConsensusDecisionDocument.model_validate(consensus_data)


def test_speech_presence_cannot_be_labeled_as_risk_support(
    consensus_data: dict[str, object],
) -> None:
    consensus_data["branch_states"][1]["agreement"] = "supports_risk"

    with pytest.raises(ValidationError, match="speech presence alone"):
        ConsensusDecisionDocument.model_validate(consensus_data)


def test_branch_statuses_must_match_risk_snapshot(
    consensus_data: dict[str, object],
) -> None:
    consensus_data["branch_states"][2] = {
        "kind": "language",
        "input_status": "missing",
        "agreement": "unavailable",
    }

    with pytest.raises(ValidationError, match="missing_branches"):
        ConsensusDecisionDocument.model_validate(consensus_data)


@pytest.mark.parametrize(
    "changes,message",
    [
        (
            (
                ("risk_assessment", "score", 74),
                ("risk_assessment", "severity", "high"),
            ),
            "safety gate",
        ),
        ((("risk_assessment", "severity", "high"),), "Phase 6 score"),
        ((("branch_states", 2, "agreement", "neutral"),), "safety gate"),
        ((("branch_states", 1, "agreement", "conflicts_risk"),), "safety gate"),
    ],
)
def test_alert_requires_critical_multi_branch_agreement_without_conflict(
    consensus_data: dict[str, object],
    changes: tuple[tuple[object, ...], ...],
    message: str,
) -> None:
    for change in changes:
        _set_nested(consensus_data, change)

    with pytest.raises(ValidationError, match=message):
        ConsensusDecisionDocument.model_validate(consensus_data)


@pytest.mark.parametrize(
    "blocking_reason",
    [
        "transcript_review_required",
        "language_ambiguous",
        "missing_data_review_required",
    ],
)
def test_risk_uncertainty_blocks_alert(
    consensus_data: dict[str, object], blocking_reason: str
) -> None:
    reasons = consensus_data["risk_assessment"]["reason_codes"]
    reasons.append(blocking_reason)
    order = [reason.value for reason in RiskReasonCode]
    reasons.sort(key=order.index)

    with pytest.raises(ValidationError, match="safety gate"):
        ConsensusDecisionDocument.model_validate(consensus_data)


def test_alert_requires_complete_reason_receipt(
    consensus_data: dict[str, object],
) -> None:
    consensus_data["reason_codes"].remove("alert_candidate")

    with pytest.raises(ValidationError, match="required reason"):
        ConsensusDecisionDocument.model_validate(consensus_data)


def test_alert_candidate_and_review_flags_follow_outcome(
    consensus_data: dict[str, object],
) -> None:
    wrong_alert = copy.deepcopy(consensus_data)
    wrong_alert["alert_candidate"] = False
    with pytest.raises(ValidationError, match="alert_candidate"):
        ConsensusDecisionDocument.model_validate(wrong_alert)

    wrong_review = copy.deepcopy(consensus_data)
    wrong_review["review_required"] = False
    with pytest.raises(ValidationError, match="review_required"):
        ConsensusDecisionDocument.model_validate(wrong_review)


def test_no_action_requires_zero_score_and_no_support_or_review(
    consensus_data: dict[str, object],
) -> None:
    payload = _outcome_payload(consensus_data, score=0, severity="none")
    payload["outcome"] = "no_action"
    payload["reason_codes"] = ["no_risk_evidence"]
    payload["review_required"] = False
    payload["alert_candidate"] = False
    payload["risk_assessment"]["reason_codes"] = ["no_risk_evidence"]
    payload["risk_assessment"]["human_review_required"] = False

    document = ConsensusDecisionDocument.model_validate(payload)
    assert document.outcome is ConsensusOutcome.NO_ACTION

    payload["risk_assessment"]["score"] = 0.1
    payload["risk_assessment"]["severity"] = "low"
    with pytest.raises(ValidationError, match="zero risk score"):
        ConsensusDecisionDocument.model_validate(payload)


def test_log_requires_low_positive_score_without_review_trigger(
    consensus_data: dict[str, object],
) -> None:
    payload = _outcome_payload(consensus_data, score=10, severity="low")
    payload["outcome"] = "log"
    payload["reason_codes"] = ["low_risk_logged"]
    payload["review_required"] = False
    payload["alert_candidate"] = False
    payload["risk_assessment"]["reason_codes"] = ["acoustic_event_candidate"]
    payload["risk_assessment"]["human_review_required"] = False

    document = ConsensusDecisionDocument.model_validate(payload)
    assert document.outcome is ConsensusOutcome.LOG

    payload["risk_assessment"]["human_review_required"] = True
    with pytest.raises(ValidationError, match="review trigger"):
        ConsensusDecisionDocument.model_validate(payload)


def test_review_accepts_score_or_safety_trigger(
    consensus_data: dict[str, object],
) -> None:
    payload = _outcome_payload(consensus_data, score=50, severity="high")
    payload["outcome"] = "review"
    payload["reason_codes"] = ["score_review_required"]
    payload["review_required"] = True
    payload["alert_candidate"] = False
    payload["risk_assessment"]["human_review_required"] = True

    assert ConsensusDecisionDocument.model_validate(payload).outcome is ConsensusOutcome.REVIEW

    payload["risk_assessment"]["score"] = 10
    payload["risk_assessment"]["severity"] = "low"
    payload["risk_assessment"]["human_review_required"] = False
    with pytest.raises(ValidationError, match="score or safety trigger"):
        ConsensusDecisionDocument.model_validate(payload)


def test_review_requires_reasons_for_each_active_safety_trigger(
    consensus_data: dict[str, object],
) -> None:
    payload = _outcome_payload(consensus_data, score=82, severity="critical")
    payload["outcome"] = "review"
    payload["reason_codes"] = ["score_review_required"]
    payload["review_required"] = True
    payload["alert_candidate"] = False
    payload["risk_assessment"]["human_review_required"] = True

    with pytest.raises(ValidationError, match="required reason"):
        ConsensusDecisionDocument.model_validate(payload)

    payload["reason_codes"] = [
        "score_review_required",
        "insufficient_alert_agreement",
    ]
    assert ConsensusDecisionDocument.model_validate(payload).outcome is ConsensusOutcome.REVIEW


def test_alert_eligible_payload_cannot_be_silently_downgraded(
    consensus_data: dict[str, object],
) -> None:
    consensus_data["outcome"] = "review"
    consensus_data["alert_candidate"] = False

    with pytest.raises(ValidationError, match="alert-eligible"):
        ConsensusDecisionDocument.model_validate(consensus_data)


def test_reason_codes_are_unique_and_canonically_ordered(
    consensus_data: dict[str, object],
) -> None:
    duplicated = copy.deepcopy(consensus_data)
    duplicated["reason_codes"].append("alert_candidate")
    with pytest.raises(ValidationError, match="unique"):
        ConsensusDecisionDocument.model_validate(duplicated)

    reordered = copy.deepcopy(consensus_data)
    reordered["reason_codes"] = list(reversed(reordered["reason_codes"]))
    with pytest.raises(ValidationError, match="canonical"):
        ConsensusDecisionDocument.model_validate(reordered)


def test_risk_reference_rejects_naive_time_and_invalid_branch_sets(
    consensus_data: dict[str, object],
) -> None:
    naive = copy.deepcopy(consensus_data)
    naive["created_at"] = "2026-09-26T12:00:00"
    with pytest.raises(ValidationError, match="timezone-aware"):
        ConsensusDecisionDocument.model_validate(naive)

    overlap = copy.deepcopy(consensus_data)
    overlap["risk_assessment"]["missing_branches"] = ["speech"]
    overlap["risk_assessment"]["not_permitted_branches"] = ["speech"]
    with pytest.raises(ValidationError, match="both missing and not permitted"):
        ConsensusDecisionDocument.model_validate(overlap)


def test_contract_is_privacy_minimized_and_does_not_deliver_alerts(
    consensus_data: dict[str, object],
) -> None:
    serialized = ConsensusDecisionDocument.model_validate(consensus_data).model_dump_json()

    for forbidden in (
        "transcript_text",
        "matched_text",
        "audio_bytes",
        "speaker",
        "phone_number",
        "email_address",
        "delivery_status",
    ):
        assert forbidden not in serialized


def test_checked_in_schema_matches_generated_contract(project_root: Path) -> None:
    checked_in = json.loads(
        (project_root / "docs" / "schemas" / "v1" / "consensus-decision.schema.json")
        .read_text(encoding="utf-8")
    )

    assert checked_in == consensus_schema_documents()["consensus-decision.schema.json"]
    assert checked_in["$id"].endswith("/consensus-decision.schema.json")


def test_schema_export_writes_portable_json(tmp_path: Path) -> None:
    exported = write_consensus_schemas(tmp_path)
    destination = exported["consensus-decision.schema.json"]

    assert destination == tmp_path / "consensus-decision.schema.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        consensus_schema_documents()["consensus-decision.schema.json"]
    )


def test_public_records_preserve_datetime_and_enum_types(
    consensus_data: dict[str, object],
) -> None:
    document = ConsensusDecisionDocument.model_validate(consensus_data)

    assert isinstance(document.created_at, datetime)
    assert document.created_at.astimezone(UTC).utcoffset().total_seconds() == 0
    assert document.reason_codes[-1] is ConsensusReasonCode.ALERT_CANDIDATE


def _outcome_payload(
    source: dict[str, object], *, score: float, severity: str
) -> dict[str, object]:
    payload = copy.deepcopy(source)
    payload["risk_assessment"]["score"] = score
    payload["risk_assessment"]["severity"] = severity
    payload["branch_states"] = [
        {"kind": "acoustic", "input_status": "present", "agreement": "neutral"},
        {"kind": "speech", "input_status": "present", "agreement": "neutral"},
        {"kind": "language", "input_status": "present", "agreement": "neutral"},
    ]
    return payload


def _set_nested(payload: dict[str, object], change: tuple[object, ...]) -> None:
    target: object = payload
    for key in change[:-2]:
        target = target[key]  # type: ignore[index]
    target[change[-2]] = change[-1]  # type: ignore[index]
