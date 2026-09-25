"""B6.2 normal and edge-case scenarios for deterministic risk scoring."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from audio_sentinel.contracts import EventLabel, RiskLevel
from audio_sentinel.language_contracts import LanguageCategory
from audio_sentinel.risk_contracts import (
    RiskEvidenceKind,
    RiskInputSet,
    RiskInputStatus,
    RiskReasonCode,
)
from audio_sentinel import risk_scoring as scoring


NOW = datetime(2026, 9, 25, 12, tzinfo=UTC)


@pytest.fixture
def base_inputs(project_root: Path) -> dict[str, object]:
    document = json.loads(
        (project_root / "docs" / "examples" / "risk-assessment.json").read_text(
            encoding="utf-8"
        )
    )
    return document["inputs"]


@pytest.fixture
def rule_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (
            project_root
            / "src"
            / "audio_sentinel"
            / "resources"
            / scoring.BUILTIN_RISK_RULE_RESOURCE
        ).read_text(encoding="utf-8")
    )


def neutral_inputs(base: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(base)
    payload["acoustic"].update(
        event_count=0,
        max_peak_score=None,
        signals=[],
    )
    payload["speech"].update(
        segment_count=0,
        accepted_transcript_count=0,
        review_required_transcript_count=0,
        max_vad_score=None,
    )
    payload["language"].update(finding_count=0, signals=[])
    return payload


def acoustic_signal(
    label: EventLabel | str,
    peak_score: float,
    *,
    index: int = 0,
) -> dict[str, object]:
    start_sample = index * 2_000
    end_sample = start_sample + 1_000
    return {
        "label": label.value if isinstance(label, EventLabel) else label,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_seconds": start_sample / 16_000,
        "end_seconds": end_sample / 16_000,
        "peak_score": peak_score,
        "source_event_index": index,
    }


def set_acoustic(
    payload: dict[str, object],
    signals: list[dict[str, object]],
) -> None:
    payload["acoustic"].update(
        event_count=len(signals),
        max_peak_score=max(
            (signal["peak_score"] for signal in signals), default=None
        ),
        signals=signals,
    )


def set_speech(
    payload: dict[str, object],
    *,
    segments: int,
    review_required: int = 0,
) -> None:
    payload["speech"].update(
        segment_count=segments,
        accepted_transcript_count=0,
        review_required_transcript_count=review_required,
        max_vad_score=0.9 if segments else None,
    )


_LANGUAGE_REASONS = {
    LanguageCategory.NO_CONCERNING_MATCH: ["no_rule_match"],
    LanguageCategory.DISTRESS: ["keyword_match"],
    LanguageCategory.THREAT: ["keyword_match"],
    LanguageCategory.WEAPON_REFERENCE: ["keyword_match"],
    LanguageCategory.AMBIGUOUS: ["keyword_match", "insufficient_context"],
    LanguageCategory.CONTEXT_SUPPRESSED: ["keyword_match", "explicit_negation"],
}


def set_language(
    payload: dict[str, object],
    categories: list[LanguageCategory],
) -> None:
    signals = []
    for index, category in enumerate(categories):
        start_sample = 10_000 + index * 1_000
        signals.append(
            {
                "finding_id": f"scenario-finding-{index:04d}",
                "segment_id": f"scenario-segment-{index:04d}",
                "category": category.value,
                "reason_codes": _LANGUAGE_REASONS[category],
                "start_sample": start_sample,
                "end_sample": start_sample + 500,
                "rule_match_count": (
                    0 if category is LanguageCategory.NO_CONCERNING_MATCH else 1
                ),
            }
        )
    payload["language"].update(finding_count=len(signals), signals=signals)


def score(
    payload: dict[str, object],
    *,
    rule_set: scoring.LoadedRiskRuleSet | None = None,
):
    return scoring.score_risk(
        RiskInputSet.model_validate(payload),
        rule_set=rule_set,
        now=NOW,
    )


def loaded_rules(data: dict[str, object]) -> scoring.LoadedRiskRuleSet:
    return scoring.load_risk_rule_set_bytes(
        (json.dumps(data, indent=2) + "\n").encode("utf-8")
    )


def set_rule_value(
    records: list[dict[str, object]],
    key: str,
    value: str,
    points: float,
) -> None:
    record = next(item for item in records if item[key] == value)
    record["points"] = points


@pytest.mark.parametrize(
    ("scenario", "expected_score", "expected_severity", "review_required"),
    [
        ("none", 0, RiskLevel.NONE, False),
        ("low", 10, RiskLevel.LOW, False),
        ("medium", 40, RiskLevel.MEDIUM, True),
        ("high", 55, RiskLevel.HIGH, True),
        ("critical", 77, RiskLevel.CRITICAL, True),
    ],
)
def test_normal_scenarios_cover_every_severity_level(
    base_inputs: dict[str, object],
    scenario: str,
    expected_score: int,
    expected_severity: RiskLevel,
    review_required: bool,
) -> None:
    payload = neutral_inputs(base_inputs)
    if scenario == "low":
        set_acoustic(payload, [acoustic_signal(EventLabel.SIREN, 0.7)])
    elif scenario == "medium":
        set_language(payload, [LanguageCategory.THREAT])
    elif scenario == "high":
        set_acoustic(payload, [acoustic_signal(EventLabel.GUNSHOT, 0.85)])
    elif scenario == "critical":
        set_acoustic(payload, [acoustic_signal(EventLabel.EXPLOSION, 0.9)])
        set_speech(payload, segments=1)
        set_language(payload, [LanguageCategory.DISTRESS])

    assessment = score(payload)

    assert assessment.score == expected_score
    assert assessment.severity is expected_severity
    assert assessment.human_review_required is review_required


@pytest.mark.parametrize(
    ("label", "expected_score"),
    [
        (EventLabel.AMBIENT, 0),
        (EventLabel.NO_SPEECH, 0),
        (EventLabel.SPEECH_PRESENT, 0),
        (EventLabel.NON_THREATENING_SPEECH, 0),
        (EventLabel.SIREN, 10),
        (EventLabel.SMOKE_ALARM, 15),
        (EventLabel.GLASS_BREAK, 25),
        (EventLabel.CROWD_PANIC, 30),
        (EventLabel.DISTRESS_SPEECH, 30),
        (EventLabel.THREATENING_SPEECH, 40),
        (EventLabel.WEAPON_REFERENCE, 40),
        (EventLabel.GUNSHOT, 45),
        (EventLabel.EXPLOSION, 40),
    ],
)
def test_every_builtin_acoustic_label_has_the_documented_weight(
    base_inputs: dict[str, object],
    label: EventLabel,
    expected_score: int,
) -> None:
    payload = neutral_inputs(base_inputs)
    set_acoustic(payload, [acoustic_signal(label, 0.84)])

    assessment = score(payload)

    assert assessment.score == expected_score
    if expected_score:
        assert assessment.reason_codes == (
            RiskReasonCode.ACOUSTIC_EVENT_CANDIDATE,
        )
    else:
        assert assessment.reason_codes == (RiskReasonCode.NO_RISK_EVIDENCE,)


@pytest.mark.parametrize(
    ("peak_score", "expected_score", "has_bonus"),
    [
        (0.849999, 45, False),
        (0.85, 55, True),
        (1.0, 55, True),
    ],
)
def test_acoustic_confidence_boundary_is_exact(
    base_inputs: dict[str, object],
    peak_score: float,
    expected_score: int,
    has_bonus: bool,
) -> None:
    payload = neutral_inputs(base_inputs)
    set_acoustic(payload, [acoustic_signal(EventLabel.GUNSHOT, peak_score)])

    assessment = score(payload)

    assert assessment.score == expected_score
    assert (
        RiskReasonCode.ACOUSTIC_HIGH_CONFIDENCE in assessment.reason_codes
    ) is has_bonus


def test_high_confidence_safe_event_does_not_bonus_a_lower_confidence_risk_event(
    base_inputs: dict[str, object],
) -> None:
    payload = neutral_inputs(base_inputs)
    set_acoustic(
        payload,
        [
            acoustic_signal(EventLabel.AMBIENT, 1.0),
            acoustic_signal(EventLabel.SIREN, 0.84, index=1),
        ],
    )

    assessment = score(payload)

    assert assessment.score == 10
    assert RiskReasonCode.ACOUSTIC_HIGH_CONFIDENCE not in assessment.reason_codes


def test_acoustic_events_use_maximum_weight_instead_of_sum(
    base_inputs: dict[str, object],
) -> None:
    payload = neutral_inputs(base_inputs)
    set_acoustic(
        payload,
        [
            acoustic_signal(EventLabel.GLASS_BREAK, 0.7),
            acoustic_signal(EventLabel.CROWD_PANIC, 0.7, index=1),
        ],
    )

    assert score(payload).score == 30


def test_acoustic_points_stop_at_the_branch_cap(
    base_inputs: dict[str, object],
    rule_data: dict[str, object],
) -> None:
    payload = neutral_inputs(base_inputs)
    set_acoustic(payload, [acoustic_signal(EventLabel.SIREN, 1.0)])
    set_rule_value(
        rule_data["acoustic"]["label_scores"], "label", "siren", 100
    )
    rule_data["acoustic"]["high_confidence_bonus"] = 100

    assert score(payload, rule_set=loaded_rules(rule_data)).score == 60


@pytest.mark.parametrize(
    ("category", "expected_score", "expected_reason", "review_required"),
    [
        (LanguageCategory.NO_CONCERNING_MATCH, 0, None, False),
        (LanguageCategory.DISTRESS, 25, RiskReasonCode.LANGUAGE_DISTRESS, True),
        (LanguageCategory.THREAT, 40, RiskReasonCode.LANGUAGE_THREAT, True),
        (
            LanguageCategory.WEAPON_REFERENCE,
            35,
            RiskReasonCode.LANGUAGE_WEAPON_REFERENCE,
            True,
        ),
        (LanguageCategory.AMBIGUOUS, 5, RiskReasonCode.LANGUAGE_AMBIGUOUS, True),
        (
            LanguageCategory.CONTEXT_SUPPRESSED,
            0,
            RiskReasonCode.LANGUAGE_CONTEXT_SUPPRESSED,
            False,
        ),
    ],
)
def test_every_language_category_has_the_documented_behavior(
    base_inputs: dict[str, object],
    category: LanguageCategory,
    expected_score: int,
    expected_reason: RiskReasonCode | None,
    review_required: bool,
) -> None:
    payload = neutral_inputs(base_inputs)
    set_language(payload, [category])

    assessment = score(payload)

    assert assessment.score == expected_score
    assert assessment.human_review_required is review_required
    if expected_reason is not None:
        assert expected_reason in assessment.reason_codes
    if expected_score == 0:
        assert RiskReasonCode.NO_RISK_EVIDENCE in assessment.reason_codes


def test_repeated_language_findings_stop_at_the_branch_cap(
    base_inputs: dict[str, object],
) -> None:
    payload = neutral_inputs(base_inputs)
    set_language(payload, [LanguageCategory.THREAT] * 5)

    assessment = score(payload)

    assert assessment.score == 50
    assert assessment.reason_codes == (RiskReasonCode.LANGUAGE_THREAT,)


def test_combined_branch_scores_stop_at_the_total_cap(
    base_inputs: dict[str, object],
) -> None:
    payload = neutral_inputs(base_inputs)
    set_acoustic(payload, [acoustic_signal(EventLabel.GUNSHOT, 0.9)])
    set_speech(payload, segments=2, review_required=2)
    set_language(payload, [LanguageCategory.THREAT, LanguageCategory.THREAT])

    assessment = score(payload)

    assert assessment.score == 100
    assert assessment.severity is RiskLevel.CRITICAL


@pytest.mark.parametrize(
    ("segments", "review_count", "expected_score", "review_required"),
    [
        (0, 0, 0, False),
        (1, 0, 2, False),
        (1, 1, 7, True),
        (2, 2, 12, True),
        (3, 3, 12, True),
    ],
)
def test_speech_presence_review_points_and_cap(
    base_inputs: dict[str, object],
    segments: int,
    review_count: int,
    expected_score: int,
    review_required: bool,
) -> None:
    payload = neutral_inputs(base_inputs)
    set_speech(payload, segments=segments, review_required=review_count)

    assessment = score(payload)

    assert assessment.score == expected_score
    assert assessment.human_review_required is review_required


def test_extremely_large_review_count_caps_without_numeric_overflow(
    base_inputs: dict[str, object],
) -> None:
    payload = neutral_inputs(base_inputs)
    count = 10**400
    set_speech(payload, segments=count, review_required=count)

    assessment = score(payload)

    assert assessment.score == 12
    assert assessment.severity is RiskLevel.LOW
    assert assessment.human_review_required is True


def test_subnormal_review_weight_caps_without_threshold_overflow(
    base_inputs: dict[str, object],
    rule_data: dict[str, object],
) -> None:
    payload = neutral_inputs(base_inputs)
    count = 10**400
    set_speech(payload, segments=count, review_required=count)
    rule_data["speech"]["review_required_points_per_transcript"] = 5e-324

    assessment = score(payload, rule_set=loaded_rules(rule_data))

    assert assessment.score == 12
    assert assessment.human_review_required is True


@pytest.mark.parametrize(
    ("score_value", "severity", "review_required"),
    [
        (1, RiskLevel.LOW, False),
        (24, RiskLevel.LOW, False),
        (25, RiskLevel.MEDIUM, True),
        (49, RiskLevel.MEDIUM, True),
        (50, RiskLevel.HIGH, True),
        (74, RiskLevel.HIGH, True),
        (75, RiskLevel.CRITICAL, True),
        (100, RiskLevel.CRITICAL, True),
    ],
)
def test_score_and_review_boundaries_through_the_configurable_engine(
    base_inputs: dict[str, object],
    rule_data: dict[str, object],
    score_value: int,
    severity: RiskLevel,
    review_required: bool,
) -> None:
    payload = neutral_inputs(base_inputs)
    set_acoustic(payload, [acoustic_signal(EventLabel.SIREN, 0.5)])
    set_rule_value(
        rule_data["acoustic"]["label_scores"],
        "label",
        "siren",
        score_value,
    )
    rule_data["acoustic"].update(high_confidence_bonus=0, max_points=100)

    assessment = score(payload, rule_set=loaded_rules(rule_data))

    assert assessment.score == score_value
    assert assessment.severity is severity
    assert assessment.human_review_required is review_required


@pytest.mark.parametrize(
    ("branch", "kind", "missing_reason"),
    [
        (
            "acoustic",
            RiskEvidenceKind.ACOUSTIC,
            RiskReasonCode.MISSING_ACOUSTIC_EVIDENCE,
        ),
        (
            "speech",
            RiskEvidenceKind.SPEECH,
            RiskReasonCode.MISSING_SPEECH_EVIDENCE,
        ),
        (
            "language",
            RiskEvidenceKind.LANGUAGE,
            RiskReasonCode.MISSING_LANGUAGE_EVIDENCE,
        ),
    ],
)
def test_each_missing_branch_is_named_and_requires_review(
    base_inputs: dict[str, object],
    branch: str,
    kind: RiskEvidenceKind,
    missing_reason: RiskReasonCode,
) -> None:
    payload = neutral_inputs(base_inputs)
    branch_payload = {
        "status": RiskInputStatus.MISSING.value,
        "evidence": None,
    }
    if branch == "acoustic":
        branch_payload.update(event_count=0, max_peak_score=None, signals=[])
    elif branch == "speech":
        branch_payload.update(
            segment_count=0,
            accepted_transcript_count=0,
            review_required_transcript_count=0,
            max_vad_score=None,
        )
        payload["language"] = {
            "status": RiskInputStatus.NOT_APPLICABLE.value,
            "evidence": None,
            "finding_count": 0,
            "signals": [],
        }
    else:
        branch_payload.update(finding_count=0, signals=[])
    payload[branch] = branch_payload

    assessment = score(payload)

    assert assessment.missing_data.missing_branches == (kind,)
    assert assessment.missing_data.review_required is True
    assert assessment.human_review_required is True
    assert missing_reason in assessment.reason_codes
    assert RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED in assessment.reason_codes


@pytest.mark.parametrize(
    "status",
    [RiskInputStatus.NOT_APPLICABLE, RiskInputStatus.NO_ACCEPTED_TEXT],
)
def test_nonmissing_language_absence_does_not_force_review(
    base_inputs: dict[str, object],
    status: RiskInputStatus,
) -> None:
    payload = neutral_inputs(base_inputs)
    payload["language"] = {
        "status": status.value,
        "evidence": None,
        "finding_count": 0,
        "signals": [],
    }

    assessment = score(payload)

    assert assessment.score == 0
    assert assessment.missing_data.missing_branches == ()
    assert assessment.missing_data.review_required is False
    assert assessment.human_review_required is False
    assert assessment.reason_codes == (
        RiskReasonCode.NO_RISK_EVIDENCE,
        RiskReasonCode.LANGUAGE_NOT_APPLICABLE,
    )


def test_combined_reasons_remain_complete_unique_and_canonical(
    base_inputs: dict[str, object],
) -> None:
    payload = neutral_inputs(base_inputs)
    set_acoustic(payload, [acoustic_signal(EventLabel.GUNSHOT, 0.9)])
    set_speech(payload, segments=1, review_required=1)
    payload["language"] = {
        "status": RiskInputStatus.MISSING.value,
        "evidence": None,
        "finding_count": 0,
        "signals": [],
    }

    assessment = score(payload)

    assert assessment.score == 62
    assert assessment.reason_codes == (
        RiskReasonCode.ACOUSTIC_EVENT_CANDIDATE,
        RiskReasonCode.ACOUSTIC_HIGH_CONFIDENCE,
        RiskReasonCode.SPEECH_PRESENT,
        RiskReasonCode.TRANSCRIPT_REVIEW_REQUIRED,
        RiskReasonCode.MISSING_LANGUAGE_EVIDENCE,
        RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED,
    )


def test_fractional_configuration_is_rounded_once_to_six_places(
    base_inputs: dict[str, object],
    rule_data: dict[str, object],
) -> None:
    payload = neutral_inputs(base_inputs)
    set_acoustic(payload, [acoustic_signal(EventLabel.SIREN, 0.5)])
    set_rule_value(
        rule_data["acoustic"]["label_scores"],
        "label",
        "siren",
        0.123456789,
    )
    rule_data["acoustic"]["high_confidence_bonus"] = 0

    assessment = score(payload, rule_set=loaded_rules(rule_data))

    assert assessment.score == 0.123457
    assert assessment.severity is RiskLevel.LOW
