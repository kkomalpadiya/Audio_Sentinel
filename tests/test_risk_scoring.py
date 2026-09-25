"""B6.1 tests for configurable deterministic risk scoring."""

from dataclasses import replace
from datetime import UTC, datetime
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from audio_sentinel.contracts import RiskLevel
from audio_sentinel.risk_contracts import RiskInputSet, RiskReasonCode
from audio_sentinel import risk_scoring as scoring


NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


@pytest.fixture
def risk_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (project_root / "docs" / "examples" / "risk-assessment.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def inputs(risk_data: dict[str, object]) -> RiskInputSet:
    return RiskInputSet.model_validate(risk_data["inputs"])


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


def loaded_from_data(data: dict[str, object]) -> scoring.LoadedRiskRuleSet:
    return scoring.load_risk_rule_set_bytes(
        (json.dumps(data, indent=2) + "\n").encode("utf-8")
    )


def test_builtin_rule_set_is_pinned_complete_and_portable(project_root: Path) -> None:
    loaded = scoring.load_builtin_risk_rule_set()
    artifact = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / scoring.BUILTIN_RISK_RULE_RESOURCE
    ).read_bytes()

    assert loaded.rule_set.rule_set_id == scoring.BUILTIN_RISK_RULE_SET_ID
    assert loaded.rule_set.rule_set_version == scoring.BUILTIN_RISK_RULE_SET_VERSION
    assert loaded.artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert loaded.artifact_sha256 == scoring.BUILTIN_RISK_RULE_SET_SHA256
    assert tuple(item.label.value for item in loaded.rule_set.acoustic.label_scores) == (
        "ambient",
        "no_speech",
        "speech_present",
        "non_threatening_speech",
        "siren",
        "smoke_alarm",
        "glass_break",
        "crowd_panic",
        "distress_speech",
        "threatening_speech",
        "weapon_reference",
        "gunshot",
        "explosion",
    )


def test_example_inputs_score_to_documented_critical_result(
    inputs: RiskInputSet,
) -> None:
    assessment = scoring.score_risk(inputs, now=NOW)

    assert assessment.score == 82
    assert assessment.severity is RiskLevel.CRITICAL
    assert assessment.reason_codes == (
        RiskReasonCode.ACOUSTIC_EVENT_CANDIDATE,
        RiskReasonCode.ACOUSTIC_HIGH_CONFIDENCE,
        RiskReasonCode.SPEECH_PRESENT,
        RiskReasonCode.TRANSCRIPT_REVIEW_REQUIRED,
        RiskReasonCode.LANGUAGE_DISTRESS,
    )
    assert assessment.human_review_required is True
    assert assessment.scoring_policy.rule_set.artifact_sha256 == (
        scoring.BUILTIN_RISK_RULE_SET_SHA256
    )


def test_assessment_id_is_deterministic_and_time_is_not_identity(
    inputs: RiskInputSet,
) -> None:
    first = scoring.score_risk(inputs, now=NOW)
    second = scoring.score_risk(
        inputs, now=datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
    )

    assert first.assessment_id == second.assessment_id
    assert first.created_at != second.created_at


def test_custom_rule_weights_change_score_and_identity(
    inputs: RiskInputSet,
    rule_data: dict[str, object],
) -> None:
    rule_data["rule_set_id"] = "custom-risk-rules"
    rule_data["rule_set_version"] = "1.0.1"
    rule_data["acoustic"]["label_scores"][-1]["points"] = 20
    custom = scoring.score_risk(
        inputs,
        rule_set=loaded_from_data(rule_data),
        now=NOW,
    )
    baseline = scoring.score_risk(inputs, now=NOW)

    assert custom.score == 62
    assert custom.severity is RiskLevel.HIGH
    assert custom.assessment_id != baseline.assessment_id
    assert custom.scoring_policy.rule_set.rule_set_id == "custom-risk-rules"


def test_high_confidence_threshold_is_inclusive(
    risk_data: dict[str, object],
) -> None:
    risk_data["inputs"]["acoustic"]["signals"][0]["peak_score"] = 0.85
    risk_data["inputs"]["acoustic"]["max_peak_score"] = 0.85
    assessment = scoring.score_risk(
        RiskInputSet.model_validate(risk_data["inputs"]), now=NOW
    )

    assert assessment.score == 82
    assert RiskReasonCode.ACOUSTIC_HIGH_CONFIDENCE in assessment.reason_codes


def test_total_and_branch_scores_are_capped(
    risk_data: dict[str, object],
) -> None:
    base = risk_data["inputs"]["language"]["signals"][0]
    signals = []
    for index in range(4):
        signal = copy.deepcopy(base)
        signal["finding_id"] = f"finding-{index}"
        signal["category"] = "threat"
        signals.append(signal)
    risk_data["inputs"]["language"]["signals"] = signals
    risk_data["inputs"]["language"]["finding_count"] = len(signals)

    assessment = scoring.score_risk(
        RiskInputSet.model_validate(risk_data["inputs"]), now=NOW
    )

    assert assessment.score == 100
    assert assessment.severity is RiskLevel.CRITICAL


def test_zero_weight_evidence_does_not_create_risk(
    risk_data: dict[str, object],
) -> None:
    acoustic = risk_data["inputs"]["acoustic"]
    acoustic["signals"][0]["label"] = "ambient"
    speech = risk_data["inputs"]["speech"]
    speech.update(
        segment_count=0,
        accepted_transcript_count=0,
        review_required_transcript_count=0,
        max_vad_score=None,
    )
    language = risk_data["inputs"]["language"]
    language.update(finding_count=0, signals=[])

    assessment = scoring.score_risk(
        RiskInputSet.model_validate(risk_data["inputs"]), now=NOW
    )

    assert assessment.score == 0
    assert assessment.severity is RiskLevel.NONE
    assert assessment.reason_codes == (RiskReasonCode.NO_RISK_EVIDENCE,)
    assert assessment.human_review_required is False


def test_missing_branches_require_review_without_adding_points(
    risk_data: dict[str, object],
) -> None:
    risk_data["inputs"]["acoustic"] = {
        "status": "missing",
        "evidence": None,
        "event_count": 0,
        "max_peak_score": None,
        "signals": [],
    }
    risk_data["inputs"]["speech"] = {
        "status": "missing",
        "evidence": None,
        "segment_count": 0,
        "accepted_transcript_count": 0,
        "review_required_transcript_count": 0,
        "max_vad_score": None,
    }
    risk_data["inputs"]["language"] = {
        "status": "missing",
        "evidence": None,
        "finding_count": 0,
        "signals": [],
    }

    assessment = scoring.score_risk(
        RiskInputSet.model_validate(risk_data["inputs"]), now=NOW
    )

    assert assessment.score == 0
    assert assessment.missing_data.missing_branches == (
        "acoustic",
        "speech",
        "language",
    )
    assert assessment.missing_data.review_required is True
    assert assessment.human_review_required is True
    assert RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED in assessment.reason_codes


def test_consent_limited_branches_are_not_treated_as_missing(
    risk_data: dict[str, object],
) -> None:
    risk_data["inputs"]["source"]["processing_scope"] = "acoustic_only"
    risk_data["inputs"]["acoustic"]["signals"][0]["label"] = "ambient"
    risk_data["inputs"]["speech"] = {
        "status": "not_permitted",
        "evidence": None,
        "segment_count": 0,
        "accepted_transcript_count": 0,
        "review_required_transcript_count": 0,
        "max_vad_score": None,
    }
    risk_data["inputs"]["language"] = {
        "status": "not_permitted",
        "evidence": None,
        "finding_count": 0,
        "signals": [],
    }

    assessment = scoring.score_risk(
        RiskInputSet.model_validate(risk_data["inputs"]), now=NOW
    )

    assert assessment.score == 0
    assert assessment.missing_data.missing_branches == ()
    assert assessment.missing_data.not_permitted_branches == ("speech", "language")
    assert assessment.missing_data.review_required is False
    assert RiskReasonCode.SPEECH_NOT_PERMITTED in assessment.reason_codes
    assert RiskReasonCode.LANGUAGE_NOT_APPLICABLE in assessment.reason_codes


def test_transcript_uncertainty_can_require_review_below_score_threshold(
    risk_data: dict[str, object],
) -> None:
    risk_data["inputs"]["acoustic"]["signals"][0]["label"] = "ambient"
    risk_data["inputs"]["language"].update(finding_count=0, signals=[])
    assessment = scoring.score_risk(
        RiskInputSet.model_validate(risk_data["inputs"]), now=NOW
    )

    assert assessment.score == 7
    assert assessment.severity is RiskLevel.LOW
    assert assessment.human_review_required is True


def test_ambiguous_language_can_require_review_below_score_threshold(
    risk_data: dict[str, object],
) -> None:
    risk_data["inputs"]["acoustic"]["signals"][0]["label"] = "ambient"
    risk_data["inputs"]["speech"]["review_required_transcript_count"] = 0
    risk_data["inputs"]["language"]["signals"][0]["category"] = "ambiguous"
    assessment = scoring.score_risk(
        RiskInputSet.model_validate(risk_data["inputs"]), now=NOW
    )

    assert assessment.score == 7
    assert assessment.human_review_required is True
    assert RiskReasonCode.LANGUAGE_AMBIGUOUS in assessment.reason_codes


@pytest.mark.parametrize(
    "section,key",
    [("acoustic", "label_scores"), ("language", "category_scores")],
)
def test_rule_inventory_requires_complete_canonical_mappings(
    rule_data: dict[str, object], section: str, key: str
) -> None:
    rule_data[section][key].pop()

    with pytest.raises(ValidationError, match="cover every"):
        scoring.RiskRuleSet.model_validate(rule_data)


@pytest.mark.parametrize(
    "section,index",
    [("acoustic", 0), ("language", 0), ("language", -1)],
)
def test_non_risk_categories_cannot_be_configured_with_points(
    rule_data: dict[str, object], section: str, index: int
) -> None:
    key = "label_scores" if section == "acoustic" else "category_scores"
    rule_data[section][key][index]["points"] = 1

    with pytest.raises(ValidationError, match="must have zero points"):
        scoring.RiskRuleSet.model_validate(rule_data)


@pytest.mark.parametrize(
    "payload,message",
    [(b"", "must not be empty"), (b"not-json", "contract validation"), (b"{}", "contract validation")],
)
def test_loader_rejects_empty_or_invalid_payload(payload: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        scoring.load_risk_rule_set_bytes(payload)


def test_loader_rejects_changed_bytes_and_oversized_payload(project_root: Path) -> None:
    artifact = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / scoring.BUILTIN_RISK_RULE_RESOURCE
    ).read_bytes()
    with pytest.raises(ValueError, match="checksum mismatch"):
        scoring.load_risk_rule_set_bytes(
            artifact + b"\n", expected_sha256=scoring.BUILTIN_RISK_RULE_SET_SHA256
        )
    with pytest.raises(ValueError, match="byte limit"):
        scoring.load_risk_rule_set_bytes(b"x" * (scoring.MAX_RISK_RULE_SET_BYTES + 1))


def test_scorer_rejects_invalid_input_rule_wrapper_and_time(
    inputs: RiskInputSet,
) -> None:
    with pytest.raises(scoring.RiskScoringError) as invalid_inputs:
        scoring.score_risk({"inputs": "wrong"}, now=NOW)
    assert invalid_inputs.value.code == "invalid_inputs"

    loaded = scoring.load_builtin_risk_rule_set()
    changed = replace(loaded, artifact_sha256="not-a-digest")
    with pytest.raises(scoring.RiskScoringError) as invalid_rules:
        scoring.score_risk(inputs, rule_set=changed, now=NOW)
    assert invalid_rules.value.code == "invalid_rule_set"

    with pytest.raises(scoring.RiskScoringError) as invalid_time:
        scoring.score_risk(inputs, now=datetime(2026, 9, 25, 12, 0))
    assert invalid_time.value.code == "invalid_time"


def test_output_contains_no_raw_audio_transcript_or_alert_data(
    inputs: RiskInputSet,
) -> None:
    serialized = scoring.score_risk(inputs, now=NOW).model_dump_json()

    assert "audio_bytes" not in serialized
    assert "transcript_text" not in serialized
    assert "speaker" not in serialized
    assert "alert_id" not in serialized
    assert "consensus" not in serialized


def test_engine_loads_without_network_or_ml_runtime() -> None:
    code = """
import sys
class NoModelImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'onnxruntime', 'faster_whisper'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
def reject_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError('Unexpected network access')
sys.addaudithook(reject_network)
from audio_sentinel.risk_scoring import load_builtin_risk_rule_set
assert load_builtin_risk_rule_set().rule_set.total_max_score == 100
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        capture_output=True,
        timeout=30,
    )


def test_checked_in_rule_schema_matches_generated_contract(project_root: Path) -> None:
    checked_in = json.loads(
        (project_root / "docs" / "schemas" / "v1" / "risk-rule-set.schema.json")
        .read_text(encoding="utf-8")
    )

    assert checked_in == scoring.risk_rule_schema_documents()["risk-rule-set.schema.json"]
    assert checked_in["$id"].endswith("/risk-rule-set.schema.json")


def test_schema_export_writes_portable_json(tmp_path: Path) -> None:
    exported = scoring.write_risk_rule_schemas(tmp_path)
    destination = exported["risk-rule-set.schema.json"]

    assert json.loads(destination.read_text(encoding="utf-8")) == (
        scoring.risk_rule_schema_documents()["risk-rule-set.schema.json"]
    )
