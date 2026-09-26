"""A6.4 risk-policy acceptance validation tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from audio_sentinel.contracts import RiskLevel
from audio_sentinel.risk_contracts import RiskInputSet
from audio_sentinel.risk_scoring import load_builtin_risk_rule_set
from audio_sentinel import risk_validation as validation


NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


@pytest.fixture
def loaded_suite() -> validation.LoadedRiskValidationSuite:
    return validation.load_builtin_risk_validation_suite()


def test_builtin_suite_is_pinned_complete_and_rule_linked(
    project_root: Path,
    loaded_suite: validation.LoadedRiskValidationSuite,
) -> None:
    artifact = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / validation.BUILTIN_RISK_VALIDATION_RESOURCE
    ).read_bytes()

    assert loaded_suite.artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert loaded_suite.artifact_sha256 == validation.BUILTIN_RISK_VALIDATION_SHA256
    assert loaded_suite.suite.rule_set_sha256 == (
        load_builtin_risk_rule_set().artifact_sha256
    )
    assert len(loaded_suite.suite.cases) == 13
    assert {
        marker for case in loaded_suite.suite.cases for marker in case.coverage
    } == set(validation.RiskValidationCoverage)


def test_every_compact_case_expands_through_the_a6_1_contract(
    loaded_suite: validation.LoadedRiskValidationSuite,
) -> None:
    for case in loaded_suite.suite.cases:
        inputs = validation.build_risk_validation_inputs(case)

        assert isinstance(inputs, RiskInputSet)
        assert inputs.source.clip_id == f"{case.case_id}-clip"


def test_builtin_policy_passes_every_acceptance_case() -> None:
    report = validation.validate_builtin_risk_policy(now=NOW)

    assert report.decision_status == "validated_for_deterministic_v1_scenarios"
    assert (report.scenario_count, report.passed_count, report.failed_count) == (
        13,
        13,
        0,
    )
    assert all(case.passed for case in report.cases)
    assert {case.observed.severity for case in report.cases} == set(RiskLevel)
    assert report.coverage == tuple(validation.RiskValidationCoverage)


def test_report_records_the_exact_policy_thresholds() -> None:
    thresholds = validation.validate_builtin_risk_policy(now=NOW).thresholds

    assert thresholds.acoustic_high_confidence_threshold == 0.85
    assert thresholds.human_review_minimum_score == 25
    assert thresholds.acoustic_max_points == 60
    assert thresholds.speech_max_points == 12
    assert thresholds.language_max_points == 50
    assert thresholds.total_max_score == 100
    assert tuple(band.min_score for band in thresholds.severity_bands) == (
        0,
        1,
        25,
        50,
        75,
    )


def test_validation_identity_excludes_run_time() -> None:
    first = validation.validate_builtin_risk_policy(now=NOW)
    second = validation.validate_builtin_risk_policy(
        now=datetime(2026, 9, 27, 12, tzinfo=UTC)
    )

    assert first.validation_id == second.validation_id
    assert first.created_at != second.created_at


def test_changed_expectation_produces_an_explicit_failed_report(
    loaded_suite: validation.LoadedRiskValidationSuite,
) -> None:
    payload = loaded_suite.suite.model_dump(mode="python")
    payload["cases"][0]["expected"]["human_review_required"] = True
    changed = replace(
        loaded_suite,
        suite=validation.RiskValidationSuite.model_validate(payload),
    )

    report = validation.validate_risk_policy(
        changed,
        load_builtin_risk_rule_set(),
        now=NOW,
    )

    assert report.decision_status == "validation_failed"
    assert (report.passed_count, report.failed_count) == (12, 1)
    assert report.cases[0].passed is False


def test_suite_rejects_missing_required_coverage(
    loaded_suite: validation.LoadedRiskValidationSuite,
) -> None:
    payload = loaded_suite.suite.model_dump(mode="python")
    payload["cases"] = payload["cases"][:-1]

    with pytest.raises(ValidationError, match="every required behavior"):
        validation.RiskValidationSuite.model_validate(payload)


def test_suite_rejects_duplicate_case_ids(
    loaded_suite: validation.LoadedRiskValidationSuite,
) -> None:
    payload = loaded_suite.suite.model_dump(mode="python")
    payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]

    with pytest.raises(ValidationError, match="case IDs must be unique"):
        validation.RiskValidationSuite.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"processing_scope": "none"},
        {"processing_scope": "acoustic_only"},
        {"language": {"categories": ["distress"]}},
        {
            "speech": {"status": "missing", "segment_count": 1},
        },
    ],
)
def test_case_contract_rejects_semantically_invalid_inputs(
    loaded_suite: validation.LoadedRiskValidationSuite,
    change: dict[str, object],
) -> None:
    payload = loaded_suite.suite.cases[0].model_dump(mode="python")
    payload.update(change)

    with pytest.raises(ValidationError):
        validation.RiskValidationCase.model_validate(payload)


def test_suite_checksum_and_json_tampering_are_rejected(
    project_root: Path,
) -> None:
    artifact = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / validation.BUILTIN_RISK_VALIDATION_RESOURCE
    ).read_bytes()

    with pytest.raises(ValueError, match="checksum mismatch"):
        validation.load_risk_validation_suite_bytes(
            artifact + b" ",
            expected_sha256=validation.BUILTIN_RISK_VALIDATION_SHA256,
        )
    with pytest.raises(ValueError, match="contract validation"):
        validation.load_risk_validation_suite_bytes(b"{}")


def test_suite_rejects_a_different_rule_artifact(
    loaded_suite: validation.LoadedRiskValidationSuite,
) -> None:
    changed = replace(
        loaded_suite,
        suite=loaded_suite.suite.model_copy(
            update={"rule_set_sha256": "0" * 64}
        ),
    )

    with pytest.raises(validation.RiskValidationError) as error:
        validation.validate_risk_policy(
            changed,
            load_builtin_risk_rule_set(),
            now=NOW,
        )

    assert error.value.code == "rule_set_mismatch"


def test_naive_validation_time_is_rejected(
    loaded_suite: validation.LoadedRiskValidationSuite,
) -> None:
    with pytest.raises(validation.RiskValidationError) as error:
        validation.validate_risk_policy(
            loaded_suite,
            load_builtin_risk_rule_set(),
            now=datetime(2026, 9, 26, 12),
        )

    assert error.value.code == "invalid_time"


def test_report_is_immutable() -> None:
    report = validation.validate_builtin_risk_policy(now=NOW)

    with pytest.raises(ValidationError):
        report.passed_count = 0


def test_save_report_is_valid_json_and_never_overwrites(tmp_path: Path) -> None:
    report = validation.validate_builtin_risk_policy(now=NOW)
    output = tmp_path / "validation" / "report.json"

    validation.save_risk_validation_report(report, output)

    saved = validation.RiskValidationReport.model_validate_json(
        output.read_text(encoding="utf-8")
    )
    assert saved == report
    original = output.read_bytes()
    with pytest.raises(validation.RiskValidationError) as error:
        validation.save_risk_validation_report(report, output)
    assert error.value.code == "output_exists"
    assert output.read_bytes() == original


def test_cli_writes_a_passing_report(project_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "validate_risk_policy.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    summary = json.loads(completed.stdout)
    assert summary["decision_status"] == (
        "validated_for_deterministic_v1_scenarios"
    )
    assert summary["passed_count"] == 13
    assert validation.RiskValidationReport.model_validate_json(
        output.read_text(encoding="utf-8")
    ).failed_count == 0


def test_validation_loads_without_network_or_ml_runtime() -> None:
    code = """
import sys
class NoModelImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'onnxruntime'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
def reject_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError('Unexpected network access')
sys.addaudithook(reject_network)
from audio_sentinel.risk_validation import validate_builtin_risk_policy
assert validate_builtin_risk_policy().failed_count == 0
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        capture_output=True,
        timeout=30,
    )
