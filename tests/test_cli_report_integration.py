"""B8.3 integration tests across CLI, report, audit, and process boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from audio_sentinel.alert_audit import (
    AlertReviewStatus,
    load_alert_audit,
)
from audio_sentinel.config import load_settings
from audio_sentinel.final_report import load_final_report
from audio_sentinel.risk_contracts import RiskInputStatus
from test_cli import evaluation_args, install_fake_models, project, runner


def run_evaluation(root: Path, monkeypatch, scope: str) -> dict[str, object]:
    install_fake_models(monkeypatch)
    result = runner.invoke(runner_app(), evaluation_args(root, scope))
    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    return json.loads(result.stdout)


def runner_app():
    # Import through the same public module the console entry point resolves.
    from audio_sentinel.cli import app

    return app


def processed_path(root: Path, relative: object) -> Path:
    assert isinstance(relative, str)
    return root / "data" / "processed" / relative


def test_acoustic_only_cli_round_trip_matches_verified_report_and_audit(
    tmp_path, monkeypatch
):
    root = project(tmp_path)
    output = run_evaluation(root, monkeypatch, "acoustic_only")
    settings = load_settings(root)

    report = load_final_report(settings.paths, output["report_path"])
    audit = load_alert_audit(settings.paths, output["audit_path"])

    assert output["report_id"] == report.report_id
    assert output["audit_id"] == audit.audit.audit_id
    assert output["outcome"] == report.summary.outcome.value == "review"
    assert report.source.processing_scope.value == "acoustic_only"
    assert [receipt.input_status for receipt in report.evidence[1:]] == [
        RiskInputStatus.NOT_PERMITTED,
        RiskInputStatus.NOT_PERMITTED,
    ]
    assert audit.alert is None
    assert output["alert_id"] is output["alert_path"] is None
    assert audit.audit.notification_delivery == "not_sent"
    assert audit.audit.alert_delivery_authorized is False


def test_speech_cli_round_trip_creates_one_pending_local_alert_without_sensitive_text(
    tmp_path, monkeypatch
):
    root = project(tmp_path)
    output = run_evaluation(root, monkeypatch, "acoustic_and_speech")
    settings = load_settings(root)

    report = load_final_report(settings.paths, output["report_path"])
    audit = load_alert_audit(settings.paths, output["audit_path"])
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            processed_path(root, output["report_path"]),
            processed_path(root, output["audit_path"]),
            processed_path(root, output["alert_path"]),
        )
    )

    assert report.summary.outcome.value == output["outcome"] == "alert"
    assert audit.alert is not None
    assert audit.alert.alert_id == output["alert_id"]
    assert audit.alert.review_status is AlertReviewStatus.PENDING
    assert audit.alert.local_only is True
    assert audit.alert.notification_delivery == "not_sent"
    assert audit.alert.alert_delivery_authorized is False
    assert "I will kill you" not in persisted
    assert '"text"' not in persisted.lower()
    assert str(root) not in persisted


def test_repeated_cli_run_preserves_first_bundle_and_creates_new_assessment(
    tmp_path, monkeypatch
):
    root = project(tmp_path)
    first = run_evaluation(root, monkeypatch, "acoustic_and_speech")
    first_report = processed_path(root, first["report_path"]).read_bytes()
    first_audit = processed_path(root, first["audit_path"]).read_bytes()

    second = run_evaluation(root, monkeypatch, "acoustic_and_speech")

    assert first["report_reused"] is False
    assert first["audit_reused"] is False
    assert second["report_reused"] is False
    assert second["audit_reused"] is False
    assert second["report_id"] != first["report_id"]
    assert second["audit_id"] != first["audit_id"]
    assert processed_path(root, first["report_path"]).read_bytes() == first_report
    assert processed_path(root, first["audit_path"]).read_bytes() == first_audit
    assert processed_path(root, second["report_path"]).is_file()
    assert processed_path(root, second["audit_path"]).is_file()
    assert len(list((root / "data" / "processed" / "final-reports").iterdir())) == 2
    assert len(list((root / "data" / "processed" / "alert-audit").iterdir())) == 2


def test_fresh_python_process_inspects_persisted_report_with_compact_json(
    tmp_path, monkeypatch, project_root
):
    root = project(tmp_path)
    output = run_evaluation(root, monkeypatch, "acoustic_and_speech")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "audio_sentinel.cli",
            "inspect-report",
            str(output["report_path"]),
            "--project-root",
            str(root),
            "--compact",
        ],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    inspected = json.loads(completed.stdout)
    assert inspected["report_id"] == output["report_id"]
    assert inspected["summary"]["outcome"] == output["outcome"]
    assert inspected["summary"]["notification_delivery"] == "not_sent"
    assert completed.stderr == ""
    assert "\n  " not in completed.stdout
    assert "I will kill you" not in completed.stdout
    assert str(root) not in completed.stdout


def test_inspection_rejects_tampered_report_with_safe_stderr(tmp_path, monkeypatch):
    root = project(tmp_path)
    output = run_evaluation(root, monkeypatch, "acoustic_and_speech")
    report_path = processed_path(root, output["report_path"])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["summary"]["risk_score"] = 0
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        runner_app(),
        ["inspect-report", output["report_path"], "--project-root", str(root)],
    )

    assert result.exit_code == 1
    error = json.loads(result.stderr)
    assert error["code"] == "invalid_document"
    assert "risk_score" not in result.stderr
    assert str(root) not in result.stderr
    assert result.stdout == ""


def test_inspection_rejects_unexpected_report_bundle_inventory(tmp_path, monkeypatch):
    root = project(tmp_path)
    output = run_evaluation(root, monkeypatch, "acoustic_only")
    report_path = processed_path(root, output["report_path"])
    (report_path.parent / "unexpected.txt").write_text("private value", encoding="utf-8")

    result = runner.invoke(
        runner_app(),
        ["inspect-report", output["report_path"], "--project-root", str(root)],
    )

    assert result.exit_code == 1
    error = json.loads(result.stderr)
    assert error["code"] == "output_conflict"
    assert "private value" not in result.stderr
    assert result.stdout == ""


def test_inspection_rejects_relative_traversal_before_reading_outside_file(tmp_path):
    root = project(tmp_path)
    outside = root / "data" / "outside.json"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text('{"secret":"private value"}', encoding="utf-8")

    result = runner.invoke(
        runner_app(),
        ["inspect-report", "../outside.json", "--project-root", str(root)],
    )

    assert result.exit_code == 1
    error = json.loads(result.stderr)
    assert error["code"] == "invalid_path"
    assert "private value" not in result.stderr
    assert result.stdout == ""


def test_cli_output_paths_are_portable_relative_and_identity_bound(tmp_path, monkeypatch):
    root = project(tmp_path)
    output = run_evaluation(root, monkeypatch, "acoustic_and_speech")

    assert output["report_path"] == (
        f"final-reports/{output['report_id']}/report.json"
    )
    assert output["audit_path"] == f"alert-audit/{output['audit_id']}/audit.json"
    assert output["alert_path"] == f"alert-audit/{output['audit_id']}/alert.json"
    for key in ("report_path", "audit_path", "alert_path"):
        value = output[key]
        assert isinstance(value, str)
        assert "\\" not in value
        assert not Path(value).is_absolute()
        assert ".." not in Path(value).parts
