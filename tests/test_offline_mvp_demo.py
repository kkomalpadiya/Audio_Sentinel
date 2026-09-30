"""A9.3 offline MVP demonstration-script tests without large model runtimes."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _load_demo_script(project_root: Path) -> ModuleType:
    path = project_root / "scripts" / "run_offline_mvp_demo.py"
    spec = importlib.util.spec_from_file_location("offline_mvp_demo_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arguments(**overrides: object) -> argparse.Namespace:
    now = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
    values: dict[str, object] = {
        "audio": "examples/authorized.wav",
        "clip_id": "demo-clip-001",
        "consent_id": "demo-consent-001",
        "scope": "acoustic_only",
        "granted_at": (now - timedelta(minutes=5)).isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "source_dataset": "authorized-local-demo",
        "acoustic_threshold": 0.5,
        "audio_config": None,
        "retention_config": "configs/retention.example.json",
        "confirm_consent_reviewed": True,
        "confirm_authorized_device": True,
        "preflight_only": False,
        "compact": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _temporary_demo_root(tmp_path: Path) -> Path:
    root = tmp_path / "Project_1"
    (root / "data" / "raw" / "examples").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (root / "data" / "raw" / "examples" / "authorized.wav").write_bytes(b"RIFF")
    (root / "configs" / "retention.example.json").write_text("{}\n", encoding="utf-8")
    return root


def test_validate_request_requires_confirmations_and_safe_local_path(
    project_root: Path,
    tmp_path: Path,
) -> None:
    demo = _load_demo_script(project_root)
    root = _temporary_demo_root(tmp_path)
    now = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)

    with pytest.raises(demo.DemoError, match="external consent") as consent_error:
        demo.validate_request(
            _arguments(confirm_consent_reviewed=False), project_root=root, now=now
        )
    assert consent_error.value.code == "consent_confirmation_required"

    with pytest.raises(demo.DemoError, match="approved project directory") as path_error:
        demo.validate_request(
            _arguments(audio="../private.wav"), project_root=root, now=now
        )
    assert path_error.value.code == "invalid_audio_path"


def test_validate_request_preserves_authorized_scope_and_times(
    project_root: Path,
    tmp_path: Path,
) -> None:
    demo = _load_demo_script(project_root)
    root = _temporary_demo_root(tmp_path)
    now = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
    request = demo.validate_request(
        _arguments(scope="acoustic_and_speech"), project_root=root, now=now
    )

    assert request.scope == "acoustic_and_speech"
    assert request.audio_path == "examples/authorized.wav"
    assert request.granted_at == "2026-10-01T09:55:00+00:00"
    assert request.expires_at == "2026-10-01T11:00:00+00:00"
    assert request.retention_config == "configs/retention.example.json"


def test_evaluate_command_has_explicit_authorization_and_no_delivery_options(
    project_root: Path,
    tmp_path: Path,
) -> None:
    demo = _load_demo_script(project_root)
    root = _temporary_demo_root(tmp_path)
    request = demo.validate_request(
        _arguments(),
        project_root=root,
        now=datetime(2026, 10, 1, 10, 0, tzinfo=UTC),
    )
    command = demo.build_evaluate_command(
        request, python_executable="python-demo", project_root=root
    )

    assert command[:4] == ["python-demo", "-m", "audio_sentinel.cli", "evaluate"]
    assert "--device-authorized" in command
    assert command[command.index("--scope") + 1] == "acoustic_only"
    assert command[command.index("--acoustic-threshold") + 1] == "0.5"
    assert "--recipient" not in command
    assert "--notify" not in command
    assert "--apply" not in command


def test_execute_demo_revalidates_report_and_keeps_retention_dry(
    project_root: Path,
) -> None:
    demo = _load_demo_script(project_root)
    request = demo.DemoRequest(
        audio_path="examples/authorized.wav",
        clip_id="demo-clip-001",
        consent_id="demo-consent-001",
        scope="acoustic_only",
        granted_at="2026-10-01T09:55:00+00:00",
        expires_at=None,
        source_dataset="authorized-local-demo",
        acoustic_threshold=0.5,
        audio_config=None,
        retention_config="configs/retention.example.json",
    )
    replies = iter(
        (
            {
                "clip_id": "demo-clip-001",
                "processing_scope": "acoustic_only",
                "outcome": "review",
                "risk_score": 40.0,
                "risk_severity": "medium",
                "review_required": True,
                "alert_candidate": False,
                "report_id": "report-demo",
                "report_path": "final-reports/report-demo/report.json",
                "report_reused": False,
                "audit_id": "audit-demo",
                "audit_path": "alert-audit/audit-demo/audit.json",
                "audit_reused": False,
                "alert_id": None,
                "alert_path": None,
                "notification_delivery": "not_sent",
                "alert_delivery_authorized": False,
            },
            {
                "report_id": "report-demo",
                "decision_id": "decision-demo",
                "evidence": [
                    {"kind": "acoustic", "input_status": "present"},
                    {"kind": "speech", "input_status": "not_permitted"},
                    {"kind": "language", "input_status": "not_permitted"},
                ],
                "summary": {
                    "outcome": "review",
                    "risk_score": 40.0,
                    "risk_severity": "medium",
                    "review_required": True,
                    "alert_candidate": False,
                    "notification_delivery": "not_sent",
                    "alert_delivery_authorized": False,
                },
            },
            {
                "policy_enabled": False,
                "scanned_report_count": 1,
                "scanned_audit_count": 1,
                "protected_report_count": 1,
                "protected_pending_alert_count": 0,
                "target_count": 0,
                "applied": False,
                "raw_audio_deleted": False,
                "notification_delivery": "not_sent",
            },
        )
    )
    commands: list[list[str]] = []

    def fake_runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(next(replies)), stderr="")

    result = demo.execute_demo(
        request,
        project_root=project_root,
        python_executable="python-demo",
        runner=fake_runner,
        progress=io.StringIO(),
    )

    assert [command[3] for command in commands] == [
        "evaluate",
        "inspect-report",
        "retention",
    ]
    assert "--apply" not in commands[2]
    assert result["status"] == "complete"
    assert result["human_review"] == {
        "policy_review_required": True,
        "status": "pending_external_manual_workflow",
    }
    assert result["retention_dry_run"]["applied"] is False
    assert result["safety"] == {
        "notification_delivery": "not_sent",
        "alert_delivery_authorized": False,
        "external_action_authorized": False,
    }


def test_execute_demo_stops_on_delivery_or_report_mismatch(project_root: Path) -> None:
    demo = _load_demo_script(project_root)
    request = demo.DemoRequest(
        audio_path="examples/authorized.wav",
        clip_id="demo-clip-001",
        consent_id="demo-consent-001",
        scope="acoustic_only",
        granted_at="2026-10-01T09:55:00+00:00",
        expires_at=None,
        source_dataset="authorized-local-demo",
        acoustic_threshold=0.5,
        audio_config=None,
        retention_config="configs/retention.example.json",
    )
    unsafe = {
        "processing_scope": "acoustic_only",
        "report_id": "report-demo",
        "report_path": "final-reports/report-demo/report.json",
        "notification_delivery": "sent",
        "alert_delivery_authorized": True,
    }

    def fake_runner(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=json.dumps(unsafe), stderr="")

    with pytest.raises(demo.DemoError, match="no-delivery") as error:
        demo.execute_demo(
            request,
            project_root=project_root,
            runner=fake_runner,
            progress=io.StringIO(),
        )
    assert error.value.code == "delivery_boundary_failed"


def test_demo_help_runs_without_model_packages(project_root: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_offline_mvp_demo.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "--confirm-consent-reviewed" in completed.stdout
    assert "--confirm-authorized-device" in completed.stdout
    assert "--preflight-only" in completed.stdout
    assert "plan retention without deleting anything" in completed.stdout
