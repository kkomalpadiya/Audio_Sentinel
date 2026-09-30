"""B9.2 retention, deletion safety, and local audit-log tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import timedelta
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import audio_sentinel.cli as cli
from audio_sentinel.alert_audit import save_alert_audit
from audio_sentinel.config import RetentionSettings, load_settings
from audio_sentinel.contracts import ProcessingScope
from audio_sentinel.retention import (
    RETENTION_AUDIT_DIRECTORY,
    RETENTION_AUDIT_FILENAME,
    RetentionAuditPersistencePolicy,
    RetentionAuditRecord,
    RetentionError,
    load_retention_audit,
    plan_retention,
    retention_audit_schema_documents,
    run_retention,
)
from test_alert_audit import report_relative, saved_report
from test_evaluator import NOW


runner = CliRunner()


def non_alert_bundle(settings, *, audit_time=NOW):
    report = saved_report(
        settings,
        scope=ProcessingScope.ACOUSTIC_ONLY,
        speech=False,
    )
    audit = save_alert_audit(
        settings.paths,
        report_relative(report),
        now=audit_time,
    )
    return report, audit


def alert_bundle(settings, *, audit_time=NOW):
    report = saved_report(settings)
    audit = save_alert_audit(
        settings.paths,
        report_relative(report),
        now=audit_time,
    )
    return report, audit


def enabled_policy(**updates) -> RetentionSettings:
    values = {
        "enabled": True,
        "final_report_days": 30,
        "alert_audit_days": 30,
        "allow_pending_alert_deletion": False,
        "max_delete_count": 100,
    }
    values.update(updates)
    return RetentionSettings(**values)


def audit_relative(audit_id: str) -> str:
    return f"{RETENTION_AUDIT_DIRECTORY}/{audit_id}/{RETENTION_AUDIT_FILENAME}"


def test_retention_settings_default_to_disabled_and_load_explicit_config(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Project_1"
    root.mkdir()
    config = root / "retention.json"
    config.write_text(
        json.dumps(
            {
                "enabled": True,
                "final_report_days": None,
                "alert_audit_days": 45,
                "allow_pending_alert_deletion": True,
                "max_delete_count": 12,
            }
        ),
        encoding="utf-8",
    )

    assert RetentionSettings().enabled is False
    loaded = load_settings(root, retention_config_path=config)
    assert loaded.retention == RetentionSettings(
        enabled=True,
        final_report_days=None,
        alert_audit_days=45,
        allow_pending_alert_deletion=True,
        max_delete_count=12,
    )
    with pytest.raises(ValidationError):
        RetentionSettings(extra_setting=True)
    with pytest.raises(ValidationError):
        RetentionSettings(final_report_days="30")


def test_plan_is_dry_run_and_selects_expired_non_alert_bundle(
    temporary_settings,
) -> None:
    report, audit = non_alert_bundle(temporary_settings)

    result = run_retention(
        temporary_settings.paths,
        enabled_policy(),
        now=NOW + timedelta(days=31),
    )

    assert result.applied is False
    assert [item.kind for item in result.plan.targets] == [
        "alert_audit",
        "final_report",
    ]
    assert report.directory.is_dir()
    assert audit.directory.is_dir()
    assert result.audit is result.audit_path is None
    assert not (
        temporary_settings.paths.processed_data / RETENTION_AUDIT_DIRECTORY
    ).exists()


def test_apply_deletes_verified_bundles_and_writes_privacy_minimized_audit(
    temporary_settings,
) -> None:
    report, alert_audit = non_alert_bundle(temporary_settings)

    result = run_retention(
        temporary_settings.paths,
        enabled_policy(),
        apply=True,
        now=NOW + timedelta(days=31),
    )

    assert result.applied is True
    assert not report.directory.exists()
    assert not alert_audit.directory.exists()
    assert result.audit is not None and result.audit_path is not None
    assert result.audit.deleted_targets == result.plan.targets
    assert result.audit.local_only is True
    assert result.audit.raw_audio_deleted is False
    assert result.audit.notification_delivery == "not_sent"
    assert result.audit.external_deletion_authorized is False
    loaded = load_retention_audit(
        temporary_settings.paths,
        audit_relative(result.audit.audit_id),
    )
    assert loaded == result.audit
    serialized = result.audit_path.read_text(encoding="utf-8")
    assert str(temporary_settings.paths.root) not in serialized
    assert ".wav" not in serialized
    assert "consent_id" not in serialized
    assert "transcript" not in serialized.lower()


def test_retained_audit_protects_expired_final_report(temporary_settings) -> None:
    report, audit = non_alert_bundle(
        temporary_settings,
        audit_time=NOW + timedelta(days=10),
    )

    plan = plan_retention(
        temporary_settings.paths,
        enabled_policy(final_report_days=30, alert_audit_days=60),
        now=NOW + timedelta(days=40),
    )

    assert plan.targets == ()
    assert plan.protected_report_count == 1
    assert report.directory.is_dir() and audit.directory.is_dir()


def test_pending_alert_is_protected_unless_policy_explicitly_allows_deletion(
    temporary_settings,
) -> None:
    report, audit = alert_bundle(temporary_settings)
    current = NOW + timedelta(days=31)

    protected = plan_retention(
        temporary_settings.paths,
        enabled_policy(),
        now=current,
    )
    allowed = plan_retention(
        temporary_settings.paths,
        enabled_policy(allow_pending_alert_deletion=True),
        now=current,
    )

    assert protected.targets == ()
    assert protected.protected_pending_alert_count == 1
    assert protected.protected_report_count == 1
    assert [item.kind for item in allowed.targets] == [
        "alert_audit",
        "final_report",
    ]
    assert report.directory.is_dir() and audit.directory.is_dir()


def test_apply_requires_enabled_policy_and_stops_before_delete_limit(
    temporary_settings,
) -> None:
    report, audit = non_alert_bundle(temporary_settings)
    current = NOW + timedelta(days=31)

    with pytest.raises(RetentionError) as disabled:
        run_retention(
            temporary_settings.paths,
            RetentionSettings(),
            apply=True,
            now=current,
        )
    assert disabled.value.code == "retention_disabled"

    with pytest.raises(RetentionError) as limited:
        run_retention(
            temporary_settings.paths,
            enabled_policy(max_delete_count=1),
            apply=True,
            now=current,
        )
    assert limited.value.code == "delete_limit_exceeded"
    assert report.directory.is_dir() and audit.directory.is_dir()


def test_audit_write_failure_rolls_back_every_selected_bundle(
    temporary_settings,
) -> None:
    report, audit = non_alert_bundle(temporary_settings)

    with pytest.raises(RetentionError) as captured:
        run_retention(
            temporary_settings.paths,
            enabled_policy(),
            apply=True,
            now=NOW + timedelta(days=31),
            audit_policy=RetentionAuditPersistencePolicy(max_document_bytes=32),
        )

    assert captured.value.code == "output_too_large"
    assert report.directory.is_dir() and audit.directory.is_dir()
    assert not list(
        temporary_settings.paths.processed_data.glob(".retention-delete-*")
    )


def test_invalid_or_tampered_inventory_fails_closed_before_deletion(
    temporary_settings,
) -> None:
    report, audit = non_alert_bundle(temporary_settings)
    (audit.directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(RetentionError) as captured:
        run_retention(
            temporary_settings.paths,
            enabled_policy(),
            apply=True,
            now=NOW + timedelta(days=31),
        )

    assert captured.value.code == "invalid_inventory"
    assert report.directory.is_dir() and audit.directory.is_dir()
    assert not (
        temporary_settings.paths.processed_data / RETENTION_AUDIT_DIRECTORY
    ).exists()


def test_apply_can_delete_audit_while_retaining_younger_report(
    temporary_settings,
) -> None:
    report, audit = non_alert_bundle(temporary_settings)

    result = run_retention(
        temporary_settings.paths,
        enabled_policy(final_report_days=60, alert_audit_days=30),
        apply=True,
        now=NOW + timedelta(days=31),
    )

    assert [(item.kind, item.artifact_id) for item in result.plan.targets] == [
        ("alert_audit", audit.audit.audit_id)
    ]
    assert report.directory.is_dir()
    assert not audit.directory.exists()


def test_retention_audit_rejects_path_escape_tampering_and_size_limit(
    temporary_settings,
) -> None:
    non_alert_bundle(temporary_settings)
    result = run_retention(
        temporary_settings.paths,
        enabled_policy(),
        apply=True,
        now=NOW + timedelta(days=31),
    )
    assert result.audit is not None and result.audit_path is not None

    with pytest.raises(RetentionError) as outside:
        load_retention_audit(temporary_settings.paths, "../audit.json")
    assert outside.value.code == "invalid_path"

    with pytest.raises(RetentionError) as bounded:
        load_retention_audit(
            temporary_settings.paths,
            audit_relative(result.audit.audit_id),
            policy=RetentionAuditPersistencePolicy(max_document_bytes=32),
        )
    assert bounded.value.code == "file_too_large"

    payload = json.loads(result.audit_path.read_text(encoding="utf-8"))
    payload["raw_audio_deleted"] = True
    result.audit_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RetentionError) as tampered:
        load_retention_audit(
            temporary_settings.paths,
            audit_relative(result.audit.audit_id),
        )
    assert tampered.value.code == "invalid_document"


def test_retention_contract_is_frozen_and_schema_is_checked_in(
    temporary_settings,
    project_root,
) -> None:
    non_alert_bundle(temporary_settings)
    result = run_retention(
        temporary_settings.paths,
        enabled_policy(),
        apply=True,
        now=NOW + timedelta(days=31),
    )
    assert result.audit is not None

    with pytest.raises((ValidationError, FrozenInstanceError)):
        result.audit.raw_audio_deleted = True
    with pytest.raises(ValidationError):
        RetentionAuditRecord.model_validate(
            result.audit.model_dump(mode="python") | {"recipient": "someone"}
        )
    expected = json.loads(
        (
            project_root
            / "docs"
            / "schemas"
            / "v1"
            / "retention-audit.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert retention_audit_schema_documents() == {
        "retention-audit.schema.json": expected
    }


def test_cli_dry_run_is_non_destructive_and_apply_needs_enabled_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Project_1"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    config = root / "retention.json"
    config.write_text(
        json.dumps(RetentionSettings().model_dump(mode="json")),
        encoding="utf-8",
    )
    args = [
        "retention",
        "--project-root",
        str(root),
        "--retention-config",
        "retention.json",
        "--compact",
    ]

    dry_run = runner.invoke(cli.app, args)
    applied = runner.invoke(cli.app, args + ["--apply"])

    assert dry_run.exit_code == 0, dry_run.stderr
    payload = json.loads(dry_run.stdout)
    assert payload["applied"] is False
    assert payload["target_count"] == 0
    assert payload["raw_audio_deleted"] is False
    assert applied.exit_code == 1
    assert json.loads(applied.stderr)["code"] == "retention_disabled"


def test_cli_rejects_retention_config_outside_project(tmp_path: Path) -> None:
    root = tmp_path / "Project_1"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    outside = tmp_path / "retention.json"
    outside.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "retention",
            "--project-root",
            str(root),
            "--retention-config",
            str(outside),
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "invalid_retention_config"
