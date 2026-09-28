"""A8.2 tests for local alerts and immutable audit records."""

from dataclasses import FrozenInstanceError
from datetime import timedelta
import json
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from audio_sentinel.alert_audit import (
    ALERT_AUDIT_DIRECTORY,
    ALERT_AUDIT_FILENAME,
    LOCAL_ALERT_FILENAME,
    AlertAuditAction,
    AlertAuditError,
    AlertAuditPersistencePolicy,
    AlertAuditReason,
    AlertAuditRecord,
    AlertReviewStatus,
    LocalAlertDocument,
    alert_audit_schema_documents,
    build_alert_audit,
    load_alert_audit,
    save_alert_audit,
)
from audio_sentinel.config import AudioSettings
from audio_sentinel.consensus_contracts import ConsensusOutcome
from audio_sentinel.contracts import ProcessingScope, RiskLevel
from audio_sentinel.final_report import save_final_report
from test_evaluator import NOW, evaluator, input_clip


def saved_report(
    settings,
    *,
    scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH,
    speech: bool = True,
):
    service, *_ = evaluator(settings, speech=speech)
    result = service.evaluate(
        input_clip(settings, scope),
        audio_settings=AudioSettings(
            normalize_loudness=False,
            window_seconds=(1.0,),
        ),
        now=NOW,
    )
    return save_final_report(settings.paths, result, now=NOW)


def report_relative(saved) -> str:
    return saved.report_path.relative_to(
        saved.report_path.parents[2]
    ).as_posix()


def audit_relative(audit_id: str) -> str:
    return f"{ALERT_AUDIT_DIRECTORY}/{audit_id}/{ALERT_AUDIT_FILENAME}"


def test_alert_outcome_builds_pending_local_alert_and_audit(temporary_settings):
    saved = saved_report(temporary_settings)

    audit, alert = build_alert_audit(saved.report, now=NOW)
    payload = json.dumps(
        {
            "audit": audit.model_dump(mode="json"),
            "alert": None if alert is None else alert.model_dump(mode="json"),
        }
    )

    assert alert is not None
    assert alert.alert_id.startswith("alert-")
    assert alert.report.report_id == saved.report.report_id
    assert alert.report.decision_id == saved.report.decision.decision_id
    assert alert.outcome is ConsensusOutcome.ALERT
    assert alert.risk_score == 92
    assert alert.risk_severity is RiskLevel.CRITICAL
    assert alert.review_status is AlertReviewStatus.PENDING
    assert alert.review_required is alert.local_only is True
    assert alert.notification_delivery == "not_sent"
    assert alert.alert_delivery_authorized is False
    assert audit.action is AlertAuditAction.LOCAL_ALERT_CREATED
    assert audit.reason is AlertAuditReason.ALERT_CANDIDATE_RECORDED
    assert audit.alert is not None and audit.alert.alert_id == alert.alert_id
    assert audit.notification_delivery == "not_sent"
    assert audit.alert_delivery_authorized is False
    assert "I will kill you" not in payload
    assert ".wav" not in payload
    assert '"transcript":' not in payload.lower()
    assert str(saved.report_path) not in payload


def test_non_alert_outcome_creates_audit_without_local_alert(temporary_settings):
    saved = saved_report(
        temporary_settings,
        scope=ProcessingScope.ACOUSTIC_ONLY,
        speech=False,
    )

    audit, alert = build_alert_audit(saved.report, now=NOW)

    assert saved.report.decision.outcome is ConsensusOutcome.REVIEW
    assert alert is None and audit.alert is None
    assert audit.action is AlertAuditAction.NO_ALERT_CREATED
    assert audit.reason is AlertAuditReason.CONSENSUS_OUTCOME_NOT_ALERT
    assert audit.decision_outcome is ConsensusOutcome.REVIEW
    assert audit.review_required is True


def test_ids_are_semantic_and_exclude_only_local_creation_time(
    temporary_settings,
):
    saved = saved_report(temporary_settings)

    first_audit, first_alert = build_alert_audit(saved.report, now=NOW)
    second_audit, second_alert = build_alert_audit(
        saved.report, now=NOW + timedelta(hours=1)
    )

    assert first_alert is not None and second_alert is not None
    assert first_alert.alert_id == second_alert.alert_id
    assert first_audit.audit_id == second_audit.audit_id
    assert first_alert.model_copy(
        update={"created_at": second_alert.created_at}
    ) == second_alert
    assert first_audit.model_copy(
        update={"created_at": second_audit.created_at}
    ) == second_audit


def test_local_handling_cannot_predate_final_report(temporary_settings):
    saved = saved_report(temporary_settings)

    with pytest.raises(AlertAuditError) as captured:
        build_alert_audit(saved.report, now=NOW - timedelta(seconds=1))

    assert captured.value.code == "invalid_time"


def test_save_load_and_retry_reuse_alert_bundle(temporary_settings):
    report = saved_report(temporary_settings)
    relative = report_relative(report)

    first = save_alert_audit(temporary_settings.paths, relative, now=NOW)
    second = save_alert_audit(
        temporary_settings.paths,
        relative,
        now=NOW + timedelta(minutes=5),
    )
    loaded = load_alert_audit(
        temporary_settings.paths,
        audit_relative(first.audit.audit_id),
    )

    assert first.reused is False
    assert second.reused is True
    assert loaded.audit == first.audit == second.audit
    assert loaded.alert == first.alert == second.alert
    assert first.alert_path == first.directory / LOCAL_ALERT_FILENAME
    assert {item.name for item in first.directory.iterdir()} == {
        ALERT_AUDIT_FILENAME,
        LOCAL_ALERT_FILENAME,
    }
    assert first.audit_path.read_bytes().endswith(b"\n")
    assert first.alert_path is not None
    assert first.alert_path.read_bytes().endswith(b"\n")


def test_save_non_alert_bundle_has_only_audit_document(temporary_settings):
    report = saved_report(
        temporary_settings,
        scope=ProcessingScope.ACOUSTIC_ONLY,
        speech=False,
    )

    saved = save_alert_audit(
        temporary_settings.paths,
        report_relative(report),
        now=NOW,
    )

    assert saved.alert is saved.alert_path is None
    assert {item.name for item in saved.directory.iterdir()} == {
        ALERT_AUDIT_FILENAME
    }
    reloaded = load_alert_audit(
        temporary_settings.paths,
        audit_relative(saved.audit.audit_id),
    )
    assert reloaded.alert is reloaded.alert_path is None
    assert reloaded.audit == saved.audit


def test_save_rejects_output_over_limit_without_creating_audit_directory(
    temporary_settings,
):
    report = saved_report(temporary_settings)

    with pytest.raises(AlertAuditError) as captured:
        save_alert_audit(
            temporary_settings.paths,
            report_relative(report),
            policy=AlertAuditPersistencePolicy(
                max_audit_bytes=32,
                max_alert_bytes=32,
            ),
            now=NOW,
        )

    assert captured.value.code == "output_too_large"
    assert not (
        temporary_settings.paths.processed_data / ALERT_AUDIT_DIRECTORY
    ).exists()


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../audit.json",
        Path("C:/outside/audit.json"),
    ),
)
def test_load_rejects_paths_outside_processed_data(
    temporary_settings,
    unsafe_path,
):
    with pytest.raises(AlertAuditError) as captured:
        load_alert_audit(temporary_settings.paths, unsafe_path)

    assert captured.value.code == "invalid_path"


def test_load_rejects_tampered_alert(temporary_settings):
    report = saved_report(temporary_settings)
    saved = save_alert_audit(
        temporary_settings.paths,
        report_relative(report),
        now=NOW,
    )
    assert saved.alert_path is not None
    payload = json.loads(saved.alert_path.read_text(encoding="utf-8"))
    payload["risk_score"] = 0
    saved.alert_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AlertAuditError) as captured:
        load_alert_audit(
            temporary_settings.paths,
            audit_relative(saved.audit.audit_id),
        )

    assert captured.value.code == "invalid_document"


def test_load_rejects_unexpected_files_and_wrong_identity_path(
    temporary_settings,
):
    report = saved_report(temporary_settings)
    saved = save_alert_audit(
        temporary_settings.paths,
        report_relative(report),
        now=NOW,
    )
    (saved.directory / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(AlertAuditError) as captured:
        load_alert_audit(
            temporary_settings.paths,
            audit_relative(saved.audit.audit_id),
        )
    assert captured.value.code == "output_conflict"

    (saved.directory / "unexpected.txt").unlink()
    wrong_id = "audit-" + "0" * 64
    wrong = (
        temporary_settings.paths.processed_data
        / ALERT_AUDIT_DIRECTORY
        / wrong_id
    )
    shutil.copytree(saved.directory, wrong)
    with pytest.raises(AlertAuditError) as captured:
        load_alert_audit(
            temporary_settings.paths,
            audit_relative(wrong_id),
        )
    assert captured.value.code == "output_conflict"


def test_load_requires_referenced_final_report(temporary_settings):
    report = saved_report(temporary_settings)
    saved = save_alert_audit(
        temporary_settings.paths,
        report_relative(report),
        now=NOW,
    )
    report.report_path.unlink()

    with pytest.raises(AlertAuditError) as captured:
        load_alert_audit(
            temporary_settings.paths,
            audit_relative(saved.audit.audit_id),
        )

    assert captured.value.code == "report_unavailable"


def test_load_honors_document_byte_limits(temporary_settings):
    report = saved_report(temporary_settings)
    saved = save_alert_audit(
        temporary_settings.paths,
        report_relative(report),
        now=NOW,
    )

    with pytest.raises(AlertAuditError) as captured:
        load_alert_audit(
            temporary_settings.paths,
            audit_relative(saved.audit.audit_id),
            policy=AlertAuditPersistencePolicy(
                max_audit_bytes=32,
                max_alert_bytes=32,
            ),
        )

    assert captured.value.code == "file_too_large"


def test_non_alert_contract_cannot_claim_alert_creation(temporary_settings):
    report = saved_report(
        temporary_settings,
        scope=ProcessingScope.ACOUSTIC_ONLY,
        speech=False,
    )
    audit, _ = build_alert_audit(report.report, now=NOW)

    with pytest.raises(ValidationError):
        AlertAuditRecord.model_validate(
            audit.model_dump(mode="python")
            | {"action": AlertAuditAction.LOCAL_ALERT_CREATED}
        )


def test_contracts_are_frozen_strict_and_schemas_are_checked_in(
    temporary_settings,
    project_root,
):
    report = saved_report(temporary_settings)
    audit, alert = build_alert_audit(report.report, now=NOW)
    assert alert is not None

    with pytest.raises(ValidationError):
        LocalAlertDocument.model_validate(
            alert.model_dump(mode="python") | {"recipient": "someone"}
        )
    with pytest.raises((ValidationError, FrozenInstanceError)):
        audit.action = AlertAuditAction.NO_ALERT_CREATED

    expected = {
        filename: json.loads(
            (project_root / "docs/schemas/v1" / filename).read_text(
                encoding="utf-8"
            )
        )
        for filename in (
            "alert-audit-record.schema.json",
            "local-alert.schema.json",
        )
    }
    assert alert_audit_schema_documents() == expected
