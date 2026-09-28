"""A8.2 local alert creation and immutable audit-record persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import errno
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.config import Paths
from audio_sentinel.consensus_contracts import (
    ConsensusOutcome,
    ConsensusReasonCode,
)
from audio_sentinel.contracts import RiskLevel
from audio_sentinel.feature_persistence import _read_bytes, _safe_file
from audio_sentinel.final_report import (
    FINAL_REPORT_DIRECTORY,
    FINAL_REPORT_FILENAME,
    FinalReportDocument,
    FinalReportError,
    load_final_report,
)
from audio_sentinel.persistence import _is_link, _remove_stage
from audio_sentinel.preparation import Identifier


ALERT_AUDIT_SCHEMA_VERSION = "1.0"
ALERT_AUDIT_FORMAT_VERSION = "1.0"
ALERT_AUDIT_DIRECTORY = "alert-audit"
ALERT_AUDIT_FILENAME = "audit.json"
LOCAL_ALERT_FILENAME = "alert.json"


class AlertAuditError(RuntimeError):
    """Stable A8.2 failure code plus a safe, non-sensitive message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class AlertAuditAction(str, Enum):
    """The local action recorded for one verified final report."""

    LOCAL_ALERT_CREATED = "local_alert_created"
    NO_ALERT_CREATED = "no_alert_created"


class AlertAuditReason(str, Enum):
    """Portable reason for the recorded local action."""

    ALERT_CANDIDATE_RECORDED = "alert_candidate_recorded"
    CONSENSUS_OUTCOME_NOT_ALERT = "consensus_outcome_not_alert"


class AlertReviewStatus(str, Enum):
    """A8.2 starts local alerts in a pending human-review state."""

    PENDING = "pending"


class AlertAuditRecordBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class FinalReportReference(AlertAuditRecordBase):
    """Hash-pinned final-report identity without a filesystem path."""

    report_id: Identifier
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    clip_id: Identifier
    decision_id: Identifier
    assessment_id: Identifier


class LocalAlertDocument(AlertAuditRecordBase):
    """Local-only alert candidate created from an alert-eligible final report."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/local-alert.schema.json"
        },
    )

    schema_version: Literal["1.0"] = ALERT_AUDIT_SCHEMA_VERSION
    document_type: Literal["local_alert"] = "local_alert"
    format_version: Literal["1.0"] = ALERT_AUDIT_FORMAT_VERSION
    alert_id: Identifier
    created_at: datetime
    report: FinalReportReference
    outcome: Literal[ConsensusOutcome.ALERT] = ConsensusOutcome.ALERT
    risk_score: float = Field(ge=0, le=100)
    risk_severity: Literal[RiskLevel.CRITICAL] = RiskLevel.CRITICAL
    decision_reason_codes: tuple[ConsensusReasonCode, ...] = Field(min_length=1)
    review_required: Literal[True] = True
    review_status: Literal[AlertReviewStatus.PENDING] = AlertReviewStatus.PENDING
    local_only: Literal[True] = True
    notification_delivery: Literal["not_sent"] = "not_sent"
    alert_delivery_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_document(self) -> "LocalAlertDocument":
        _validate_timestamp(self.created_at)
        if self.alert_id != _alert_id(self):
            raise ValueError("alert_id must match canonical local alert content")
        return self


class LocalAlertReference(AlertAuditRecordBase):
    """Semantic identity of the optional alert stored beside an audit record."""

    alert_id: Identifier
    alert_semantic_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class AlertAuditRecord(AlertAuditRecordBase):
    """Immutable receipt for the local handling of one final report."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": (
                "https://audio-sentinel.local/schemas/v1/"
                "alert-audit-record.schema.json"
            )
        },
    )

    schema_version: Literal["1.0"] = ALERT_AUDIT_SCHEMA_VERSION
    document_type: Literal["alert_audit_record"] = "alert_audit_record"
    format_version: Literal["1.0"] = ALERT_AUDIT_FORMAT_VERSION
    audit_id: Identifier
    created_at: datetime
    report: FinalReportReference
    decision_outcome: ConsensusOutcome
    action: AlertAuditAction
    reason: AlertAuditReason
    alert: LocalAlertReference | None = None
    review_required: bool
    notification_delivery: Literal["not_sent"] = "not_sent"
    alert_delivery_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_document(self) -> "AlertAuditRecord":
        _validate_timestamp(self.created_at)
        if self.decision_outcome is ConsensusOutcome.ALERT:
            if (
                self.action is not AlertAuditAction.LOCAL_ALERT_CREATED
                or self.reason is not AlertAuditReason.ALERT_CANDIDATE_RECORDED
                or self.alert is None
                or not self.review_required
            ):
                raise ValueError("alert outcome requires one pending local alert")
        elif (
            self.action is not AlertAuditAction.NO_ALERT_CREATED
            or self.reason is not AlertAuditReason.CONSENSUS_OUTCOME_NOT_ALERT
            or self.alert is not None
        ):
            raise ValueError("non-alert outcome cannot create a local alert")
        if self.audit_id != _audit_id(self):
            raise ValueError("audit_id must match canonical audit content")
        return self


class AlertAuditPersistencePolicy(AlertAuditRecordBase):
    """Resource limits for one A8.2 audit bundle."""

    max_audit_bytes: int = Field(default=4_194_304, gt=0, strict=True)
    max_alert_bytes: int = Field(default=4_194_304, gt=0, strict=True)


@dataclass(frozen=True)
class AlertAuditBundle:
    directory: Path
    audit_path: Path
    alert_path: Path | None
    audit: AlertAuditRecord
    alert: LocalAlertDocument | None


@dataclass(frozen=True)
class SavedAlertAuditBundle(AlertAuditBundle):
    reused: bool


def _validate_timestamp(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")


def _clock(now: datetime | None, report: FinalReportDocument) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    try:
        _validate_timestamp(value)
        normalized = value.astimezone(UTC)
        if normalized < report.created_at.astimezone(UTC):
            raise ValueError("local handling cannot predate the final report")
        return normalized
    except (OverflowError, ValueError) as error:
        raise AlertAuditError(
            "invalid_time",
            "Alert audit creation time must be timezone-aware and not predate the report.",
        ) from error


def _canonical_hash(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _artifact_hash(document: BaseModel) -> str:
    return _canonical_hash(document.model_dump(mode="json"))


def _alert_semantic_hash(alert: LocalAlertDocument) -> str:
    return _canonical_hash(
        alert.model_dump(mode="json", exclude={"alert_id", "created_at"})
    )


def _alert_id(alert: LocalAlertDocument) -> str:
    return f"alert-{_alert_semantic_hash(alert)}"


def _audit_id(audit: AlertAuditRecord) -> str:
    payload = audit.model_dump(mode="json", exclude={"audit_id", "created_at"})
    return f"audit-{_canonical_hash(payload)}"


def _validated_report(report: FinalReportDocument) -> FinalReportDocument:
    if not isinstance(report, FinalReportDocument):
        raise AlertAuditError(
            "invalid_report", "A validated B8.1 final report is required."
        )
    try:
        validated = FinalReportDocument.model_validate(
            report.model_dump(mode="python")
        )
    except Exception as error:
        raise AlertAuditError(
            "invalid_report", "The final report failed contract validation."
        ) from error
    if validated != report:
        raise AlertAuditError(
            "invalid_report", "The final report failed integrity validation."
        )
    return validated


def _report_reference(report: FinalReportDocument) -> FinalReportReference:
    return FinalReportReference(
        report_id=report.report_id,
        report_sha256=_artifact_hash(report),
        clip_id=report.source.clip_id,
        decision_id=report.decision.decision_id,
        assessment_id=report.risk_assessment.assessment_id,
    )


def build_alert_audit(
    report: FinalReportDocument,
    *,
    now: datetime | None = None,
) -> tuple[AlertAuditRecord, LocalAlertDocument | None]:
    """Build one local handling receipt and an alert only for an alert outcome."""

    try:
        validated = _validated_report(report)
        created_at = _clock(now, validated)
        reference = _report_reference(validated)
        alert: LocalAlertDocument | None = None
        if validated.decision.outcome is ConsensusOutcome.ALERT:
            provisional_alert = LocalAlertDocument.model_construct(
                schema_version=ALERT_AUDIT_SCHEMA_VERSION,
                document_type="local_alert",
                format_version=ALERT_AUDIT_FORMAT_VERSION,
                alert_id="alert-" + "0" * 64,
                created_at=created_at,
                report=reference,
                outcome=ConsensusOutcome.ALERT,
                risk_score=validated.risk_assessment.score,
                risk_severity=validated.risk_assessment.severity,
                decision_reason_codes=validated.decision.reason_codes,
                review_required=True,
                review_status=AlertReviewStatus.PENDING,
                local_only=True,
                notification_delivery="not_sent",
                alert_delivery_authorized=False,
            )
            alert = LocalAlertDocument.model_validate(
                provisional_alert.model_copy(
                    update={"alert_id": _alert_id(provisional_alert)}
                ).model_dump(mode="python")
            )

        action = (
            AlertAuditAction.LOCAL_ALERT_CREATED
            if alert is not None
            else AlertAuditAction.NO_ALERT_CREATED
        )
        reason = (
            AlertAuditReason.ALERT_CANDIDATE_RECORDED
            if alert is not None
            else AlertAuditReason.CONSENSUS_OUTCOME_NOT_ALERT
        )
        alert_reference = (
            None
            if alert is None
            else LocalAlertReference(
                alert_id=alert.alert_id,
                alert_semantic_sha256=_alert_semantic_hash(alert),
            )
        )
        provisional_audit = AlertAuditRecord.model_construct(
            schema_version=ALERT_AUDIT_SCHEMA_VERSION,
            document_type="alert_audit_record",
            format_version=ALERT_AUDIT_FORMAT_VERSION,
            audit_id="audit-" + "0" * 64,
            created_at=created_at,
            report=reference,
            decision_outcome=validated.decision.outcome,
            action=action,
            reason=reason,
            alert=alert_reference,
            review_required=validated.decision.review_required,
            notification_delivery="not_sent",
            alert_delivery_authorized=False,
        )
        audit = AlertAuditRecord.model_validate(
            provisional_audit.model_copy(
                update={"audit_id": _audit_id(provisional_audit)}
            ).model_dump(mode="python")
        )
        return audit, alert
    except AlertAuditError:
        raise
    except Exception as error:
        raise AlertAuditError(
            "invalid_report",
            "The final report could not form a valid local audit record.",
        ) from error


def _document_bytes(document: BaseModel) -> bytes:
    return (document.model_dump_json(indent=2) + "\n").encode("utf-8")


def _output_parent(paths: Paths) -> Path:
    try:
        root = paths.root.resolve(strict=True)
        processed = Path(os.path.abspath(paths.processed_data))
        parts = processed.relative_to(root).parts
        if not parts:
            raise ValueError("root is not an output directory")
        for other in (paths.raw_data, paths.interim_data, paths.models):
            protected = other.resolve()
            if processed.is_relative_to(protected) or protected.is_relative_to(
                processed
            ):
                raise ValueError("output overlaps protected project data")
        current = root
        for part in (*parts, ALERT_AUDIT_DIRECTORY):
            current = current / part
            if _is_link(current):
                raise ValueError("linked output")
            current.mkdir(exist_ok=True)
        return current.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise AlertAuditError(
            "invalid_output_path",
            "Alert audit output must remain inside separate, unlinked processed data.",
        ) from error


def _final_report_relative(report_id: str) -> str:
    return f"{FINAL_REPORT_DIRECTORY}/{report_id}/{FINAL_REPORT_FILENAME}"


def _validate_bundle(
    report: FinalReportDocument,
    audit: AlertAuditRecord,
    alert: LocalAlertDocument | None,
) -> None:
    expected_audit, expected_alert = build_alert_audit(
        report, now=audit.created_at
    )
    if audit != expected_audit or alert != expected_alert:
        raise AlertAuditError(
            "bundle_mismatch",
            "Alert audit documents do not match the referenced final report.",
        )


def _load(
    paths: Paths,
    audit_path: str | Path,
    policy: AlertAuditPersistencePolicy,
) -> AlertAuditBundle:
    try:
        path = _safe_file(paths.root, paths.processed_data, audit_path)
        audit = AlertAuditRecord.model_validate_json(
            _read_bytes(path, policy.max_audit_bytes)
        )
        expected_directory = (
            paths.processed_data / ALERT_AUDIT_DIRECTORY / audit.audit_id
        )
        expected_path = expected_directory / ALERT_AUDIT_FILENAME
        expected_inventory = {ALERT_AUDIT_FILENAME}
        if audit.alert is not None:
            expected_inventory.add(LOCAL_ALERT_FILENAME)
        if (
            path != expected_path
            or {item.name for item in path.parent.iterdir()} != expected_inventory
        ):
            raise AlertAuditError(
                "output_conflict",
                "Audit identity does not match its bundle path or inventory.",
            )

        alert: LocalAlertDocument | None = None
        alert_path: Path | None = None
        if audit.alert is not None:
            alert_relative = (
                f"{ALERT_AUDIT_DIRECTORY}/{audit.audit_id}/{LOCAL_ALERT_FILENAME}"
            )
            alert_path = _safe_file(
                paths.root, paths.processed_data, alert_relative
            )
            alert = LocalAlertDocument.model_validate_json(
                _read_bytes(alert_path, policy.max_alert_bytes)
            )
            if (
                alert.alert_id != audit.alert.alert_id
                or _alert_semantic_hash(alert)
                != audit.alert.alert_semantic_sha256
            ):
                raise AlertAuditError(
                    "bundle_mismatch",
                    "The local alert does not match its audit reference.",
                )

        try:
            report = load_final_report(
                paths, _final_report_relative(audit.report.report_id)
            )
        except FinalReportError as error:
            raise AlertAuditError(
                "report_unavailable",
                "The final report referenced by this audit could not be verified.",
            ) from error
        _validate_bundle(report, audit, alert)
        return AlertAuditBundle(
            directory=expected_directory,
            audit_path=expected_path,
            alert_path=alert_path,
            audit=audit,
            alert=alert,
        )
    except AlertAuditError:
        raise
    except MemoryError as error:
        raise AlertAuditError(
            "insufficient_memory", "Not enough memory to load the alert audit."
        ) from error
    except Exception as error:
        code = getattr(error, "code", "invalid_document")
        raise AlertAuditError(
            code, "Alert audit documents could not be read or verified."
        ) from error


def load_alert_audit(
    paths: Paths,
    audit_path: str | Path,
    *,
    policy: AlertAuditPersistencePolicy | None = None,
) -> AlertAuditBundle:
    """Reload one relative audit path and verify its report and optional alert."""

    resolved = AlertAuditPersistencePolicy.model_validate(
        (policy or AlertAuditPersistencePolicy()).model_dump(mode="python")
    )
    return _load(paths, audit_path, resolved)


def save_alert_audit(
    paths: Paths,
    report_path: str | Path,
    *,
    policy: AlertAuditPersistencePolicy | None = None,
    now: datetime | None = None,
) -> SavedAlertAuditBundle:
    """Create and atomically persist local handling for one saved final report."""

    resolved = AlertAuditPersistencePolicy.model_validate(
        (policy or AlertAuditPersistencePolicy()).model_dump(mode="python")
    )
    stage: Path | None = None
    parent: Path | None = None
    try:
        try:
            report = load_final_report(paths, report_path)
        except FinalReportError as error:
            raise AlertAuditError(
                "invalid_report",
                "The saved final report could not be verified for local handling.",
            ) from error
        audit, alert = build_alert_audit(report, now=now)
        audit_raw = _document_bytes(audit)
        if len(audit_raw) > resolved.max_audit_bytes:
            raise AlertAuditError(
                "output_too_large",
                "The alert audit exceeds its configured byte limit.",
            )
        alert_raw = None if alert is None else _document_bytes(alert)
        if alert_raw is not None and len(alert_raw) > resolved.max_alert_bytes:
            raise AlertAuditError(
                "output_too_large",
                "The local alert exceeds its configured byte limit.",
            )

        parent = _output_parent(paths)
        destination = parent / audit.audit_id
        relative = (
            f"{ALERT_AUDIT_DIRECTORY}/{audit.audit_id}/{ALERT_AUDIT_FILENAME}"
        )
        if _is_link(destination) or (
            destination.exists() and not destination.is_dir()
        ):
            raise AlertAuditError(
                "output_conflict",
                "Alert audit destination is not a regular bundle directory.",
            )

        stage = Path(mkdtemp(prefix=".pending-", dir=parent)).resolve()
        staged_audit = stage / ALERT_AUDIT_FILENAME
        with staged_audit.open("xb") as stream:
            stream.write(audit_raw)
            stream.flush()
            os.fsync(stream.fileno())
        staged_alert: Path | None = None
        if alert_raw is not None:
            staged_alert = stage / LOCAL_ALERT_FILENAME
            with staged_alert.open("xb") as stream:
                stream.write(alert_raw)
                stream.flush()
                os.fsync(stream.fileno())

        verified_audit = AlertAuditRecord.model_validate_json(
            _read_bytes(staged_audit, resolved.max_audit_bytes)
        )
        verified_alert = (
            None
            if staged_alert is None
            else LocalAlertDocument.model_validate_json(
                _read_bytes(staged_alert, resolved.max_alert_bytes)
            )
        )
        if verified_audit != audit or verified_alert != alert:
            raise AlertAuditError(
                "verification_failed", "Staged alert audit failed readback."
            )
        _validate_bundle(report, verified_audit, verified_alert)

        reused = destination.exists()
        if not reused:
            try:
                stage.rename(destination)
                stage = None
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) and not isinstance(
                    error, FileExistsError
                ):
                    raise
                reused = True
        if reused:
            existing = _load(paths, relative, resolved)
            comparable_audit = existing.audit.model_copy(
                update={"created_at": audit.created_at}
            )
            comparable_alert = (
                None
                if existing.alert is None
                else existing.alert.model_copy(
                    update={"created_at": alert.created_at if alert else audit.created_at}
                )
            )
            if comparable_audit != audit or comparable_alert != alert:
                raise AlertAuditError(
                    "output_conflict",
                    "Existing alert audit differs; it was not overwritten.",
                )
            audit = existing.audit
            alert = existing.alert

        loaded = _load(paths, relative, resolved)
        if loaded.audit != audit or loaded.alert != alert:
            raise AlertAuditError(
                "verification_failed", "Persisted alert audit failed verification."
            )
        return SavedAlertAuditBundle(
            directory=loaded.directory,
            audit_path=loaded.audit_path,
            alert_path=loaded.alert_path,
            audit=loaded.audit,
            alert=loaded.alert,
            reused=reused,
        )
    except AlertAuditError:
        raise
    except MemoryError as error:
        raise AlertAuditError(
            "insufficient_memory", "Not enough memory to persist the alert audit."
        ) from error
    except OSError as error:
        raise AlertAuditError(
            "write_failed", "The alert audit could not be written or verified."
        ) from error
    finally:
        if stage is not None and parent is not None:
            _remove_stage(stage, parent)


def alert_audit_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public JSON Schemas for the A8.2 documents."""

    return {
        "alert-audit-record.schema.json": AlertAuditRecord.model_json_schema(),
        "local-alert.schema.json": LocalAlertDocument.model_json_schema(),
    }


def write_alert_audit_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the A8.2 JSON Schemas for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in alert_audit_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        exported[filename] = destination
    return exported
