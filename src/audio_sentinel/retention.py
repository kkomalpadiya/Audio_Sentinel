"""B9.2 dependency-aware retention and locally audited deletion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import mkdtemp
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.alert_audit import (
    ALERT_AUDIT_DIRECTORY,
    ALERT_AUDIT_FILENAME,
    AlertAuditBundle,
    load_alert_audit,
)
from audio_sentinel.config import Paths, RetentionSettings
from audio_sentinel.feature_persistence import _read_bytes, _safe_file
from audio_sentinel.final_report import (
    FINAL_REPORT_DIRECTORY,
    FINAL_REPORT_FILENAME,
    FinalReportDocument,
    load_final_report,
)
from audio_sentinel.persistence import _is_link
from audio_sentinel.preparation import Identifier


RETENTION_SCHEMA_VERSION = "1.0"
RETENTION_FORMAT_VERSION = "1.0"
RETENTION_AUDIT_DIRECTORY = "retention-audit"
RETENTION_AUDIT_FILENAME = "audit.json"
_REPORT_ID = re.compile(r"^report-[a-f0-9]{64}$")
_AUDIT_ID = re.compile(r"^audit-[a-f0-9]{64}$")


class RetentionError(RuntimeError):
    """Stable B9.2 failure code plus a safe, non-sensitive message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RetentionRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class RetentionTarget(RetentionRecord):
    """One verified bundle selected by opaque identity, never by caller path."""

    kind: Literal["alert_audit", "final_report"]
    artifact_id: Identifier
    created_at: datetime

    @model_validator(mode="after")
    def validate_target(self) -> "RetentionTarget":
        _validate_time(self.created_at)
        expected = _AUDIT_ID if self.kind == "alert_audit" else _REPORT_ID
        if expected.fullmatch(self.artifact_id) is None:
            raise ValueError("artifact identity does not match retention target kind")
        return self


class RetentionPlan(RetentionRecord):
    """A bounded, verified deletion plan; planning itself changes no artifacts."""

    generated_at: datetime
    policy: RetentionSettings
    scanned_report_count: int = Field(ge=0, strict=True)
    scanned_audit_count: int = Field(ge=0, strict=True)
    protected_report_count: int = Field(ge=0, strict=True)
    protected_pending_alert_count: int = Field(ge=0, strict=True)
    targets: tuple[RetentionTarget, ...]

    @model_validator(mode="after")
    def validate_plan(self) -> "RetentionPlan":
        _validate_time(self.generated_at)
        if self.protected_report_count > self.scanned_report_count:
            raise ValueError("protected report count exceeds scanned reports")
        if self.protected_pending_alert_count > self.scanned_audit_count:
            raise ValueError("protected pending-alert count exceeds scanned audits")
        if tuple(sorted(self.targets, key=_target_sort_key)) != self.targets:
            raise ValueError("retention targets must use canonical ordering")
        if len({(item.kind, item.artifact_id) for item in self.targets}) != len(
            self.targets
        ):
            raise ValueError("retention targets must be unique")
        if len(self.targets) > self.policy.max_delete_count:
            raise ValueError("retention plan exceeds max_delete_count")
        return self


class RetentionAuditRecord(RetentionRecord):
    """Immutable local receipt for one successfully applied deletion plan."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/retention-audit.schema.json"
        },
    )

    schema_version: Literal["1.0"] = RETENTION_SCHEMA_VERSION
    document_type: Literal["retention_deletion_audit"] = "retention_deletion_audit"
    format_version: Literal["1.0"] = RETENTION_FORMAT_VERSION
    audit_id: Identifier
    executed_at: datetime
    policy: RetentionSettings
    scanned_report_count: int = Field(ge=0, strict=True)
    scanned_audit_count: int = Field(ge=0, strict=True)
    protected_report_count: int = Field(ge=0, strict=True)
    protected_pending_alert_count: int = Field(ge=0, strict=True)
    deleted_targets: tuple[RetentionTarget, ...]
    local_only: Literal[True] = True
    raw_audio_deleted: Literal[False] = False
    notification_delivery: Literal["not_sent"] = "not_sent"
    external_deletion_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_audit(self) -> "RetentionAuditRecord":
        _validate_time(self.executed_at)
        if not self.policy.enabled:
            raise ValueError("applied retention requires an enabled policy")
        if self.protected_report_count > self.scanned_report_count:
            raise ValueError("protected report count exceeds scanned reports")
        if self.protected_pending_alert_count > self.scanned_audit_count:
            raise ValueError("protected pending-alert count exceeds scanned audits")
        if tuple(sorted(self.deleted_targets, key=_target_sort_key)) != self.deleted_targets:
            raise ValueError("deleted targets must use canonical ordering")
        if len(
            {(item.kind, item.artifact_id) for item in self.deleted_targets}
        ) != len(self.deleted_targets):
            raise ValueError("deleted targets must be unique")
        if len(self.deleted_targets) > self.policy.max_delete_count:
            raise ValueError("deleted targets exceed max_delete_count")
        if self.audit_id != _retention_audit_id(self):
            raise ValueError("audit_id must match canonical retention audit content")
        return self


class RetentionAuditPersistencePolicy(RetentionRecord):
    max_document_bytes: int = Field(default=4_194_304, gt=0, strict=True)


@dataclass(frozen=True)
class RetentionRunResult:
    plan: RetentionPlan
    applied: bool
    audit: RetentionAuditRecord | None
    audit_path: Path | None
    audit_reused: bool


def _validate_time(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    try:
        _validate_time(value)
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise RetentionError(
            "invalid_time", "Retention time must include a valid UTC offset."
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


def _retention_audit_id(record: RetentionAuditRecord) -> str:
    payload = record.model_dump(mode="json", exclude={"audit_id"})
    return f"retention-{_canonical_hash(payload)}"


def _target_sort_key(target: RetentionTarget) -> tuple[int, str]:
    return (0 if target.kind == "alert_audit" else 1, target.artifact_id)


def _managed_processed(paths: Paths, *, create: bool) -> Path:
    try:
        root = paths.root.resolve(strict=True)
        processed = Path(os.path.abspath(paths.processed_data))
        parts = processed.relative_to(root).parts
        if not parts:
            raise ValueError("processed data cannot be the project root")
        for other in (paths.raw_data, paths.interim_data, paths.models):
            other = Path(os.path.abspath(other))
            if processed.is_relative_to(other) or other.is_relative_to(processed):
                raise ValueError("managed directories overlap")
        current = root
        for part in parts:
            current = current / part
            if _is_link(current):
                raise ValueError("linked processed directory")
            if create:
                current.mkdir(exist_ok=True)
            elif not current.exists():
                return processed
            if current.exists() and not current.is_dir():
                raise ValueError("processed component is not a directory")
        return current.resolve(strict=True) if current.exists() else processed
    except (OSError, RuntimeError, ValueError) as error:
        raise RetentionError(
            "invalid_output_path",
            "Retention storage must remain in unlinked processed data.",
        ) from error


def _bundle_directories(base: Path, name: str, pattern: re.Pattern[str]) -> tuple[Path, ...]:
    parent = base / name
    if not parent.exists():
        return ()
    try:
        if _is_link(parent) or not parent.is_dir() or parent.resolve() != parent:
            raise ValueError("invalid parent")
        children = []
        for child in parent.iterdir():
            if (
                _is_link(child)
                or not child.is_dir()
                or child.resolve() != child
                or pattern.fullmatch(child.name) is None
            ):
                raise ValueError("invalid bundle inventory")
            children.append(child)
        return tuple(sorted(children, key=lambda item: item.name))
    except (OSError, RuntimeError, ValueError) as error:
        raise RetentionError(
            "invalid_inventory",
            "Retention stopped because a managed bundle inventory was not canonical.",
        ) from error


def _verified_reports(paths: Paths, base: Path) -> dict[str, FinalReportDocument]:
    reports: dict[str, FinalReportDocument] = {}
    try:
        for directory in _bundle_directories(base, FINAL_REPORT_DIRECTORY, _REPORT_ID):
            relative = (
                f"{FINAL_REPORT_DIRECTORY}/{directory.name}/{FINAL_REPORT_FILENAME}"
            )
            report = load_final_report(paths, relative)
            reports[report.report_id] = report
        return reports
    except RetentionError:
        raise
    except Exception as error:
        raise RetentionError(
            "invalid_inventory",
            "Retention stopped because a final-report bundle failed verification.",
        ) from error


def _verified_audits(paths: Paths, base: Path) -> dict[str, AlertAuditBundle]:
    audits: dict[str, AlertAuditBundle] = {}
    try:
        for directory in _bundle_directories(base, ALERT_AUDIT_DIRECTORY, _AUDIT_ID):
            relative = f"{ALERT_AUDIT_DIRECTORY}/{directory.name}/{ALERT_AUDIT_FILENAME}"
            bundle = load_alert_audit(paths, relative)
            audits[bundle.audit.audit_id] = bundle
        return audits
    except RetentionError:
        raise
    except Exception as error:
        raise RetentionError(
            "invalid_inventory",
            "Retention stopped because an alert-audit bundle failed verification.",
        ) from error


def _expired(created_at: datetime, days: int | None, now: datetime) -> bool:
    return days is not None and created_at.astimezone(UTC) <= now - timedelta(days=days)


def plan_retention(
    paths: Paths,
    policy: RetentionSettings,
    *,
    now: datetime | None = None,
) -> RetentionPlan:
    """Verify inventories and return a bounded plan without modifying them."""

    resolved = RetentionSettings.model_validate(policy.model_dump(mode="python"))
    current = _clock(now)
    base = _managed_processed(paths, create=False)
    if not base.exists():
        reports: dict[str, FinalReportDocument] = {}
        audits: dict[str, AlertAuditBundle] = {}
    else:
        reports = _verified_reports(paths, base)
        audits = _verified_audits(paths, base)

    audit_targets: dict[str, RetentionTarget] = {}
    protected_pending = 0
    for audit_id, bundle in audits.items():
        if not _expired(bundle.audit.created_at, resolved.alert_audit_days, current):
            continue
        if bundle.alert is not None and not resolved.allow_pending_alert_deletion:
            protected_pending += 1
            continue
        audit_targets[audit_id] = RetentionTarget(
            kind="alert_audit",
            artifact_id=audit_id,
            created_at=bundle.audit.created_at,
        )

    retained_report_references = {
        bundle.audit.report.report_id
        for audit_id, bundle in audits.items()
        if audit_id not in audit_targets
    }
    report_targets: dict[str, RetentionTarget] = {}
    protected_reports = 0
    for report_id, report in reports.items():
        if not _expired(report.created_at, resolved.final_report_days, current):
            continue
        if report_id in retained_report_references:
            protected_reports += 1
            continue
        report_targets[report_id] = RetentionTarget(
            kind="final_report",
            artifact_id=report_id,
            created_at=report.created_at,
        )

    targets = tuple(
        sorted(
            (*audit_targets.values(), *report_targets.values()),
            key=_target_sort_key,
        )
    )
    if len(targets) > resolved.max_delete_count:
        raise RetentionError(
            "delete_limit_exceeded",
            "Retention stopped before deletion because the plan exceeds max_delete_count.",
        )
    return RetentionPlan(
        generated_at=current,
        policy=resolved,
        scanned_report_count=len(reports),
        scanned_audit_count=len(audits),
        protected_report_count=protected_reports,
        protected_pending_alert_count=protected_pending,
        targets=targets,
    )


def _audit_record(plan: RetentionPlan) -> RetentionAuditRecord:
    provisional = RetentionAuditRecord.model_construct(
        schema_version=RETENTION_SCHEMA_VERSION,
        document_type="retention_deletion_audit",
        format_version=RETENTION_FORMAT_VERSION,
        audit_id="retention-provisional",
        executed_at=plan.generated_at,
        policy=plan.policy,
        scanned_report_count=plan.scanned_report_count,
        scanned_audit_count=plan.scanned_audit_count,
        protected_report_count=plan.protected_report_count,
        protected_pending_alert_count=plan.protected_pending_alert_count,
        deleted_targets=plan.targets,
        local_only=True,
        raw_audio_deleted=False,
        notification_delivery="not_sent",
        external_deletion_authorized=False,
    )
    return RetentionAuditRecord.model_validate(
        provisional.model_copy(
            update={"audit_id": _retention_audit_id(provisional)}
        ).model_dump(mode="python")
    )


def _document_bytes(record: RetentionAuditRecord) -> bytes:
    return (record.model_dump_json(indent=2) + "\n").encode("utf-8")


def _retention_audit_parent(paths: Paths) -> Path:
    base = _managed_processed(paths, create=True)
    parent = base / RETENTION_AUDIT_DIRECTORY
    try:
        if _is_link(parent):
            raise ValueError("linked audit directory")
        parent.mkdir(exist_ok=True)
        if not parent.is_dir() or parent.resolve() != parent:
            raise ValueError("invalid audit directory")
        return parent
    except (OSError, RuntimeError, ValueError) as error:
        raise RetentionError(
            "invalid_output_path",
            "Retention audit storage must be an unlinked local directory.",
        ) from error


def _save_retention_audit(
    paths: Paths,
    record: RetentionAuditRecord,
    persistence: RetentionAuditPersistencePolicy,
) -> tuple[Path, bool]:
    raw = _document_bytes(record)
    if len(raw) > persistence.max_document_bytes:
        raise RetentionError(
            "output_too_large", "Retention audit exceeds its configured byte limit."
        )
    parent = _retention_audit_parent(paths)
    destination = parent / record.audit_id
    stage: Path | None = None
    try:
        if _is_link(destination) or (
            destination.exists() and not destination.is_dir()
        ):
            raise RetentionError(
                "output_conflict",
                "Retention audit destination is not a regular bundle directory.",
            )
        stage = Path(mkdtemp(prefix=".pending-", dir=parent)).resolve()
        path = stage / RETENTION_AUDIT_FILENAME
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if RetentionAuditRecord.model_validate_json(
            _read_bytes(path, persistence.max_document_bytes)
        ) != record:
            raise RetentionError(
                "verification_failed", "Retention audit failed staged readback."
            )
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
            loaded = load_retention_audit(
                paths,
                f"{RETENTION_AUDIT_DIRECTORY}/{record.audit_id}/{RETENTION_AUDIT_FILENAME}",
                policy=persistence,
            )
            if loaded != record:
                raise RetentionError(
                    "output_conflict", "Existing retention audit differs."
                )
        saved_path = destination / RETENTION_AUDIT_FILENAME
        if load_retention_audit(
            paths,
            f"{RETENTION_AUDIT_DIRECTORY}/{record.audit_id}/{RETENTION_AUDIT_FILENAME}",
            policy=persistence,
        ) != record:
            raise RetentionError(
                "verification_failed", "Persisted retention audit failed verification."
            )
        return saved_path, reused
    except RetentionError:
        raise
    except OSError as error:
        raise RetentionError(
            "write_failed", "Retention audit could not be written or verified."
        ) from error
    finally:
        if stage is not None and stage.exists():
            _remove_private_tree(stage, parent, ".pending-")


def load_retention_audit(
    paths: Paths,
    audit_path: str | Path,
    *,
    policy: RetentionAuditPersistencePolicy | None = None,
) -> RetentionAuditRecord:
    """Load one canonical retention receipt relative to processed data."""

    resolved = RetentionAuditPersistencePolicy.model_validate(
        (policy or RetentionAuditPersistencePolicy()).model_dump(mode="python")
    )
    try:
        path = _safe_file(paths.root, paths.processed_data, audit_path)
        record = RetentionAuditRecord.model_validate_json(
            _read_bytes(path, resolved.max_document_bytes)
        )
        expected = (
            paths.processed_data
            / RETENTION_AUDIT_DIRECTORY
            / record.audit_id
            / RETENTION_AUDIT_FILENAME
        )
        if path != expected or {item.name for item in path.parent.iterdir()} != {
            RETENTION_AUDIT_FILENAME
        }:
            raise RetentionError(
                "output_conflict",
                "Retention audit identity does not match its bundle path or inventory.",
            )
        return record
    except RetentionError:
        raise
    except MemoryError as error:
        raise RetentionError(
            "insufficient_memory", "Not enough memory to load the retention audit."
        ) from error
    except Exception as error:
        code = getattr(error, "code", "invalid_document")
        raise RetentionError(
            code, "Retention audit could not be read or verified."
        ) from error


def _remove_private_tree(directory: Path, parent: Path, prefix: str) -> None:
    try:
        if (
            directory.parent != parent
            or not directory.name.startswith(prefix)
            or _is_link(directory)
            or directory.resolve() != directory
            or not directory.is_relative_to(parent.resolve())
        ):
            raise ValueError("unexpected private directory")
        shutil.rmtree(directory)
    except (OSError, RuntimeError, ValueError) as error:
        raise RetentionError(
            "cleanup_failed", "Retention could not clean its private staging directory."
        ) from error


def _target_path(base: Path, target: RetentionTarget) -> Path:
    parent_name = (
        ALERT_AUDIT_DIRECTORY
        if target.kind == "alert_audit"
        else FINAL_REPORT_DIRECTORY
    )
    path = base / parent_name / target.artifact_id
    try:
        if (
            _is_link(path)
            or not path.is_dir()
            or path.resolve() != path
            or not path.is_relative_to(base.resolve())
        ):
            raise ValueError("invalid deletion target")
        return path
    except (OSError, RuntimeError, ValueError) as error:
        raise RetentionError(
            "inventory_changed",
            "Retention stopped because a verified deletion target changed.",
        ) from error


def _apply_plan(
    paths: Paths,
    plan: RetentionPlan,
    persistence: RetentionAuditPersistencePolicy,
) -> tuple[RetentionAuditRecord, Path, bool]:
    base = _managed_processed(paths, create=True)
    quarantine = Path(mkdtemp(prefix=".retention-delete-", dir=base)).resolve()
    moved: list[tuple[Path, Path]] = []
    audit_saved = False
    try:
        sources = tuple(_target_path(base, target) for target in plan.targets)
        for index, source in enumerate(sources):
            destination = quarantine / f"{index:05d}-{source.name}"
            source.rename(destination)
            moved.append((source, destination))
        record = _audit_record(plan)
        audit_path, reused = _save_retention_audit(paths, record, persistence)
        audit_saved = True
        _remove_private_tree(quarantine, base, ".retention-delete-")
        return record, audit_path, reused
    except Exception as error:
        if not audit_saved:
            for source, destination in reversed(moved):
                if destination.exists() and not source.exists():
                    try:
                        destination.rename(source)
                    except OSError as rollback_error:
                        raise RetentionError(
                            "rollback_failed",
                            "Retention could not restore a bundle after a failed operation.",
                        ) from rollback_error
            if quarantine.exists():
                _remove_private_tree(quarantine, base, ".retention-delete-")
        if isinstance(error, RetentionError):
            raise
        raise RetentionError(
            "delete_failed", "Retention could not complete the verified deletion plan."
        ) from error


def run_retention(
    paths: Paths,
    policy: RetentionSettings,
    *,
    apply: bool = False,
    now: datetime | None = None,
    audit_policy: RetentionAuditPersistencePolicy | None = None,
) -> RetentionRunResult:
    """Plan by default; apply only when both policy and caller explicitly allow it."""

    resolved = RetentionSettings.model_validate(policy.model_dump(mode="python"))
    if apply and not resolved.enabled:
        raise RetentionError(
            "retention_disabled",
            "Deletion requires an enabled retention policy and explicit apply action.",
        )
    plan = plan_retention(paths, resolved, now=now)
    if not apply:
        return RetentionRunResult(
            plan=plan,
            applied=False,
            audit=None,
            audit_path=None,
            audit_reused=False,
        )
    persistence = RetentionAuditPersistencePolicy.model_validate(
        (audit_policy or RetentionAuditPersistencePolicy()).model_dump(mode="python")
    )
    audit, audit_path, reused = _apply_plan(paths, plan, persistence)
    return RetentionRunResult(
        plan=plan,
        applied=True,
        audit=audit,
        audit_path=audit_path,
        audit_reused=reused,
    )


def retention_audit_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public JSON Schema for the B9.2 deletion receipt."""

    return {
        "retention-audit.schema.json": RetentionAuditRecord.model_json_schema()
    }


def write_retention_audit_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the portable retention-audit schema."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in retention_audit_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        exported[filename] = destination
    return exported
