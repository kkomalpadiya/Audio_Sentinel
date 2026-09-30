"""B8.2 privacy-minimized CLI for offline evaluation and report inspection."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Callable

from pydantic import ValidationError
import typer

from audio_sentinel import evaluation_service
from audio_sentinel import retention as retention_service
from audio_sentinel.config import AudioSentinelSettings, load_settings
from audio_sentinel.evaluation_service import EvaluationRequest, EvaluationScope
from audio_sentinel.final_report import FinalReportDocument, load_final_report


app = typer.Typer(
    name="audio-sentinel",
    help="Evaluate authorized local recordings and inspect verified final reports.",
    no_args_is_help=True,
    add_completion=False,
)


class CliInputError(ValueError):
    """A stable, privacy-safe CLI input failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _parse_timestamp(value: str, option_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise CliInputError(
            "invalid_time", f"{option_name} must be an ISO 8601 timestamp with a UTC offset."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CliInputError(
            "invalid_time", f"{option_name} must include a UTC offset."
        )
    return parsed


def _project_config(
    root: Path,
    config: Path | None,
    *,
    code: str,
    label: str,
) -> Path | None:
    if config is None:
        return None
    try:
        if config.is_absolute():
            raise ValueError("absolute path")
        resolved = (root / config).resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_file():
            raise ValueError("not a file")
        return resolved
    except (OSError, RuntimeError, ValueError) as error:
        raise CliInputError(
            code, f"{label} must be a file inside the project root."
        ) from error


def _load_cli_settings(
    project_root: Path,
    audio_config: Path | None,
    retention_config: Path | None = None,
) -> AudioSentinelSettings:
    root = project_root.expanduser().resolve()
    if not root.is_dir() or not (root / "pyproject.toml").is_file():
        raise CliInputError(
            "invalid_project_root",
            "Project root must be a directory containing pyproject.toml.",
        )
    resolved_audio_config = _project_config(
        root,
        audio_config,
        code="invalid_audio_config",
        label="Audio config",
    )
    resolved_retention_config = _project_config(
        root,
        retention_config,
        code="invalid_retention_config",
        label="Retention config",
    )
    settings = load_settings(
        root,
        audio_config_path=resolved_audio_config,
        retention_config_path=resolved_retention_config,
    )
    settings.ensure_directories()
    return settings


def _report_summary(report: FinalReportDocument) -> dict[str, object]:
    """Return a useful report view without transcript text, audio, or local paths."""

    return {
        "schema_version": report.schema_version,
        "document_type": report.document_type,
        "format_version": report.format_version,
        "report_id": report.report_id,
        "created_at": report.created_at.isoformat(),
        "source": report.source.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in report.evidence],
        "assessment_id": report.risk_assessment.assessment_id,
        "agreement_id": report.agreement.evaluation_id,
        "decision_id": report.decision.decision_id,
        "summary": report.summary.model_dump(mode="json"),
    }


def _evaluate_command(
    *,
    project_root: Path,
    audio: Path,
    clip_id: str,
    consent_id: str,
    scope: EvaluationScope,
    granted_at: str,
    expires_at: str | None,
    device_authorized: bool,
    source_dataset: str,
    acoustic_threshold: float,
    audio_config: Path | None,
) -> dict[str, object]:
    settings = _load_cli_settings(project_root, audio_config)
    if audio.is_absolute():
        raise CliInputError(
            "invalid_audio_path", "Audio must be a path relative to data/raw."
        )
    if not device_authorized:
        raise CliInputError(
            "device_authorization_required",
            "Evaluation requires explicit confirmation that the recording device is authorized.",
        )
    request = EvaluationRequest(
        audio_path=audio.as_posix(),
        clip_id=clip_id,
        consent_id=consent_id,
        processing_scope=scope,
        device_authorized=True,
        granted_at=_parse_timestamp(granted_at, "granted-at"),
        expires_at=(
            None if expires_at is None else _parse_timestamp(expires_at, "expires-at")
        ),
        source_dataset=source_dataset,
        acoustic_threshold=acoustic_threshold,
    )
    return evaluation_service.run_evaluation(settings, request).model_dump(mode="json")


def _inspect_report_command(
    *, project_root: Path, report: Path
) -> dict[str, object]:
    settings = _load_cli_settings(project_root, None)
    if report.is_absolute():
        raise CliInputError(
            "invalid_report_path",
            "Report must be a path relative to data/processed.",
        )
    return _report_summary(load_final_report(settings.paths, report))


def _retention_command(
    *,
    project_root: Path,
    retention_config: Path,
    apply: bool,
) -> dict[str, object]:
    settings = _load_cli_settings(project_root, None, retention_config)
    result = retention_service.run_retention(
        settings.paths,
        settings.retention,
        apply=apply,
    )
    relative_audit = None
    if result.audit_path is not None:
        relative_audit = result.audit_path.relative_to(
            settings.paths.processed_data
        ).as_posix()
    return {
        "applied": result.applied,
        "policy_enabled": result.plan.policy.enabled,
        "scanned_report_count": result.plan.scanned_report_count,
        "scanned_audit_count": result.plan.scanned_audit_count,
        "protected_report_count": result.plan.protected_report_count,
        "protected_pending_alert_count": (
            result.plan.protected_pending_alert_count
        ),
        "target_count": len(result.plan.targets),
        "targets": [item.model_dump(mode="json") for item in result.plan.targets],
        "audit_id": None if result.audit is None else result.audit.audit_id,
        "audit_path": relative_audit,
        "audit_reused": result.audit_reused,
        "raw_audio_deleted": False,
        "notification_delivery": "not_sent",
    }


def _emit_json(document: dict[str, object], pretty: bool) -> None:
    typer.echo(
        json.dumps(
            document,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
    )


def _execute(action: Callable[[], dict[str, object]], pretty: bool) -> None:
    try:
        _emit_json(action(), pretty)
    except typer.Exit:
        raise
    except Exception as error:
        code = getattr(error, "code", None)
        if isinstance(code, str) and code:
            message = str(error)
        elif isinstance(error, ValidationError):
            code = "invalid_input"
            message = "Command input failed validation."
        elif isinstance(error, (OSError, UnicodeError)):
            code = "io_error"
            message = "A required local file could not be read or written."
        else:
            code = "unexpected_error"
            message = "The command could not be completed."
        typer.echo(json.dumps({"code": code, "error": message}, sort_keys=True), err=True)
        raise typer.Exit(code=1) from error


@app.command("evaluate")
def evaluate(
    audio: Path = typer.Option(
        ...,
        "--audio",
        help="Recording path relative to data/raw.",
    ),
    clip_id: str = typer.Option(..., "--clip-id", help="Opaque clip identifier."),
    consent_id: str = typer.Option(
        ..., "--consent-id", help="Opaque consent-record identifier."
    ),
    scope: EvaluationScope = typer.Option(
        ..., "--scope", case_sensitive=False, help="Authorized processing scope."
    ),
    granted_at: str = typer.Option(
        ..., "--granted-at", help="Consent start time as offset-aware ISO 8601."
    ),
    device_authorized: bool = typer.Option(
        False,
        "--device-authorized",
        help="Confirm that the recording device is covered by the consent record.",
    ),
    source_dataset: str = typer.Option(
        ..., "--source-dataset", help="Documented source or collection name."
    ),
    acoustic_threshold: float = typer.Option(
        ...,
        "--acoustic-threshold",
        min=0.0,
        max=1.0,
        help="Explicit experimental mapped-label threshold; this is not calibrated.",
    ),
    expires_at: str | None = typer.Option(
        None, "--expires-at", help="Optional consent expiry as offset-aware ISO 8601."
    ),
    audio_config: Path | None = typer.Option(
        None, "--audio-config", help="Optional audio settings JSON, relative to project root."
    ),
    project_root: Path = typer.Option(
        Path("."), "--project-root", help="Project 1 repository root."
    ),
    pretty: bool = typer.Option(True, "--pretty/--compact", help="JSON formatting."),
) -> None:
    """Evaluate one authorized recording and save its final report and local audit."""

    _execute(
        lambda: _evaluate_command(
            project_root=project_root,
            audio=audio,
            clip_id=clip_id,
            consent_id=consent_id,
            scope=scope,
            granted_at=granted_at,
            expires_at=expires_at,
            device_authorized=device_authorized,
            source_dataset=source_dataset,
            acoustic_threshold=acoustic_threshold,
            audio_config=audio_config,
        ),
        pretty,
    )


@app.command("inspect-report")
def inspect_report(
    report: Path = typer.Argument(
        ...,
        help="Final report path relative to data/processed.",
    ),
    project_root: Path = typer.Option(
        Path("."), "--project-root", help="Project 1 repository root."
    ),
    pretty: bool = typer.Option(True, "--pretty/--compact", help="JSON formatting."),
) -> None:
    """Reload, integrity-check, and summarize one final report."""

    _execute(
        lambda: _inspect_report_command(project_root=project_root, report=report),
        pretty,
    )


@app.command("retention")
def retention(
    retention_config: Path = typer.Option(
        ...,
        "--retention-config",
        help="Retention settings JSON, relative to the project root.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Apply the verified plan; omission performs a non-writing dry run.",
    ),
    project_root: Path = typer.Option(
        Path("."), "--project-root", help="Project 1 repository root."
    ),
    pretty: bool = typer.Option(True, "--pretty/--compact", help="JSON formatting."),
) -> None:
    """Plan or apply bounded deletion of expired local report and audit bundles."""

    _execute(
        lambda: _retention_command(
            project_root=project_root,
            retention_config=retention_config,
            apply=apply,
        ),
        pretty,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
