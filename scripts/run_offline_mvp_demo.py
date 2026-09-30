"""Run the reviewed offline MVP demonstration through the public CLI boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence, TextIO


DEMO_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
SCOPES = ("acoustic_only", "acoustic_and_speech")
Runner = Callable[..., subprocess.CompletedProcess[str]]


class DemoError(RuntimeError):
    """Stable demonstration failure code plus a privacy-safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class DemoRequest:
    audio_path: str
    clip_id: str
    consent_id: str
    scope: str
    granted_at: str
    expires_at: str | None
    source_dataset: str
    acoustic_threshold: float
    audio_config: str | None
    retention_config: str


def _parse_offset_time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise DemoError(
            "invalid_time", f"{label} must be an ISO 8601 timestamp with a UTC offset."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DemoError(
            "invalid_time", f"{label} must include a UTC offset."
        )
    return parsed


def _validate_project_relative_file(
    root: Path,
    value: str,
    *,
    base: Path,
    code: str,
    label: str,
) -> str:
    if not value or "\\" in value:
        raise DemoError(code, f"{label} must use a forward-slash relative path.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in value.split("/"))
        or ":" in relative.parts[0]
    ):
        raise DemoError(code, f"{label} must stay inside its approved project directory.")
    try:
        resolved_base = base.resolve(strict=True)
        resolved = (resolved_base / Path(*relative.parts)).resolve(strict=True)
        resolved.relative_to(resolved_base)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise DemoError(code, f"{label} was not found inside its approved project directory.") from error
    if not resolved.is_file():
        raise DemoError(code, f"{label} must identify a regular local file.")
    return relative.as_posix()


def validate_request(
    arguments: argparse.Namespace,
    *,
    project_root: Path = PROJECT_ROOT,
    now: datetime | None = None,
) -> DemoRequest:
    """Validate authorization facts before starting any model runtime."""

    root = project_root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise DemoError("invalid_project_root", "The demonstration must run from Project_1.")
    if not arguments.confirm_consent_reviewed:
        raise DemoError(
            "consent_confirmation_required",
            "Confirm that the external consent record was reviewed before the demonstration.",
        )
    if not arguments.confirm_authorized_device:
        raise DemoError(
            "device_confirmation_required",
            "Confirm that the recording device is covered by the consent record.",
        )
    for value, label in (
        (arguments.clip_id, "clip-id"),
        (arguments.consent_id, "consent-id"),
    ):
        if not IDENTIFIER_PATTERN.fullmatch(value):
            raise DemoError(
                "invalid_identifier",
                f"{label} must be an opaque identifier of 3 to 128 permitted characters.",
            )
    if arguments.scope not in SCOPES:
        raise DemoError("invalid_scope", "scope must be acoustic_only or acoustic_and_speech.")
    if (
        not arguments.source_dataset.strip()
        or len(arguments.source_dataset.strip()) > 128
        or any(ord(character) < 32 for character in arguments.source_dataset)
    ):
        raise DemoError(
            "invalid_source_dataset",
            "source-dataset must be a bounded documented collection label.",
        )
    threshold = float(arguments.acoustic_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise DemoError(
            "invalid_threshold", "acoustic-threshold must be a finite value from 0 through 1."
        )

    current = now if now is not None else datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise DemoError("invalid_time", "The demonstration clock must include a UTC offset.")
    granted = _parse_offset_time(arguments.granted_at, "granted-at")
    expires = (
        None
        if arguments.expires_at is None
        else _parse_offset_time(arguments.expires_at, "expires-at")
    )
    if granted > current:
        raise DemoError("consent_not_yet_active", "Consent has not taken effect yet.")
    if expires is not None and expires <= granted:
        raise DemoError("invalid_time", "expires-at must be later than granted-at.")
    if expires is not None and current >= expires:
        raise DemoError("consent_expired", "Consent has expired.")

    audio = _validate_project_relative_file(
        root,
        arguments.audio,
        base=root / "data" / "raw",
        code="invalid_audio_path",
        label="audio",
    )
    audio_config = None
    if arguments.audio_config is not None:
        audio_config = _validate_project_relative_file(
            root,
            arguments.audio_config,
            base=root,
            code="invalid_audio_config",
            label="audio-config",
        )
    retention_config = _validate_project_relative_file(
        root,
        arguments.retention_config,
        base=root,
        code="invalid_retention_config",
        label="retention-config",
    )

    return DemoRequest(
        audio_path=audio,
        clip_id=arguments.clip_id,
        consent_id=arguments.consent_id,
        scope=arguments.scope,
        granted_at=granted.isoformat(),
        expires_at=None if expires is None else expires.isoformat(),
        source_dataset=arguments.source_dataset.strip(),
        acoustic_threshold=threshold,
        audio_config=audio_config,
        retention_config=retention_config,
    )


def validate_local_runtime(
    request: DemoRequest,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[str, ...]:
    """Check local runtime and model presence without downloading anything."""

    required_modules = ["tensorflow"]
    required_models = ["models/yamnet/1"]
    if request.scope == "acoustic_and_speech":
        required_modules.extend(("onnxruntime", "faster_whisper"))
        required_models.extend(
            ("models/silero-vad/6", "models/faster-whisper-tiny.en/1")
        )
    missing_modules = [
        name for name in required_modules if importlib.util.find_spec(name) is None
    ]
    if missing_modules:
        raise DemoError(
            "runtime_missing",
            "The current Python environment is missing required local model runtimes: "
            + ", ".join(missing_modules)
            + ".",
        )
    missing_models = [
        relative
        for relative in required_models
        if not (project_root / Path(*PurePosixPath(relative).parts)).is_dir()
    ]
    if missing_models:
        raise DemoError(
            "model_not_found",
            "Required local model directories are missing: " + ", ".join(missing_models) + ".",
        )
    return tuple(required_models)


def build_evaluate_command(
    request: DemoRequest,
    *,
    python_executable: str,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "audio_sentinel.cli",
        "evaluate",
        "--project-root",
        str(project_root),
        "--audio",
        request.audio_path,
        "--clip-id",
        request.clip_id,
        "--consent-id",
        request.consent_id,
        "--scope",
        request.scope,
        "--granted-at",
        request.granted_at,
        "--device-authorized",
        "--source-dataset",
        request.source_dataset,
        "--acoustic-threshold",
        str(request.acoustic_threshold),
        "--compact",
    ]
    if request.expires_at is not None:
        command.extend(("--expires-at", request.expires_at))
    if request.audio_config is not None:
        command.extend(("--audio-config", request.audio_config))
    return command


def _child_environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source = str(project_root / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    return environment


def _run_json(
    command: Sequence[str],
    *,
    project_root: Path,
    runner: Runner,
) -> dict[str, Any]:
    result = runner(
        list(command),
        cwd=project_root,
        env=_child_environment(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        try:
            failure = json.loads(result.stderr.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError, TypeError):
            failure = {}
        code = failure.get("code")
        message = failure.get("error")
        if not isinstance(code, str) or not isinstance(message, str):
            code = "command_failed"
            message = "A demonstration command failed without a structured safe error."
        raise DemoError(code, message)
    try:
        document = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise DemoError(
            "invalid_command_output", "A demonstration command returned invalid JSON."
        ) from error
    if not isinstance(document, dict):
        raise DemoError(
            "invalid_command_output", "A demonstration command must return one JSON object."
        )
    return document


def _require_no_delivery(document: Mapping[str, Any], *, location: str) -> None:
    if (
        document.get("notification_delivery") != "not_sent"
        or document.get("alert_delivery_authorized") is not False
    ):
        raise DemoError(
            "delivery_boundary_failed",
            f"{location} did not preserve the permanent no-delivery boundary.",
        )


def execute_demo(
    request: DemoRequest,
    *,
    project_root: Path = PROJECT_ROOT,
    python_executable: str = sys.executable,
    runner: Runner = subprocess.run,
    progress: TextIO = sys.stderr,
) -> dict[str, Any]:
    """Evaluate, revalidate the report, and perform a retention dry run."""

    print("[1/3] Evaluating the authorized local recording...", file=progress)
    evaluation = _run_json(
        build_evaluate_command(
            request,
            python_executable=python_executable,
            project_root=project_root,
        ),
        project_root=project_root,
        runner=runner,
    )
    _require_no_delivery(evaluation, location="Evaluation output")
    if evaluation.get("processing_scope") != request.scope:
        raise DemoError("result_mismatch", "Evaluation scope does not match the request.")
    report_id = evaluation.get("report_id")
    report_path = evaluation.get("report_path")
    if not isinstance(report_id, str) or not isinstance(report_path, str):
        raise DemoError("invalid_command_output", "Evaluation omitted its final-report reference.")

    print("[2/3] Reloading and integrity-checking the final report...", file=progress)
    inspection = _run_json(
        [
            python_executable,
            "-m",
            "audio_sentinel.cli",
            "inspect-report",
            report_path,
            "--project-root",
            str(project_root),
            "--compact",
        ],
        project_root=project_root,
        runner=runner,
    )
    summary = inspection.get("summary")
    if not isinstance(summary, dict):
        raise DemoError("invalid_command_output", "Report inspection omitted its summary.")
    _require_no_delivery(summary, location="Verified report")
    comparisons = {
        "report_id": (inspection.get("report_id"), report_id),
        "outcome": (summary.get("outcome"), evaluation.get("outcome")),
        "risk_score": (summary.get("risk_score"), evaluation.get("risk_score")),
        "risk_severity": (
            summary.get("risk_severity"),
            evaluation.get("risk_severity"),
        ),
        "review_required": (
            summary.get("review_required"),
            evaluation.get("review_required"),
        ),
        "alert_candidate": (
            summary.get("alert_candidate"),
            evaluation.get("alert_candidate"),
        ),
    }
    mismatches = [name for name, pair in comparisons.items() if pair[0] != pair[1]]
    if mismatches:
        raise DemoError(
            "result_mismatch",
            "Verified report fields differ from evaluation output: " + ", ".join(mismatches) + ".",
        )
    if evaluation.get("outcome") == "alert":
        if evaluation.get("review_required") is not True or not isinstance(
            evaluation.get("alert_path"), str
        ):
            raise DemoError(
                "invalid_command_output", "An alert outcome must remain pending local review."
            )
    elif evaluation.get("alert_path") is not None:
        raise DemoError(
            "invalid_command_output", "A non-alert outcome cannot reference a local alert."
        )

    print("[3/3] Planning retention without applying deletion...", file=progress)
    retention = _run_json(
        [
            python_executable,
            "-m",
            "audio_sentinel.cli",
            "retention",
            "--retention-config",
            request.retention_config,
            "--project-root",
            str(project_root),
            "--compact",
        ],
        project_root=project_root,
        runner=runner,
    )
    if (
        retention.get("applied") is not False
        or retention.get("raw_audio_deleted") is not False
        or retention.get("notification_delivery") != "not_sent"
    ):
        raise DemoError(
            "retention_boundary_failed",
            "The demonstration retention step must remain a non-deleting dry run.",
        )

    evidence = inspection.get("evidence")
    evidence_statuses = []
    if isinstance(evidence, list):
        evidence_statuses = [
            {"kind": item.get("kind"), "input_status": item.get("input_status")}
            for item in evidence
            if isinstance(item, dict)
        ]
    policy_review_required = evaluation.get("review_required") is True
    return {
        "demo_version": DEMO_VERSION,
        "status": "complete",
        "processing_scope": request.scope,
        "evaluation": {
            key: evaluation.get(key)
            for key in (
                "clip_id",
                "outcome",
                "risk_score",
                "risk_severity",
                "review_required",
                "alert_candidate",
                "report_id",
                "report_path",
                "report_reused",
                "audit_id",
                "audit_path",
                "audit_reused",
                "alert_id",
                "alert_path",
            )
        },
        "verified_report": {
            "report_id": report_id,
            "decision_id": inspection.get("decision_id"),
            "evidence_statuses": evidence_statuses,
        },
        "human_review": {
            "policy_review_required": policy_review_required,
            "status": (
                "pending_external_manual_workflow"
                if policy_review_required
                else "not_required_by_current_policy"
            ),
        },
        "retention_dry_run": {
            key: retention.get(key)
            for key in (
                "policy_enabled",
                "scanned_report_count",
                "scanned_audit_count",
                "protected_report_count",
                "protected_pending_alert_count",
                "target_count",
                "applied",
                "raw_audio_deleted",
            )
        },
        "safety": {
            "notification_delivery": "not_sent",
            "alert_delivery_authorized": False,
            "external_action_authorized": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Project_1 offline MVP demo: evaluate one authorized recording, "
            "verify its report, and plan retention without deleting anything."
        )
    )
    parser.add_argument("--audio", required=True, help="Forward-slash path under data/raw.")
    parser.add_argument("--clip-id", required=True, help="Opaque clip identifier.")
    parser.add_argument("--consent-id", required=True, help="Opaque external consent reference.")
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--granted-at", required=True, help="Offset-aware ISO 8601 timestamp.")
    parser.add_argument("--expires-at", help="Optional offset-aware ISO 8601 timestamp.")
    parser.add_argument("--source-dataset", required=True, help="Documented collection label.")
    parser.add_argument("--acoustic-threshold", required=True, type=float)
    parser.add_argument("--audio-config", help="Optional path relative to Project_1.")
    parser.add_argument(
        "--retention-config",
        default="configs/retention.example.json",
        help="Project-relative policy used for a dry-run plan only.",
    )
    parser.add_argument("--confirm-consent-reviewed", action="store_true")
    parser.add_argument("--confirm-authorized-device", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs, local runtimes, and model directories without evaluation.",
    )
    parser.add_argument("--compact", action="store_true", help="Emit one-line JSON.")
    return parser


def _emit(document: Mapping[str, Any], *, compact: bool, stream: TextIO) -> None:
    print(
        json.dumps(
            document,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
            sort_keys=True,
            allow_nan=False,
        ),
        file=stream,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        request = validate_request(arguments)
        required_models = validate_local_runtime(request)
        if arguments.preflight_only:
            result: Mapping[str, Any] = {
                "demo_version": DEMO_VERSION,
                "status": "preflight_complete",
                "processing_scope": request.scope,
                "audio_path": request.audio_path,
                "required_models": list(required_models),
                "network_required": False,
                "retention_apply_authorized": False,
            }
        else:
            result = execute_demo(request)
        _emit(result, compact=arguments.compact, stream=sys.stdout)
        return 0
    except DemoError as error:
        _emit(
            {"code": error.code, "error": str(error)},
            compact=True,
            stream=sys.stderr,
        )
        return 1
    except Exception:
        _emit(
            {
                "code": "unexpected_error",
                "error": "The offline MVP demonstration could not be completed.",
            },
            compact=True,
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
