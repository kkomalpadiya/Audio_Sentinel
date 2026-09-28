"""B8.2 command-line tests for evaluation and verified report inspection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

import audio_sentinel.cli as cli
from audio_sentinel.contracts import EventLabel
from test_evaluator import (
    FakeAcousticModel,
    FakeTranscriber,
    FakeVadSession,
    loaded_acoustic,
    loaded_transcriber,
    loaded_vad,
)


runner = CliRunner()


def project(tmp_path: Path) -> Path:
    root = tmp_path / "Project_1"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    timeline = np.arange(12_000, dtype=np.float64) / 16_000
    sf.write(
        raw / "sample.wav",
        0.1 * np.sin(2 * np.pi * 220 * timeline),
        16_000,
        subtype="PCM_16",
    )
    return root


def evaluation_args(root: Path, scope: str = "acoustic_only") -> list[str]:
    return [
        "evaluate",
        "--project-root",
        str(root),
        "--audio",
        "sample.wav",
        "--clip-id",
        "cli-clip",
        "--consent-id",
        "consent-cli-001",
        "--scope",
        scope,
        "--granted-at",
        "2020-01-01T00:00:00Z",
        "--device-authorized",
        "--source-dataset",
        "synthetic-cli-test",
        "--acoustic-threshold",
        "0.5",
    ]


def install_fake_models(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "load_yamnet",
        lambda _paths: loaded_acoustic(FakeAcousticModel(label=EventLabel.EXPLOSION)),
    )
    monkeypatch.setattr(
        cli, "load_silero_vad", lambda _paths: loaded_vad(FakeVadSession(0.9))
    )
    monkeypatch.setattr(
        cli,
        "load_transcription_model",
        lambda _paths: loaded_transcriber(FakeTranscriber("I will kill you")),
    )


def test_help_lists_only_supported_commands():
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "evaluate" in result.stdout
    assert "inspect-report" in result.stdout


def test_acoustic_only_evaluation_saves_report_and_audit_without_speech_models(
    tmp_path, monkeypatch
):
    root = project(tmp_path)
    install_fake_models(monkeypatch)
    monkeypatch.setattr(
        cli,
        "load_silero_vad",
        lambda _paths: (_ for _ in ()).throw(AssertionError("VAD must not load")),
    )
    monkeypatch.setattr(
        cli,
        "load_transcription_model",
        lambda _paths: (_ for _ in ()).throw(AssertionError("ASR must not load")),
    )

    result = runner.invoke(cli.app, evaluation_args(root))

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["clip_id"] == "cli-clip"
    assert payload["processing_scope"] == "acoustic_only"
    assert payload["report_id"].startswith("report-")
    assert payload["audit_id"].startswith("audit-")
    assert payload["alert_id"] is None
    assert payload["notification_delivery"] == "not_sent"
    assert payload["alert_delivery_authorized"] is False
    assert (root / "data" / "processed" / payload["report_path"]).is_file()
    assert (root / "data" / "processed" / payload["audit_path"]).is_file()
    assert str(root) not in result.stdout


def test_speech_authorized_evaluation_loads_both_speech_models_and_creates_local_alert(
    tmp_path, monkeypatch
):
    root = project(tmp_path)
    install_fake_models(monkeypatch)

    result = runner.invoke(
        cli.app, evaluation_args(root, "acoustic_and_speech") + ["--compact"]
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["processing_scope"] == "acoustic_and_speech"
    assert payload["outcome"] == "alert"
    assert payload["alert_candidate"] is True
    assert payload["alert_id"].startswith("alert-")
    assert (root / "data" / "processed" / payload["alert_path"]).is_file()
    assert "I will kill you" not in result.stdout
    assert "\n  " not in result.stdout


def test_inspect_report_reloads_and_returns_privacy_minimized_summary(tmp_path, monkeypatch):
    root = project(tmp_path)
    install_fake_models(monkeypatch)
    evaluated = runner.invoke(cli.app, evaluation_args(root, "acoustic_and_speech"))
    report_path = json.loads(evaluated.stdout)["report_path"]

    result = runner.invoke(
        cli.app,
        ["inspect-report", report_path, "--project-root", str(root)],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.0"
    assert payload["document_type"] == "offline_evaluation_report"
    assert payload["source"]["clip_id"] == "cli-clip"
    assert [item["kind"] for item in payload["evidence"]] == [
        "acoustic",
        "speech",
        "language",
    ]
    assert payload["summary"]["notification_delivery"] == "not_sent"
    assert "I will kill you" not in result.stdout
    assert '"text"' not in result.stdout.lower()
    assert ".wav" not in result.stdout
    assert str(root) not in result.stdout


def test_evaluation_requires_explicit_device_authorization_before_loading_models(
    tmp_path, monkeypatch
):
    root = project(tmp_path)
    monkeypatch.setattr(
        cli,
        "load_yamnet",
        lambda _paths: (_ for _ in ()).throw(AssertionError("must not load")),
    )
    args = evaluation_args(root)
    args.remove("--device-authorized")

    result = runner.invoke(cli.app, args)

    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "device_authorization_required"


def test_evaluation_rejects_naive_consent_time_with_safe_json(tmp_path, monkeypatch):
    root = project(tmp_path)
    args = evaluation_args(root)
    args[args.index("2020-01-01T00:00:00Z")] = "2020-01-01T00:00:00"

    result = runner.invoke(cli.app, args)

    assert result.exit_code == 1
    payload = json.loads(result.stderr)
    assert payload["code"] == "invalid_time"
    assert "UTC offset" in payload["error"]


def test_inspection_rejects_absolute_path(tmp_path):
    root = project(tmp_path)

    result = runner.invoke(
        cli.app,
        ["inspect-report", str(root / "report.json"), "--project-root", str(root)],
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "invalid_report_path"


def test_evaluation_rejects_audio_config_outside_project(tmp_path):
    root = project(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        cli.app, evaluation_args(root) + ["--audio-config", str(outside)]
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "invalid_audio_config"


def test_unexpected_errors_do_not_expose_exception_details(tmp_path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.setattr(
        cli,
        "_inspect_report_command",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("private-value")),
    )

    result = runner.invoke(
        cli.app,
        ["inspect-report", "final-reports/x/report.json", "--project-root", str(root)],
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr) == {
        "code": "unexpected_error",
        "error": "The command could not be completed.",
    }
    assert "private-value" not in result.stderr
