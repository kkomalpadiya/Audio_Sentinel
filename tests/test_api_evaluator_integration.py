"""A8.4 end-to-end tests across the HTTP and offline evaluator boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from audio_sentinel import api, evaluation_service
from audio_sentinel.alert_audit import AlertReviewStatus, load_alert_audit
from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import EventLabel
from audio_sentinel.final_report import load_final_report
from audio_sentinel.main import app
from audio_sentinel.risk_contracts import RiskInputStatus
from test_evaluator import (
    FakeAcousticModel,
    FakeTranscriber,
    FakeVadSession,
    InvalidAcousticModel,
    loaded_acoustic,
    loaded_transcriber,
    loaded_vad,
)


PRIVATE_PHRASE = "I will kill you beside the private marker"


def integration_settings(tmp_path: Path) -> AudioSentinelSettings:
    root = tmp_path / "Project_1"
    settings = AudioSentinelSettings.from_project_root(root)
    settings.ensure_directories()
    timeline = np.arange(12_000, dtype=np.float64) / 16_000
    sf.write(
        settings.paths.raw_data / "sample.wav",
        0.1 * np.sin(2 * np.pi * 220 * timeline),
        16_000,
        subtype="PCM_16",
    )
    return settings


def request_payload(
    *,
    scope: str,
    clip_id: str = "api-integration-clip",
    threshold: float = 0.5,
    audio_path: str = "sample.wav",
) -> dict[str, object]:
    return {
        "audio_path": audio_path,
        "clip_id": clip_id,
        "consent_id": f"consent-{clip_id}",
        "processing_scope": scope,
        "device_authorized": True,
        "granted_at": "2020-01-01T00:00:00Z",
        "source_dataset": "synthetic-api-integration",
        "acoustic_threshold": threshold,
    }


def install_models(
    monkeypatch,
    *,
    acoustic_label: EventLabel | None = EventLabel.EXPLOSION,
    acoustic_score: float = 0.9,
    vad_score: float = 0.9,
    transcript: str = PRIVATE_PHRASE,
    acoustic_model: object | None = None,
) -> tuple[object, FakeVadSession, FakeTranscriber]:
    acoustic = acoustic_model or FakeAcousticModel(
        label=acoustic_label,
        score=acoustic_score,
    )
    vad = FakeVadSession(vad_score)
    transcriber = FakeTranscriber(transcript)
    monkeypatch.setattr(
        evaluation_service,
        "load_yamnet",
        lambda _paths: loaded_acoustic(acoustic),
    )
    monkeypatch.setattr(
        evaluation_service,
        "load_silero_vad",
        lambda _paths: loaded_vad(vad),
    )
    monkeypatch.setattr(
        evaluation_service,
        "load_transcription_model",
        lambda _paths: loaded_transcriber(transcriber),
    )
    return acoustic, vad, transcriber


def post_evaluation(payload: dict[str, object]):
    with TestClient(app, client=("127.0.0.1", 50_000)) as client:
        return client.post("/api/v1/evaluations", json=payload)


def processed_file(settings: AudioSentinelSettings, relative: object) -> Path:
    assert isinstance(relative, str)
    return settings.paths.processed_data / relative


def test_acoustic_only_http_request_runs_real_evaluator_and_skips_speech_models(
    tmp_path, monkeypatch
):
    settings = integration_settings(tmp_path)
    monkeypatch.setattr(api, "load_settings", lambda: settings)
    acoustic, _vad, _transcriber = install_models(monkeypatch)
    monkeypatch.setattr(
        evaluation_service,
        "load_silero_vad",
        lambda _paths: (_ for _ in ()).throw(AssertionError("VAD must not load")),
    )
    monkeypatch.setattr(
        evaluation_service,
        "load_transcription_model",
        lambda _paths: (_ for _ in ()).throw(AssertionError("ASR must not load")),
    )

    response = post_evaluation(request_payload(scope="acoustic_only"))

    assert response.status_code == 201, response.text
    payload = response.json()
    report = load_final_report(settings.paths, payload["report_path"])
    audit = load_alert_audit(settings.paths, payload["audit_path"])
    assert getattr(acoustic, "calls") >= 1
    assert payload["outcome"] == report.summary.outcome.value == "review"
    assert report.source.processing_scope.value == "acoustic_only"
    assert [receipt.input_status for receipt in report.evidence[1:]] == [
        RiskInputStatus.NOT_PERMITTED,
        RiskInputStatus.NOT_PERMITTED,
    ]
    assert audit.alert is None
    assert payload["alert_id"] is payload["alert_path"] is None
    assert payload["notification_delivery"] == "not_sent"
    assert payload["alert_delivery_authorized"] is False


def test_speech_authorized_http_request_persists_one_pending_local_alert(
    tmp_path, monkeypatch
):
    settings = integration_settings(tmp_path)
    monkeypatch.setattr(api, "load_settings", lambda: settings)
    acoustic, vad, transcriber = install_models(monkeypatch)

    response = post_evaluation(request_payload(scope="acoustic_and_speech"))

    assert response.status_code == 201, response.text
    payload = response.json()
    report = load_final_report(settings.paths, payload["report_path"])
    audit = load_alert_audit(settings.paths, payload["audit_path"])
    persisted = "\n".join(
        processed_file(settings, payload[key]).read_text(encoding="utf-8")
        for key in ("report_path", "audit_path", "alert_path")
    )
    assert getattr(acoustic, "calls") >= 1
    assert vad.calls >= 1
    assert transcriber.calls >= 1
    assert payload["outcome"] == report.summary.outcome.value == "alert"
    assert audit.alert is not None
    assert audit.alert.alert_id == payload["alert_id"]
    assert audit.alert.review_status is AlertReviewStatus.PENDING
    assert audit.alert.local_only is True
    assert audit.alert.notification_delivery == "not_sent"
    assert audit.alert.alert_delivery_authorized is False
    assert PRIVATE_PHRASE not in response.text
    assert PRIVATE_PHRASE not in persisted
    assert str(settings.paths.root) not in response.text
    assert str(settings.paths.root) not in persisted


def test_no_detected_speech_completes_through_http_without_transcription_or_alert(
    tmp_path, monkeypatch
):
    settings = integration_settings(tmp_path)
    monkeypatch.setattr(api, "load_settings", lambda: settings)
    _acoustic, vad, transcriber = install_models(
        monkeypatch,
        acoustic_label=None,
        vad_score=0.1,
        transcript=PRIVATE_PHRASE,
    )

    response = post_evaluation(
        request_payload(scope="acoustic_and_speech", clip_id="no-speech-clip")
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    report = load_final_report(settings.paths, payload["report_path"])
    audit = load_alert_audit(settings.paths, payload["audit_path"])
    assert vad.calls >= 1
    assert transcriber.calls == 0
    assert payload["outcome"] == report.summary.outcome.value == "no_action"
    assert report.evidence[2].input_status is RiskInputStatus.NO_ACCEPTED_TEXT
    assert payload["risk_score"] == 0
    assert payload["alert_candidate"] is False
    assert audit.alert is None


def test_http_threshold_reaches_real_acoustic_aggregation_policy(tmp_path, monkeypatch):
    settings = integration_settings(tmp_path)
    monkeypatch.setattr(api, "load_settings", lambda: settings)
    install_models(monkeypatch, acoustic_score=0.9)

    response = post_evaluation(
        request_payload(
            scope="acoustic_only",
            clip_id="high-threshold-clip",
            threshold=0.95,
        )
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    report = load_final_report(settings.paths, payload["report_path"])
    assert payload["outcome"] == report.summary.outcome.value == "no_action"
    assert payload["risk_score"] == 0
    assert report.summary.acoustic_event_count == 0


def test_missing_recording_returns_safe_404_without_final_artifacts(
    tmp_path, monkeypatch
):
    settings = integration_settings(tmp_path)
    monkeypatch.setattr(api, "load_settings", lambda: settings)
    install_models(monkeypatch)

    response = post_evaluation(
        request_payload(
            scope="acoustic_only",
            clip_id="missing-audio-clip",
            audio_path="missing.wav",
        )
    )

    assert response.status_code == 404
    assert response.json()["code"] == "file_not_found"
    assert "missing.wav" not in response.text
    assert str(settings.paths.root) not in response.text
    assert not (settings.paths.processed_data / "final-reports").exists()
    assert not (settings.paths.processed_data / "alert-audit").exists()
    assert not api._evaluation_lock.locked()


def test_real_stage_failure_releases_http_lock_and_allows_clean_retry(
    tmp_path, monkeypatch
):
    settings = integration_settings(tmp_path)
    monkeypatch.setattr(api, "load_settings", lambda: settings)
    install_models(
        monkeypatch,
        acoustic_model=InvalidAcousticModel(label=EventLabel.EXPLOSION),
    )

    failed = post_evaluation(
        request_payload(scope="acoustic_only", clip_id="retry-clip")
    )

    assert failed.status_code == 400
    assert failed.json()["code"] == "invalid_output"
    assert str(settings.paths.root) not in failed.text
    assert PRIVATE_PHRASE not in failed.text
    assert not api._evaluation_lock.locked()
    assert not (settings.paths.processed_data / "final-reports").exists()
    assert not (settings.paths.processed_data / "alert-audit").exists()

    install_models(monkeypatch)
    retried = post_evaluation(
        request_payload(scope="acoustic_only", clip_id="retry-clip")
    )

    assert retried.status_code == 201, retried.text
    payload = retried.json()
    assert processed_file(settings, payload["report_path"]).is_file()
    assert processed_file(settings, payload["audit_path"]).is_file()
    assert not api._evaluation_lock.locked()
    assert PRIVATE_PHRASE not in json.dumps(payload)
