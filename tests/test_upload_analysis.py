"""Audio upload, risk analysis, and explanation tests.

These tests run the real preparation, inference orchestration, aggregation,
evidence, speech, language, scoring, consensus, final-report, and audit code. Only
the three model boundaries (YAMNet, Silero VAD, Whisper) are replaced with the
deterministic fakes already used by the evaluator test suite, so these are
integration tests with mocked models, not real-model end-to-end runs.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from audio_sentinel import api, evaluation_service, upload_analysis, upload_api
from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import EventLabel
from audio_sentinel.final_report import _risk_assessment_identity, load_final_report
from audio_sentinel.main import app
from audio_sentinel.risk_contracts import RiskInputSet
from audio_sentinel.risk_scoring import score_risk
from test_evaluator import (
    FakeAcousticModel,
    FakeTranscriber,
    FakeVadSession,
    InvalidAcousticModel,
    loaded_acoustic,
    loaded_transcriber,
    loaded_vad,
)


CAPPING_PHRASE = "help me please he has a gun"
LOOPBACK = ("127.0.0.1", 50_000)


# ----------------------------------------------------------------- helpers --
def wav_bytes(
    seconds: float = 1.0,
    *,
    rate: int = 16_000,
    amplitude: float = 0.2,
    channels: int = 1,
    fmt: str = "WAV",
) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    tone = amplitude * np.sin(2 * np.pi * 440 * t)
    data = np.repeat(tone[:, None], channels, axis=1) if channels > 1 else tone
    buffer = io.BytesIO()
    subtype = "PCM_16"
    sf.write(buffer, data, rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch) -> AudioSentinelSettings:
    value = AudioSentinelSettings.from_project_root(tmp_path / "Project_1")
    value.ensure_directories()
    monkeypatch.setattr(upload_api, "load_settings", lambda: value)
    return value


def install_models(
    monkeypatch,
    *,
    label: EventLabel | None = EventLabel.GUNSHOT,
    score: float = 0.9,
    transcript: str = "hello there",
    acoustic_model: object | None = None,
):
    acoustic = acoustic_model or FakeAcousticModel(label=label, score=score)
    monkeypatch.setattr(evaluation_service, "load_yamnet", lambda _p: loaded_acoustic(acoustic))
    monkeypatch.setattr(evaluation_service, "load_silero_vad", lambda _p: loaded_vad(FakeVadSession(0.9)))
    monkeypatch.setattr(
        evaluation_service,
        "load_transcription_model",
        lambda _p: loaded_transcriber(FakeTranscriber(transcript)),
    )
    return acoustic


def post(
    data: bytes,
    filename: str = "clip.wav",
    *,
    scope: str = "acoustic_only",
    consent: str = "true",
    client_address=LOOPBACK,
):
    with TestClient(app, client=client_address) as client:
        return client.post(
            "/api/v1/uploads/analyze",
            files={"file": (filename, data, "application/octet-stream")},
            data={"processing_scope": scope, "consent_confirmed": consent},
        )


def get_explanation(upload_id: str):
    with TestClient(app, client=LOOPBACK) as client:
        return client.get(f"/api/v1/uploads/{upload_id}/explanation")


def audio_files(settings: AudioSentinelSettings) -> list[Path]:
    roots = (settings.paths.raw_data, settings.paths.interim_data)
    return [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
    ]


def assert_no_retained_audio(settings: AudioSentinelSettings) -> None:
    assert audio_files(settings) == []


# ---------------------------------------------------- successful analysis --
def test_valid_wav_runs_pipeline_and_returns_real_explained_score(settings, monkeypatch):
    acoustic = install_models(monkeypatch, label=EventLabel.GUNSHOT, score=0.9)

    response = post(wav_bytes(2.0), "../../evil name.wav")

    assert response.status_code == 201, response.text
    body = response.json()
    assert acoustic.calls >= 1
    report = load_final_report(settings.paths, body["evaluation"]["report_path"])
    ex = body["explanation"]
    # The displayed score/level/outcome come straight from the verified report.
    assert ex["risk_score"] == body["evaluation"]["risk_score"] == report.risk_assessment.score
    assert ex["risk_level"] == report.risk_assessment.severity.value == "high"
    assert ex["outcome"] == report.decision.outcome.value == "review"
    assert ex["human_review_recommended"] is True
    assert ex["verdict"] == "possible_risk_review_recommended"
    # Gunshot rule = 45 points + 10 high-confidence bonus (peak 0.9 >= 0.85).
    assert ex["breakdown"]["reproduced"] is True
    branches = {c["branch"]: c for c in ex["contributions"]}
    assert branches["acoustic"]["points"] == 55.0
    assert branches["speech"]["input_status"] == "not_permitted"
    assert ex["acoustic_events"][0]["label"] == "gunshot"
    assert ex["acoustic_events"][0]["high_confidence"] is True
    assert ex["model_confidence"] is None
    assert "not a determination" in " ".join(ex["limitations"])
    # Filename is display-only and sanitized; nothing path-like reaches disk.
    assert body["input"]["display_name"] == "evil name.wav"
    assert body["input"]["duration_seconds"] == pytest.approx(2.0)
    assert body["upload_retained"] is False
    assert_no_retained_audio(settings)
    assert str(settings.paths.root) not in response.text


def test_explanation_endpoint_answers_from_saved_verified_report(settings, monkeypatch):
    install_models(monkeypatch)
    first = post(wav_bytes()).json()

    asked = get_explanation(first["upload_id"])

    assert asked.status_code == 200, asked.text
    assert asked.json() == first["explanation"]
    assert asked.json()["question"] == "Is this audio dangerous?"


def test_speech_scope_capped_score_produces_alert_candidate_without_transcript(settings, monkeypatch):
    """Regression: a score capped at 100 used to crash final-report validation."""

    install_models(monkeypatch, label=EventLabel.GUNSHOT, score=0.9, transcript=CAPPING_PHRASE)

    response = post(wav_bytes(1.0), scope="acoustic_and_speech")

    assert response.status_code == 201, response.text
    ex = response.json()["explanation"]
    assert ex["risk_score"] == 100.0
    assert ex["risk_level"] == "critical"
    assert ex["outcome"] == "alert"
    assert ex["alert_candidate"] is True
    assert ex["verdict"] == "elevated_risk_alert_candidate"
    assert ex["breakdown"]["total_before_cap"] > 100
    assert {f["category"] for f in ex["language_findings"]} == {"distress", "weapon_reference"}
    assert "not a confirmed emergency" in ex["answer"]
    assert ex["notification_delivery"] == "not_sent"
    assert CAPPING_PHRASE not in response.text
    persisted = "".join(
        p.read_text(encoding="utf-8") for p in settings.paths.processed_data.rglob("*.json")
    )
    assert CAPPING_PHRASE not in persisted
    assert_no_retained_audio(settings)


def test_capped_score_is_float_so_assessment_identity_is_stable(settings, monkeypatch):
    install_models(monkeypatch, transcript=CAPPING_PHRASE)
    body = post(wav_bytes(), scope="acoustic_and_speech").json()
    report = load_final_report(settings.paths, body["evaluation"]["report_path"])
    rescored = score_risk(
        RiskInputSet.model_validate(report.risk_assessment.inputs.model_dump()),
        now=report.risk_assessment.created_at,
    )
    assert isinstance(rescored.score, float)
    assert rescored.assessment_id == _risk_assessment_identity(rescored)


def test_flac_and_stereo_upload_are_supported(settings, monkeypatch):
    install_models(monkeypatch, label=None)

    response = post(wav_bytes(1.0, rate=44_100, channels=2, fmt="FLAC"), "clip.FLAC")

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["input"]["format"] == "FLAC"
    assert body["input"]["channels"] == 2
    assert body["explanation"]["risk_score"] == 0
    assert body["explanation"]["verdict"] == "no_indicators_detected"
    assert "does not prove" in body["explanation"]["answer"]


def test_silent_recording_is_reported_as_inconclusive(settings, monkeypatch):
    install_models(monkeypatch, label=None)

    response = post(wav_bytes(1.0, amplitude=0.0))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["input"]["silent"] is True
    assert body["explanation"]["verdict"] == "inconclusive"
    assert body["explanation"]["human_review_recommended"] is True
    assert "silent" in body["explanation"]["answer"]


def test_second_upload_after_completed_analysis_is_independent(settings, monkeypatch):
    install_models(monkeypatch)
    first = post(wav_bytes(1.0))
    second = post(wav_bytes(1.5, amplitude=0.3))

    assert first.status_code == second.status_code == 201
    assert first.json()["upload_id"] != second.json()["upload_id"]
    assert first.json()["evaluation"]["report_id"] != second.json()["evaluation"]["report_id"]
    assert not api._evaluation_lock.locked()
    assert_no_retained_audio(settings)


# ------------------------------------------------------- input validation --
@pytest.mark.parametrize(
    ("data", "filename", "status", "code"),
    [
        (b"", "clip.wav", 400, "empty_upload"),
        (b"ID3\x03\x00\x00\x00" + b"\x00" * 200, "clip.mp3", 415, "unsupported_format"),
        (b"plain text, not audio", "notes.txt", 415, "unsupported_format"),
        (b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\xff" * 64, "broken.wav", 400, "invalid_audio"),
        (b"just some bytes pretending" * 10, "fake.wav", 400, "invalid_audio"),
    ],
    ids=["empty", "mp3", "text", "corrupted-wav", "no-wav-signature"],
)
def test_invalid_uploads_return_clear_errors_and_store_nothing(
    settings, monkeypatch, data, filename, status, code
):
    acoustic = install_models(monkeypatch)

    response = post(data, filename)

    assert response.status_code == status, response.text
    assert response.json()["code"] == code
    assert response.json()["error"]
    assert acoustic.calls == 0
    assert_no_retained_audio(settings)
    assert not (settings.paths.processed_data / "final-reports").exists()


def test_extension_content_mismatch_is_rejected(settings, monkeypatch):
    install_models(monkeypatch)
    response = post(wav_bytes(fmt="FLAC"), "looks-like.wav")
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_audio"


def test_file_over_size_limit_is_rejected(settings, monkeypatch):
    install_models(monkeypatch)
    oversized = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * upload_analysis.MAX_UPLOAD_BYTES
    response = post(oversized)
    assert response.status_code == 413
    assert response.json()["code"] == "upload_too_large"
    assert_no_retained_audio(settings)


def test_recording_over_duration_limit_is_rejected(settings, monkeypatch):
    acoustic = install_models(monkeypatch)
    response = post(wav_bytes(upload_analysis.MAX_UPLOAD_SECONDS + 1, rate=8_000))
    assert response.status_code == 413
    assert response.json()["code"] == "audio_too_long"
    assert acoustic.calls == 0
    assert_no_retained_audio(settings)


def test_consent_is_required(settings, monkeypatch):
    acoustic = install_models(monkeypatch)
    response = post(wav_bytes(), consent="false")
    assert response.status_code == 403
    assert response.json()["code"] == "consent_required"
    assert acoustic.calls == 0
    assert_no_retained_audio(settings)


def test_invalid_scope_is_rejected_by_request_validation(settings, monkeypatch):
    install_models(monkeypatch)
    response = post(wav_bytes(), scope="everything")
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_non_loopback_clients_are_rejected(settings, monkeypatch):
    acoustic = install_models(monkeypatch)
    response = post(wav_bytes(), client_address=("203.0.113.10", 50_000))
    assert response.status_code == 403
    assert response.json()["code"] == "local_access_required"
    assert acoustic.calls == 0
    with TestClient(app, client=("203.0.113.10", 50_000)) as client:
        assert client.get("/api/v1/uploads/" + "a" * 32 + "/explanation").status_code == 403


# ------------------------------------------------------ failure handling --
def test_model_inference_failure_is_safe_and_retryable(settings, monkeypatch):
    install_models(monkeypatch, acoustic_model=InvalidAcousticModel(label=EventLabel.GUNSHOT))

    failed = post(wav_bytes())

    assert failed.status_code == 400
    assert failed.json()["code"] == "invalid_output"
    assert "Traceback" not in failed.text
    assert str(settings.paths.root) not in failed.text
    assert not api._evaluation_lock.locked()
    assert_no_retained_audio(settings)
    assert not (settings.paths.processed_data / "upload-receipts").exists()

    install_models(monkeypatch)
    retried = post(wav_bytes())
    assert retried.status_code == 201, retried.text


def test_unexpected_failure_is_redacted(settings, monkeypatch):
    install_models(monkeypatch)

    def explode(*_args, **_kwargs):
        raise RuntimeError("secret internal detail /home/user/private")

    monkeypatch.setattr(upload_analysis, "run_evaluation_with_report", explode)
    response = post(wav_bytes())
    assert response.status_code == 500
    assert response.json() == {"code": "unexpected_error", "error": "The analysis could not be completed."}
    assert not api._evaluation_lock.locked()
    assert_no_retained_audio(settings)


def test_busy_evaluation_lock_returns_conflict(settings, monkeypatch):
    install_models(monkeypatch)
    assert api._evaluation_lock.acquire(blocking=False)
    try:
        response = post(wav_bytes())
    finally:
        api._evaluation_lock.release()
    assert response.status_code == 409
    assert response.json()["code"] == "evaluation_busy"
    assert_no_retained_audio(settings)


@pytest.mark.parametrize("upload_id", ["not-a-valid-id", "0" * 32])
def test_unknown_explanation_ids_return_not_found(settings, upload_id):
    response = get_explanation(upload_id)
    assert response.status_code == 404


def test_tampered_report_yields_no_explanation(settings, monkeypatch):
    install_models(monkeypatch)
    body = post(wav_bytes()).json()
    report_path = settings.paths.processed_data / body["evaluation"]["report_path"]
    document = json.loads(report_path.read_text(encoding="utf-8"))
    document["risk_assessment"]["score"] = 0.0
    report_path.write_text(json.dumps(document), encoding="utf-8")

    response = get_explanation(body["upload_id"])

    assert response.status_code == 409
    assert response.json()["code"] == "report_unavailable"


# ----------------------------------------------------------------- page ---
def test_analyzer_page_and_assets_are_served(settings):
    with TestClient(app, client=LOOPBACK) as client:
        page = client.get("/analyze")
        assert page.status_code == 200
        assert "Analyze Audio" in page.text
        assert "Is this audio dangerous?" in page.text
        for asset in ("analyze.js", "analyze.css", "style.css"):
            assert client.get(f"/demo/assets/{asset}").status_code == 200
        limits = client.get("/api/v1/uploads/limits").json()
        assert limits["formats"] == ["FLAC", "WAV"]
        assert limits["max_duration_seconds"] == upload_analysis.MAX_UPLOAD_SECONDS


def test_safe_display_name_strips_paths_and_markup():
    assert upload_analysis.safe_display_name("C:\\x\\..\\<b>a.wav") == "ba.wav"
    assert upload_analysis.safe_display_name(None) == "upload"
    assert len(upload_analysis.safe_display_name("a" * 500 + ".wav")) == 120
