"""A8.3 tests for the loopback-only evaluation HTTP boundary."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from audio_sentinel import api
from audio_sentinel.evaluation_service import (
    EvaluationRequest,
    EvaluationResponse,
)
from audio_sentinel.main import app


def valid_payload() -> dict[str, object]:
    return {
        "audio_path": "authorized/sample.wav",
        "clip_id": "api-clip-001",
        "consent_id": "consent-api-001",
        "processing_scope": "acoustic_only",
        "device_authorized": True,
        "granted_at": "2026-09-01T09:00:00+05:30",
        "source_dataset": "authorized-local-test",
        "acoustic_threshold": 0.5,
    }


def successful_response() -> EvaluationResponse:
    return EvaluationResponse(
        clip_id="api-clip-001",
        processing_scope="acoustic_only",
        outcome="review",
        risk_score=62.5,
        risk_severity="high",
        review_required=True,
        alert_candidate=False,
        report_id="report-" + "a" * 64,
        report_path="final-reports/report-" + "a" * 64 + "/report.json",
        report_reused=False,
        audit_id="audit-" + "b" * 64,
        audit_path="alert-audit/audit-" + "b" * 64 + "/audit.json",
        audit_reused=False,
    )


@pytest.fixture
def local_client():
    with TestClient(app, client=("127.0.0.1", 50_000)) as session:
        yield session


def test_local_evaluation_returns_privacy_minimized_created_response(
    local_client, monkeypatch
):
    expected = successful_response()
    captured = []

    def run(settings, request):
        captured.append((settings, request))
        return expected

    monkeypatch.setattr(api.evaluation_service, "run_evaluation", run)

    response = local_client.post("/api/v1/evaluations", json=valid_payload())

    assert response.status_code == 201
    assert response.json() == expected.model_dump(mode="json")
    assert len(captured) == 1
    assert captured[0][1].audio_path == "authorized/sample.wav"
    assert captured[0][1].granted_at.utcoffset() is not None
    assert "transcript" not in response.text.lower()
    assert "Project_1" not in response.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("audio_path", "../private.wav"),
        ("audio_path", "C:/private.wav"),
        ("audio_path", "folder\\sample.wav"),
        ("audio_path", "folder//sample.wav"),
        ("processing_scope", "none"),
        ("device_authorized", False),
        ("granted_at", "2026-09-01T09:00:00"),
        ("acoustic_threshold", "0.5"),
        ("acoustic_threshold", 1.1),
    ],
)
def test_invalid_request_values_are_rejected_without_echoing_input(
    local_client, field, value
):
    payload = valid_payload()
    payload[field] = value

    response = local_client.post("/api/v1/evaluations", json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "code": "invalid_request",
        "error": "Request body failed validation.",
    }
    assert str(value) not in response.text


def test_extra_fields_are_rejected_without_exposing_their_values(local_client):
    payload = valid_payload() | {"transcript": "private phrase"}

    response = local_client.post("/api/v1/evaluations", json=payload)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert "private phrase" not in response.text


def test_expiry_must_follow_grant(local_client):
    payload = valid_payload() | {"expires_at": "2026-09-01T08:59:59+05:30"}

    response = local_client.post("/api/v1/evaluations", json=payload)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_non_loopback_client_is_rejected_before_evaluation(monkeypatch):
    monkeypatch.setattr(
        api.evaluation_service,
        "run_evaluation",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    with TestClient(app, client=("192.0.2.10", 50_000)) as client:
        response = client.post("/api/v1/evaluations", json=valid_payload())

    assert response.status_code == 403
    assert response.json()["code"] == "local_access_required"


def test_concurrent_evaluation_is_rejected_without_releasing_active_lock(local_client):
    with api._evaluation_lock:
        response = local_client.post("/api/v1/evaluations", json=valid_payload())
        assert api._evaluation_lock.locked()

    assert response.status_code == 409
    assert response.json()["code"] == "evaluation_busy"


class TypedFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("file_not_found", 404),
        ("consent_expired", 403),
        ("output_conflict", 409),
        ("invalid_audio", 400),
        ("model_not_found", 503),
        ("runtime_missing", 503),
    ],
)
def test_typed_pipeline_errors_keep_safe_code_and_status(
    local_client, monkeypatch, code, status
):
    monkeypatch.setattr(
        api.evaluation_service,
        "run_evaluation",
        lambda *_args: (_ for _ in ()).throw(TypedFailure(code, "Safe failure.")),
    )

    response = local_client.post("/api/v1/evaluations", json=valid_payload())

    assert response.status_code == status
    assert response.json() == {"code": code, "error": "Safe failure."}
    assert not api._evaluation_lock.locked()


def test_unexpected_failure_is_redacted_and_lock_is_released(local_client, monkeypatch):
    monkeypatch.setattr(
        api.evaluation_service,
        "run_evaluation",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("private path detail")),
    )

    response = local_client.post("/api/v1/evaluations", json=valid_payload())

    assert response.status_code == 500
    assert response.json() == {
        "code": "unexpected_error",
        "error": "The evaluation could not be completed.",
    }
    assert "private path detail" not in response.text
    assert not api._evaluation_lock.locked()


def test_request_contract_is_frozen_and_uses_timezone_aware_times():
    request = EvaluationRequest(
        **(
            valid_payload()
            | {"granted_at": datetime(2026, 9, 1, tzinfo=UTC)}
        )
    )

    with pytest.raises(ValidationError):
        request.clip_id = "changed"  # type: ignore[misc]


def test_openapi_exposes_only_authorized_scopes_and_structured_errors(local_client):
    document = local_client.get("/openapi.json").json()
    operation = document["paths"]["/api/v1/evaluations"]["post"]
    scope_schema = document["components"]["schemas"]["EvaluationScope"]

    assert scope_schema["enum"] == ["acoustic_only", "acoustic_and_speech"]
    assert operation["responses"]["201"]
    for status in ("400", "403", "409", "422", "500", "503"):
        assert operation["responses"][status]
