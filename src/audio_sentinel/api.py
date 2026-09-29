"""A8.3 loopback-only HTTP boundary for offline clip evaluation."""

from __future__ import annotations

import ipaddress
import logging
from threading import Lock

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from audio_sentinel import evaluation_service
from audio_sentinel.config import load_settings
from audio_sentinel.evaluation_service import EvaluationRequest, EvaluationResponse


router = APIRouter(prefix="/api/v1", tags=["Offline evaluation"])
_evaluation_lock = Lock()
_logger = logging.getLogger(__name__)


class ApiError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    error: str


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    body = ApiError(code=code, error=message)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _is_loopback(request: Request) -> bool:
    if request.client is None:
        return False
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return False


def _status_for_code(code: str) -> int:
    if code in {"file_not_found", "model_not_found"}:
        return 404 if code == "file_not_found" else 503
    if code in {
        "runtime_missing",
        "runtime_mismatch",
        "artifact_mismatch",
        "invalid_artifact",
        "model_load_failed",
    }:
        return 503
    if code in {"output_conflict", "artifact_changed", "source_changed"}:
        return 409
    if code.startswith("consent_") or code in {
        "device_not_authorized",
        "processing_not_permitted",
    }:
        return 403
    return 400


@router.post(
    "/evaluations",
    response_model=EvaluationResponse,
    responses={
        400: {"model": ApiError},
        403: {"model": ApiError},
        409: {"model": ApiError},
        422: {"model": ApiError},
        500: {"model": ApiError},
        503: {"model": ApiError},
    },
    status_code=201,
)
def evaluate_recording(
    payload: EvaluationRequest,
    request: Request,
) -> EvaluationResponse | JSONResponse:
    """Evaluate one local recording without accepting uploads or external callers."""

    if not _is_loopback(request):
        return _error(403, "local_access_required", "This endpoint accepts loopback clients only.")
    if not _evaluation_lock.acquire(blocking=False):
        return _error(409, "evaluation_busy", "Another evaluation is already running.")
    try:
        settings = load_settings()
        return evaluation_service.run_evaluation(settings, payload)
    except Exception as error:
        code = getattr(error, "code", None)
        if isinstance(code, str) and code:
            return _error(_status_for_code(code), code, str(error))
        _logger.exception("Offline evaluation API failed")
        return _error(500, "unexpected_error", "The evaluation could not be completed.")
    finally:
        _evaluation_lock.release()
