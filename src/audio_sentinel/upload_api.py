"""Loopback-only HTTP endpoints for audio upload analysis and its explanation.

Reuses the existing A8.3 API conventions: loopback-only access, the shared
single-evaluation lock, the ``{"code", "error"}`` error body, and the same
code-to-status mapping. Unexpected failures are logged and redacted.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from audio_sentinel import api, upload_analysis
from audio_sentinel.api import ApiError
from audio_sentinel.config import load_settings
from audio_sentinel.demo import WEB_DIRECTORY
from audio_sentinel.evaluation_service import EvaluationScope
from audio_sentinel.risk_explanation import RiskExplanation
from audio_sentinel.upload_analysis import (
    MAX_UPLOAD_BYTES,
    UploadAnalysisError,
    UploadAnalysisResponse,
)


router = APIRouter(tags=["Audio upload analysis"])
_logger = logging.getLogger(__name__)
# Multipart framing overhead allowance on top of the audio byte limit.
_MULTIPART_OVERHEAD = 64 * 1024
_READ_CHUNK = 1024 * 1024

_ERRORS = {
    400: {"model": ApiError},
    403: {"model": ApiError},
    404: {"model": ApiError},
    409: {"model": ApiError},
    413: {"model": ApiError},
    415: {"model": ApiError},
    422: {"model": ApiError},
    500: {"model": ApiError},
    503: {"model": ApiError},
}


@router.get("/analyze", include_in_schema=False)
def analyze_page() -> FileResponse:
    return FileResponse(WEB_DIRECTORY / "analyze.html")


@router.get("/api/v1/uploads/limits")
def get_upload_limits() -> dict[str, object]:
    return upload_analysis.upload_limits()


async def _read_bounded(file: UploadFile) -> bytes:
    data = bytearray()
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            return bytes(data)
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadAnalysisError(
                "upload_too_large",
                f"The file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                413,
            )


def _failure(error: Exception) -> JSONResponse:
    if isinstance(error, UploadAnalysisError):
        return api._error(error.status_code, error.code, str(error))
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return api._error(api._status_for_code(code), code, str(error))
    _logger.exception("Upload analysis failed")
    return api._error(500, "unexpected_error", "The analysis could not be completed.")


@router.post(
    "/api/v1/uploads/analyze",
    response_model=UploadAnalysisResponse,
    responses=_ERRORS,
    status_code=201,
)
async def analyze_uploaded_audio(
    request: Request,
    file: UploadFile = File(...),
    processing_scope: EvaluationScope = Form(EvaluationScope.ACOUSTIC_ONLY),
    consent_confirmed: bool = Form(False),
) -> UploadAnalysisResponse | JSONResponse:
    """Analyse one short WAV/FLAC upload with the existing offline pipeline."""

    if not api._is_loopback(request):
        return api._error(403, "local_access_required", "This endpoint accepts loopback clients only.")
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD:
        return api._error(413, "upload_too_large", "The file exceeds the upload limit.")
    try:
        data = await _read_bounded(file)
    except Exception as error:
        return _failure(error)
    finally:
        await file.close()

    if not api._evaluation_lock.acquire(blocking=False):
        return api._error(409, "evaluation_busy", "Another evaluation is already running.")
    try:
        # Model inference is CPU-bound and synchronous; run it off the event loop.
        from starlette.concurrency import run_in_threadpool

        settings = load_settings()
        return await run_in_threadpool(
            upload_analysis.analyze_upload,
            settings,
            data,
            file.filename,
            processing_scope=processing_scope,
            consent_confirmed=consent_confirmed,
        )
    except Exception as error:
        return _failure(error)
    finally:
        api._evaluation_lock.release()


@router.get(
    "/api/v1/uploads/{upload_id}/explanation",
    response_model=RiskExplanation,
    responses=_ERRORS,
)
def explain_uploaded_audio(upload_id: str, request: Request) -> RiskExplanation | JSONResponse:
    """Answer "Is this audio dangerous?" from the verified saved report."""

    if not api._is_loopback(request):
        return api._error(403, "local_access_required", "This endpoint accepts loopback clients only.")
    try:
        return upload_analysis.explain_upload(load_settings(), upload_id)
    except Exception as error:
        return _failure(error)
