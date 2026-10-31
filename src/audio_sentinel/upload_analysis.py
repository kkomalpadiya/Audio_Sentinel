"""Audio upload → existing offline pipeline → evidence-based risk explanation.

Flow for one upload:

1. Validate the bytes (non-empty, size cap, WAV/FLAC signature, allowed extension).
2. Stage them under ``data/raw/uploads/`` with a server-generated name; the client
   filename is never used as a path.
3. Probe the container (format, channels, duration cap) and measure the RMS level
   so near-silent recordings are flagged as inconclusive rather than "safe".
4. Run the *existing* Phase 8 evaluation service (preparation, YAMNet acoustic
   inference, optional VAD + Whisper + language rules, risk scoring, consensus,
   final report and local audit).
5. Build the deterministic explanation from the verified final report.
6. Always delete the staged upload and the prepared (resampled) audio copy,
   because upload consent records ``raw_audio_retention_allowed=False``.
   Only privacy-minimized JSON evidence (no audio, no transcript text) remains.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import soundfile as sf

from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.evaluation_service import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationScope,
    run_evaluation_with_report,
)
from audio_sentinel.final_report import load_final_report
from audio_sentinel.risk_explanation import RiskExplanation, explain_report


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_SECONDS = 30.0
MIN_UPLOAD_SECONDS = 0.1
ALLOWED_EXTENSIONS = {".wav": "WAV", ".flac": "FLAC"}
UPLOAD_SOURCE_DATASET = "user-upload"
# Same explicit, documented experimental value used by the CLI/MVP demo examples.
DEFAULT_ACOUSTIC_THRESHOLD = 0.5
UPLOAD_SUBDIRECTORY = "uploads"
RECEIPT_SUBDIRECTORY = "upload-receipts"
_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")


class UploadAnalysisError(RuntimeError):
    """Stable, user-safe failure with an HTTP status hint."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class UploadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class UploadInputSummary(UploadRecord):
    display_name: str
    size_bytes: int = Field(gt=0)
    format: str
    sample_rate_hz: int
    channels: int
    duration_seconds: float
    rms_dbfs: float | None
    silent: bool


class UploadAnalysisResponse(UploadRecord):
    upload_id: str
    input: UploadInputSummary
    evaluation: EvaluationResponse
    explanation: RiskExplanation
    upload_retained: bool = False
    limits: dict[str, object]


class UploadReceipt(UploadRecord):
    """Audio-free record linking an upload id to its verified final report."""

    upload_id: str
    report_path: str
    created_at: datetime
    input: UploadInputSummary


@dataclass(frozen=True)
class StagedUpload:
    upload_id: str
    path: Path
    relative_path: str
    summary: UploadInputSummary


def upload_limits() -> dict[str, object]:
    return {
        "max_bytes": MAX_UPLOAD_BYTES,
        "max_duration_seconds": MAX_UPLOAD_SECONDS,
        "min_duration_seconds": MIN_UPLOAD_SECONDS,
        "formats": sorted(ALLOWED_EXTENSIONS.values()),
        "extensions": sorted(ALLOWED_EXTENSIONS),
    }


def safe_display_name(filename: str | None) -> str:
    """Return a harmless label for UI display only (never used as a path)."""

    name = (filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable() and ch not in "<>\"'`")
    return (name.strip() or "upload")[:120]


def _extension(filename: str | None) -> str:
    return os.path.splitext(safe_display_name(filename))[1].lower()


def _check_signature(data: bytes, expected_format: str) -> None:
    if expected_format == "WAV":
        ok = len(data) >= 12 and data[:4] in (b"RIFF", b"RF64") and data[8:12] == b"WAVE"
    else:
        ok = data[:4] == b"fLaC"
    if not ok:
        raise UploadAnalysisError(
            "invalid_audio",
            f"The file does not look like a valid {expected_format} recording.",
        )


def validate_upload_bytes(data: bytes, filename: str | None) -> str:
    """Cheap checks before anything touches disk; returns the container format."""

    if not data:
        raise UploadAnalysisError("empty_upload", "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadAnalysisError(
            "upload_too_large",
            f"The file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
            413,
        )
    extension = _extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadAnalysisError(
            "unsupported_format",
            "Unsupported file type. Upload a .wav or .flac recording.",
            415,
        )
    expected = ALLOWED_EXTENSIONS[extension]
    _check_signature(data, expected)
    return expected


def _probe(path: Path, expected_format: str, silence_floor_dbfs: float) -> tuple[sf._SoundFileInfo, float | None, bool]:
    try:
        info = sf.info(str(path))
    except Exception as error:  # libsndfile raises several error types
        raise UploadAnalysisError(
            "invalid_audio", "The recording is corrupted or cannot be decoded."
        ) from error
    if info.format != expected_format:
        raise UploadAnalysisError(
            "unsupported_format",
            "The file contents do not match its .wav/.flac extension.",
            415,
        )
    if info.frames <= 0 or info.samplerate <= 0:
        raise UploadAnalysisError("empty_audio", "The recording contains no audio frames.")
    duration = info.frames / info.samplerate
    if duration > MAX_UPLOAD_SECONDS:
        raise UploadAnalysisError(
            "audio_too_long",
            f"The recording is {duration:.1f} s long; the limit is {MAX_UPLOAD_SECONDS:g} s.",
            413,
        )
    if duration < MIN_UPLOAD_SECONDS:
        raise UploadAnalysisError(
            "audio_too_short",
            f"The recording is shorter than {MIN_UPLOAD_SECONDS:g} s.",
        )
    try:
        samples, _ = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as error:
        raise UploadAnalysisError(
            "invalid_audio", "The recording is corrupted or cannot be decoded."
        ) from error
    if samples.size == 0:
        raise UploadAnalysisError("empty_audio", "The recording contains no audio frames.")
    if not np.isfinite(samples).all():
        raise UploadAnalysisError("invalid_audio", "The recording contains invalid sample values.")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    rms_dbfs = None if rms <= 0 else round(20 * math.log10(rms), 2)
    silent = rms_dbfs is None or rms_dbfs < silence_floor_dbfs
    return info, rms_dbfs, silent


def stage_upload(
    settings: AudioSentinelSettings,
    data: bytes,
    filename: str | None,
) -> StagedUpload:
    """Validate and write one upload under data/raw/uploads with a server name."""

    expected_format = validate_upload_bytes(data, filename)
    settings.ensure_directories()
    upload_id = uuid.uuid4().hex
    extension = _extension(filename)
    directory = settings.paths.raw_data / UPLOAD_SUBDIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{upload_id}{extension}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        info, rms_dbfs, silent = _probe(
            path, expected_format, settings.audio.silence_floor_dbfs
        )
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    summary = UploadInputSummary(
        display_name=safe_display_name(filename),
        size_bytes=len(data),
        format=info.format,
        sample_rate_hz=int(info.samplerate),
        channels=int(info.channels),
        duration_seconds=round(info.frames / info.samplerate, 3),
        rms_dbfs=rms_dbfs,
        silent=silent,
    )
    return StagedUpload(
        upload_id=upload_id,
        path=path,
        relative_path=f"{UPLOAD_SUBDIRECTORY}/{path.name}",
        summary=summary,
    )


def _remove_prepared_audio(settings: AudioSentinelSettings, clip_id: str) -> None:
    """Delete prepared WAV copies for this upload's clip id (evidence JSON stays)."""

    prepared_root = settings.paths.interim_data / "prepared"
    if not prepared_root.is_dir():
        return
    root = prepared_root.resolve()
    for manifest in prepared_root.glob("*/manifest.json"):
        try:
            owner = json.loads(manifest.read_text(encoding="utf-8"))["clip"]["clip_id"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        directory = manifest.parent.resolve()
        if owner == clip_id and directory.parent == root:
            shutil.rmtree(directory, ignore_errors=True)


def _receipt_path(settings: AudioSentinelSettings, upload_id: str) -> Path:
    return settings.paths.processed_data / RECEIPT_SUBDIRECTORY / f"{upload_id}.json"


def analyze_upload(
    settings: AudioSentinelSettings,
    data: bytes,
    filename: str | None,
    *,
    processing_scope: EvaluationScope,
    consent_confirmed: bool,
    now: datetime | None = None,
) -> UploadAnalysisResponse:
    """Validate, analyse with the existing pipeline, explain, and clean up."""

    if consent_confirmed is not True:
        raise UploadAnalysisError(
            "consent_required",
            "Confirm that you are authorized to analyse this recording.",
            403,
        )
    staged = stage_upload(settings, data, filename)
    clip_id = f"upload-{staged.upload_id}"
    try:
        request = EvaluationRequest(
            audio_path=staged.relative_path,
            clip_id=clip_id,
            consent_id=f"upload-consent-{staged.upload_id}",
            processing_scope=processing_scope,
            device_authorized=True,
            granted_at=now or datetime.now(UTC),
            source_dataset=UPLOAD_SOURCE_DATASET,
            acoustic_threshold=DEFAULT_ACOUSTIC_THRESHOLD,
        )
        evaluation, report = run_evaluation_with_report(settings, request)
        explanation = explain_report(report, silent_input=staged.summary.silent)
        receipt = UploadReceipt(
            upload_id=staged.upload_id,
            report_path=evaluation.report_path,
            created_at=datetime.now(UTC),
            input=staged.summary,
        )
        receipt_path = _receipt_path(settings, staged.upload_id)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    finally:
        staged.path.unlink(missing_ok=True)
        _remove_prepared_audio(settings, clip_id)
    return UploadAnalysisResponse(
        upload_id=staged.upload_id,
        input=staged.summary,
        evaluation=evaluation,
        explanation=explanation,
        upload_retained=staged.path.exists(),
        limits=upload_limits(),
    )


def explain_upload(settings: AudioSentinelSettings, upload_id: str) -> RiskExplanation:
    """Re-answer "Is this audio dangerous?" from the verified saved report."""

    if not isinstance(upload_id, str) or not _UPLOAD_ID.fullmatch(upload_id):
        raise UploadAnalysisError("invalid_upload_id", "Unknown analysis id.", 404)
    path = _receipt_path(settings, upload_id)
    if not path.is_file():
        raise UploadAnalysisError("analysis_not_found", "Unknown analysis id.", 404)
    try:
        receipt = UploadReceipt.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if receipt.upload_id != upload_id:
            raise ValueError("receipt id mismatch")
        report = load_final_report(settings.paths, receipt.report_path)
    except Exception as error:
        raise UploadAnalysisError(
            "report_unavailable",
            "The saved analysis could not be verified, so no explanation is given.",
            409,
        ) from error
    return explain_report(report, silent_input=receipt.input.silent)
