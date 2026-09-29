"""Shared Phase 8 application service for one persisted offline evaluation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from audio_sentinel.acoustic_aggregation import AcousticAggregationSettings
from audio_sentinel.acoustic_loader import load_yamnet
from audio_sentinel.alert_audit import save_alert_audit
from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import ConsentRecord, ConsentStatus, ProcessingScope
from audio_sentinel.evaluator import (
    OfflineClipEvaluator,
    OfflineEvaluatorModels,
    OfflineEvaluatorPolicies,
)
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.preparation import Identifier
from audio_sentinel.transcription import load_transcription_model
from audio_sentinel.vad import load_silero_vad
from audio_sentinel.final_report import save_final_report


class EvaluationServiceError(RuntimeError):
    """Stable application-service failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class EvaluationScope(str, Enum):
    """The two scopes that can authorize an evaluation run."""

    ACOUSTIC_ONLY = ProcessingScope.ACOUSTIC_ONLY.value
    ACOUSTIC_AND_SPEECH = ProcessingScope.ACOUSTIC_AND_SPEECH.value


class EvaluationRequest(BaseModel):
    """Validated, identity-only request shared by local CLI and HTTP callers."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    audio_path: str = Field(min_length=1, max_length=512)
    clip_id: Identifier
    consent_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    processing_scope: EvaluationScope
    device_authorized: StrictBool
    granted_at: datetime
    expires_at: datetime | None = None
    source_dataset: str = Field(min_length=1, max_length=128)
    acoustic_threshold: float = Field(ge=0, le=1, strict=True)

    @field_validator("audio_path")
    @classmethod
    def validate_audio_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("audio_path must use forward slashes")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in ("", ".", "..") for part in value.split("/"))
            or ":" in path.parts[0]
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("audio_path must stay inside data/raw")
        return path.as_posix()

    @field_validator("granted_at", "expires_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("consent timestamps must include a UTC offset")
        return value

    @field_validator("source_dataset")
    @classmethod
    def validate_source_dataset(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ValueError("source_dataset must not contain control characters")
        return value

    @model_validator(mode="after")
    def validate_authorization(self) -> "EvaluationRequest":
        if not self.device_authorized:
            raise ValueError("device_authorized must be true")
        if self.expires_at is not None and self.expires_at <= self.granted_at:
            raise ValueError("expires_at must be later than granted_at")
        return self


class EvaluationResponse(BaseModel):
    """Privacy-minimized result returned by both operator boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    clip_id: Identifier
    processing_scope: EvaluationScope
    outcome: Literal["no_action", "log", "review", "alert"]
    risk_score: float = Field(ge=0, le=100)
    risk_severity: Literal["none", "low", "medium", "high", "critical"]
    review_required: bool
    alert_candidate: bool
    report_id: Identifier
    report_path: str
    report_reused: bool
    audit_id: Identifier
    audit_path: str
    audit_reused: bool
    alert_id: Identifier | None = None
    alert_path: str | None = None
    notification_delivery: Literal["not_sent"] = "not_sent"
    alert_delivery_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_alert_reference(self) -> "EvaluationResponse":
        if (self.alert_id is None) != (self.alert_path is None):
            raise ValueError("alert identity and path must be present together")
        expected_report = f"final-reports/{self.report_id}/report.json"
        expected_audit = f"alert-audit/{self.audit_id}/audit.json"
        expected_alert = (
            None
            if self.alert_id is None
            else f"alert-audit/{self.audit_id}/alert.json"
        )
        if (
            self.report_path != expected_report
            or self.audit_path != expected_audit
            or self.alert_path != expected_alert
        ):
            raise ValueError("output paths must match their artifact identities")
        if self.outcome == "alert":
            if (
                not self.alert_candidate
                or not self.review_required
                or self.alert_id is None
            ):
                raise ValueError("alert outcome requires one reviewable local alert")
        elif self.alert_candidate or self.alert_id is not None:
            raise ValueError("non-alert outcome cannot return a local alert")
        return self


def _relative_processed_path(settings: AudioSentinelSettings, path: Path) -> str:
    try:
        return path.relative_to(settings.paths.processed_data).as_posix()
    except ValueError as error:
        raise EvaluationServiceError(
            "invalid_output_path", "Generated output is outside data/processed."
        ) from error


def run_evaluation(
    settings: AudioSentinelSettings,
    request: EvaluationRequest,
) -> EvaluationResponse:
    """Run, report, and locally audit one validated authorized recording."""

    if not isinstance(settings, AudioSentinelSettings) or not isinstance(
        request, EvaluationRequest
    ):
        raise EvaluationServiceError(
            "invalid_request", "A validated evaluation request and settings are required."
        )
    settings.ensure_directories()
    consent = ConsentRecord(
        consent_id=request.consent_id,
        status=ConsentStatus.GRANTED,
        processing_scope=ProcessingScope(request.processing_scope.value),
        device_authorized=True,
        raw_audio_retention_allowed=False,
        granted_at=request.granted_at,
        expires_at=request.expires_at,
    )
    acoustic = load_yamnet(settings.paths)
    vad = None
    transcription = None
    if request.processing_scope is EvaluationScope.ACOUSTIC_AND_SPEECH:
        vad = load_silero_vad(settings.paths)
        transcription = load_transcription_model(settings.paths)

    evaluator = OfflineClipEvaluator(
        settings=settings,
        source_dataset=request.source_dataset,
        models=OfflineEvaluatorModels(
            acoustic=acoustic,
            vad=vad,
            transcription=transcription,
        ),
        policies=OfflineEvaluatorPolicies(
            acoustic_aggregation=AcousticAggregationSettings.uniform(
                request.acoustic_threshold
            )
        ),
    )
    result = evaluator.evaluate(
        InputAudio(request.clip_id, Path(request.audio_path), consent)
    )
    saved_report = save_final_report(settings.paths, result)
    report_relative = _relative_processed_path(settings, saved_report.report_path)
    saved_audit = save_alert_audit(settings.paths, report_relative)

    return EvaluationResponse(
        clip_id=result.clip_id,
        processing_scope=request.processing_scope,
        outcome=result.decision.outcome.value,
        risk_score=result.risk_assessment.score,
        risk_severity=result.risk_assessment.severity.value,
        review_required=result.decision.review_required,
        alert_candidate=result.decision.alert_candidate,
        report_id=saved_report.report.report_id,
        report_path=report_relative,
        report_reused=saved_report.reused,
        audit_id=saved_audit.audit.audit_id,
        audit_path=_relative_processed_path(settings, saved_audit.audit_path),
        audit_reused=saved_audit.reused,
        alert_id=None if saved_audit.alert is None else saved_audit.alert.alert_id,
        alert_path=(
            None
            if saved_audit.alert_path is None
            else _relative_processed_path(settings, saved_audit.alert_path)
        ),
    )
