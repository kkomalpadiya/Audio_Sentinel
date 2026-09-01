"""Versioned contracts shared by offline data preparation and later pipeline stages."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0"


class EventLabel(str, Enum):
    """The deliberately small, stable event taxonomy for the first release."""

    AMBIENT = "ambient"
    NO_SPEECH = "no_speech"
    SPEECH_PRESENT = "speech_present"
    NON_THREATENING_SPEECH = "non_threatening_speech"
    SIREN = "siren"
    SMOKE_ALARM = "smoke_alarm"
    GLASS_BREAK = "glass_break"
    CROWD_PANIC = "crowd_panic"
    DISTRESS_SPEECH = "distress_speech"
    THREATENING_SPEECH = "threatening_speech"
    WEAPON_REFERENCE = "weapon_reference"
    GUNSHOT = "gunshot"
    EXPLOSION = "explosion"


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConsentStatus(str, Enum):
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"


class ProcessingScope(str, Enum):
    NONE = "none"
    ACOUSTIC_ONLY = "acoustic_only"
    ACOUSTIC_AND_SPEECH = "acoustic_and_speech"


RISK_LEVEL_ORDER = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}

MINIMUM_RISK_BY_LABEL = {
    EventLabel.AMBIENT: RiskLevel.NONE,
    EventLabel.NO_SPEECH: RiskLevel.NONE,
    EventLabel.SPEECH_PRESENT: RiskLevel.NONE,
    EventLabel.NON_THREATENING_SPEECH: RiskLevel.NONE,
    EventLabel.SIREN: RiskLevel.LOW,
    EventLabel.SMOKE_ALARM: RiskLevel.LOW,
    EventLabel.GLASS_BREAK: RiskLevel.MEDIUM,
    EventLabel.CROWD_PANIC: RiskLevel.MEDIUM,
    EventLabel.DISTRESS_SPEECH: RiskLevel.MEDIUM,
    EventLabel.THREATENING_SPEECH: RiskLevel.HIGH,
    EventLabel.WEAPON_REFERENCE: RiskLevel.HIGH,
    EventLabel.GUNSHOT: RiskLevel.HIGH,
    EventLabel.EXPLOSION: RiskLevel.HIGH,
}

SPEECH_EVENT_LABELS = frozenset(
    {
        EventLabel.NON_THREATENING_SPEECH,
        EventLabel.DISTRESS_SPEECH,
        EventLabel.THREATENING_SPEECH,
        EventLabel.WEAPON_REFERENCE,
    }
)


class ConsentRecord(BaseModel):
    """Permission evidence for a clip, without storing personal information."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"$id": "https://audio-sentinel.local/schemas/v1/consent-record.schema.json"},
    )

    consent_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        description="Opaque reference to the consent record held outside this manifest.",
    )
    status: ConsentStatus
    processing_scope: ProcessingScope
    device_authorized: bool
    raw_audio_retention_allowed: bool = False
    granted_at: datetime | None = None
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def validate_permission_boundary(self) -> "ConsentRecord":
        if self.status is ConsentStatus.GRANTED:
            if not self.device_authorized:
                raise ValueError("granted consent requires an authorized device")
            if self.processing_scope is ProcessingScope.NONE:
                raise ValueError("granted consent requires an acoustic processing scope")
            if self.granted_at is None:
                raise ValueError("granted consent requires granted_at")
            if self.expires_at is not None and self.expires_at <= self.granted_at:
                raise ValueError("expires_at must be later than granted_at")
        elif (
            self.processing_scope is not ProcessingScope.NONE
            or self.device_authorized
            or self.raw_audio_retention_allowed
        ):
            raise ValueError("denied or withdrawn consent cannot permit device processing or retention")
        return self

    @property
    def permits_acoustic_processing(self) -> bool:
        return self.status is ConsentStatus.GRANTED and self.processing_scope is not ProcessingScope.NONE

    @property
    def permits_speech_processing(self) -> bool:
        return self.status is ConsentStatus.GRANTED and self.processing_scope is ProcessingScope.ACOUSTIC_AND_SPEECH


class EventAnnotation(BaseModel):
    """A time-bounded v1 event annotation or model result."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"$id": "https://audio-sentinel.local/schemas/v1/event-annotation.schema.json"},
    )

    label: EventLabel
    risk_level: RiskLevel
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_annotation(self) -> "EventAnnotation":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")

        minimum_risk = MINIMUM_RISK_BY_LABEL[self.label]
        if RISK_LEVEL_ORDER[self.risk_level] < RISK_LEVEL_ORDER[minimum_risk]:
            raise ValueError(f"{self.label.value} requires at least {minimum_risk.value} risk")
        return self


class PreparedClipRecord(BaseModel):
    """A privacy-minimized manifest for a prepared offline audio clip."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"$id": "https://audio-sentinel.local/schemas/v1/prepared-clip.schema.json"},
    )

    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^1\.0$")
    clip_id: str = Field(min_length=3, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    source_dataset: str = Field(min_length=1, max_length=128)
    audio_path: str = Field(
        min_length=1,
        description="Relative path under the approved data root; never a raw audio payload.",
    )
    sample_rate_hz: int = Field(ge=8_000, le=192_000)
    duration_seconds: float = Field(gt=0)
    consent: ConsentRecord
    annotations: list[EventAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_prepared_clip(self) -> "PreparedClipRecord":
        if Path(self.audio_path).is_absolute() or ".." in Path(self.audio_path).parts:
            raise ValueError("audio_path must be a relative path within the approved data root")
        if not self.consent.permits_acoustic_processing:
            raise ValueError("prepared clips require active acoustic-processing consent")

        for annotation in self.annotations:
            if annotation.end_seconds > self.duration_seconds:
                raise ValueError("annotation end_seconds cannot exceed duration_seconds")
            if annotation.label in SPEECH_EVENT_LABELS and not self.consent.permits_speech_processing:
                raise ValueError("speech event labels require acoustic_and_speech consent")
        return self


def json_schema_documents() -> dict[str, dict[str, object]]:
    """Return the JSON Schema documents for every public v1 contract."""

    return {
        "consent-record.schema.json": ConsentRecord.model_json_schema(),
        "event-annotation.schema.json": EventAnnotation.model_json_schema(),
        "prepared-clip.schema.json": PreparedClipRecord.model_json_schema(),
    }


def write_json_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the public v1 JSON Schemas for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in json_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        exported[filename] = destination
    return exported
