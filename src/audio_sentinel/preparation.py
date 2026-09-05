"""A1.1 metadata contracts. These models do not load or transform audio."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.config import AudioSettings, WindowSeconds
from audio_sentinel.contracts import PreparedClipRecord


Identifier = Annotated[str, Field(min_length=3, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]


def validate_relative_audio_path(value: str) -> str:
    """Portable paths for manifests shared between Windows and Linux."""
    path = PurePosixPath(value)
    if (
        not value or path.is_absolute() or "\\" in value or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("audio_path must be a relative POSIX file path without traversal")
    return value


class PreparationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SourceAudioMetadata(PreparationRecord):
    """Original file properties; audio_path is relative to data/raw."""

    audio_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    format: Literal["WAV", "FLAC"]
    sample_rate_hz: int = Field(ge=8_000, le=192_000)
    channels: int = Field(ge=1, le=32)
    num_frames: int = Field(gt=0)
    size_bytes: int = Field(gt=0)

    _validate_path = field_validator("audio_path")(validate_relative_audio_path)


class PreparedWindowRecord(PreparationRecord):
    """Offsets count frames in the prepared full clip; end_sample is exclusive."""

    window_id: Identifier
    audio_path: str = Field(min_length=1)
    window_seconds: WindowSeconds
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    padding_samples: int = Field(default=0, ge=0)

    _validate_path = field_validator("audio_path")(validate_relative_audio_path)

    @model_validator(mode="after")
    def validate_span(self) -> "PreparedWindowRecord":
        if self.end_sample <= self.start_sample:
            raise ValueError("end_sample must be greater than start_sample")
        if PurePosixPath(self.audio_path).suffix.lower() != ".wav":
            raise ValueError("prepared window audio_path must end in .wav")
        return self


class PreparedAudioManifest(PreparationRecord):
    """One source, one full prepared clip, and its complete window inventory."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False,
        json_schema_extra={"$id": "https://audio-sentinel.local/schemas/v1/prepared-audio-manifest.schema.json"},
    )

    schema_version: Literal["1.0"] = "1.0"
    preprocessing_version: Literal["1.0"] = "1.0"
    source: SourceAudioMetadata
    settings: AudioSettings
    clip: PreparedClipRecord
    channels: int = Field(ge=1, le=32)
    num_frames: int = Field(gt=0)
    windows: tuple[PreparedWindowRecord, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> "PreparedAudioManifest":
        settings, source, clip = self.settings, self.source, self.clip
        validate_relative_audio_path(clip.audio_path)
        if PurePosixPath(clip.audio_path).suffix.lower() != ".wav":
            raise ValueError("prepared clip audio_path must end in .wav")
        if source.format not in settings.accepted_formats:
            raise ValueError("source format is not accepted by settings")
        if source.size_bytes > settings.max_input_bytes or source.channels > settings.max_input_channels:
            raise ValueError("source exceeds input size or channel limits")
        if source.num_frames / source.sample_rate_hz > settings.max_duration_seconds:
            raise ValueError("source exceeds maximum duration")
        if clip.sample_rate_hz != settings.target_sample_rate_hz:
            raise ValueError("clip sample rate must match settings")
        expected_frames = round(source.num_frames * settings.target_sample_rate_hz / source.sample_rate_hz)
        if self.num_frames != expected_frames:
            raise ValueError("num_frames must match the resampled source length")
        if self.channels != (1 if settings.convert_to_mono else source.channels):
            raise ValueError("channels must match the mono conversion policy")
        if not math.isfinite(clip.duration_seconds) or not math.isclose(
            clip.duration_seconds, self.num_frames / clip.sample_rate_hz, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("clip duration must equal num_frames / sample_rate_hz")

        ids: set[str] = set()
        paths = {clip.audio_path.casefold()}
        grouped = {seconds: [] for seconds in settings.window_seconds}
        for window in self.windows:
            if window.window_id in ids or window.audio_path.casefold() in paths:
                raise ValueError("window IDs and prepared audio paths must be unique")
            ids.add(window.window_id)
            paths.add(window.audio_path.casefold())
            if window.window_seconds not in grouped:
                raise ValueError("window duration is not configured")
            grouped[window.window_seconds].append(window)
        if list(self.windows) != sorted(self.windows, key=lambda w: (w.window_seconds, w.start_sample)):
            raise ValueError("windows must be ordered by duration then start_sample")

        for seconds, windows in grouped.items():
            length, hop = settings.window_sample_counts(seconds)
            remainder = self.num_frames - length
            if settings.tail_policy == "pad":
                count = 1 + max(0, (remainder + hop - 1) // hop)
            else:
                count = 0 if remainder < 0 else 1 + remainder // hop
            if len(windows) != count:
                raise ValueError("window inventory is incomplete for the configured tail policy")
            for index, window in enumerate(windows):
                start = index * hop
                end = min(start + length, self.num_frames)
                if (window.start_sample, window.end_sample, window.padding_samples) != (
                    start, end, length - (end - start)
                ):
                    raise ValueError("window offsets or padding do not match the configured sample grid")
        return self


def preparation_schema_documents() -> dict[str, dict[str, object]]:
    settings_schema = AudioSettings.model_json_schema()
    settings_schema["$id"] = "https://audio-sentinel.local/schemas/v1/preprocessing-settings.schema.json"
    return {
        "preprocessing-settings.schema.json": settings_schema,
        "prepared-audio-manifest.schema.json": PreparedAudioManifest.model_json_schema(),
    }


def write_preparation_schemas(output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, document in preparation_schema_documents().items():
        (output_directory / filename).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
