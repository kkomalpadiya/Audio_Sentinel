"""A2.1 metadata for one window's numeric Log-Mel artifact.

Validation describes metadata, not the existence or integrity of files. B2.1
computes arrays in log_mel.py; A2.2 writes and verifies them in feature_persistence.py.
"""

from datetime import datetime
import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.preparation import PreparedAudioManifest, PreparedWindowRecord, validate_relative_audio_path


Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class LogMelSource(BaseModel):
    """References to data/interim plus raw-file identity inherited from preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    preparation_manifest_path: str = Field(min_length=1)
    preparation_manifest_sha256: Sha256
    raw_audio_sha256: Sha256
    window_audio_sha256: Sha256
    clip_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
    sample_rate_hz: int = Field(ge=8_000, le=192_000, strict=True)
    channels: Literal[1] = 1
    window: PreparedWindowRecord

    @field_validator("preparation_manifest_path")
    @classmethod
    def validate_manifest_path(cls, value: str) -> str:
        validate_relative_audio_path(value)
        if PurePosixPath(value).suffix != ".json":
            raise ValueError("preparation_manifest_path must end in .json")
        return value

    @model_validator(mode="after")
    def validate_length(self) -> "LogMelSource":
        if self.num_samples != round(self.window.window_seconds * self.sample_rate_hz):
            raise ValueError("source window span and padding must match its duration and rate")
        return self

    @property
    def num_samples(self) -> int:
        return self.window.end_sample - self.window.start_sample + self.window.padding_samples

    def validate_against(self, manifest: PreparedAudioManifest) -> None:
        """Check semantic linkage to a loaded manifest; caller verifies file hashes."""
        manifest = PreparedAudioManifest.model_validate(manifest.model_dump())
        if (self.clip_id, self.raw_audio_sha256, self.sample_rate_hz, self.channels) != (
            manifest.clip.clip_id, manifest.source.sha256, manifest.clip.sample_rate_hz, manifest.channels
        ):
            raise ValueError("feature source does not match the preparation manifest")
        if self.window not in manifest.windows:
            raise ValueError("source window is not in the preparation manifest")


class LogMelFeatureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    feature_type: Literal["log_mel_power"] = "log_mel_power"
    source: LogMelSource
    settings: LogMelSettings
    feature_path: str = Field(min_length=1)
    feature_sha256: Sha256
    storage_format: Literal["npy"] = "npy"
    dtype: Literal["float32"] = "float32"
    axes: tuple[Literal["mel"], Literal["time"]] = ("mel", "time")
    shape: tuple[PositiveInt, PositiveInt]
    analysis_padding_samples: int = Field(default=0, ge=0, strict=True)
    extractor_name: Literal["librosa"] = "librosa"
    extractor_version: str = Field(min_length=1, max_length=64, pattern=r"^\S+$")
    implementation_version: Literal["1.0"] = "1.0"
    created_at: datetime

    @field_validator("feature_path")
    @classmethod
    def validate_feature_path(cls, value: str) -> str:
        validate_relative_audio_path(value)
        if PurePosixPath(value).suffix != ".npy":
            raise ValueError("feature_path must end in .npy")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_feature(self) -> "LogMelFeatureMetadata":
        if self.source.sample_rate_hz != self.settings.sample_rate_hz:
            raise ValueError("feature sample rate must match prepared source; no implicit resampling")
        if self.shape != self.settings.expected_shape(self.source.num_samples):
            raise ValueError("shape must match the recipe and source sample count")
        if self.analysis_padding_samples != max(0, self.settings.n_fft - self.source.num_samples):
            raise ValueError("analysis padding must only extend short inputs to n_fft")
        return self


def feature_schema_documents() -> dict[str, dict[str, object]]:
    documents = {
        "log-mel-settings.schema.json": LogMelSettings.model_json_schema(),
        "log-mel-feature.schema.json": LogMelFeatureMetadata.model_json_schema(),
    }
    for name, document in documents.items():
        document["$id"] = f"https://audio-sentinel.local/schemas/v1/{name}"
    return documents


def write_feature_schemas(output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, document in feature_schema_documents().items():
        (output_directory / name).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
