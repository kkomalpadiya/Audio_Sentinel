"""Central settings and safe project-path helpers for Audio Sentinel."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_SAMPLE_RATE_HZ = 16_000
DEFAULT_WINDOW_SECONDS = (1.0, 5.0, 10.0)
WindowSeconds = Annotated[float, Field(gt=0, le=600, allow_inf_nan=False)]


class NoiseReductionSettings(BaseModel):
    """Optional stationary spectral gating; implemented in B1.2."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    enabled: bool = False
    method: Literal["stationary_spectral_gate"] = "stationary_spectral_gate"
    reduction_strength: float = Field(default=0.5, gt=0, le=1)


class Paths(BaseModel):
    """The approved project directories used by offline preparation and models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path
    raw_data: Path
    interim_data: Path
    processed_data: Path
    models: Path

    @field_validator("root")
    @classmethod
    def normalize_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def validate_managed_paths(self) -> "Paths":
        for name, path in self.directory_map.items():
            try:
                path.resolve().relative_to(self.root)
            except ValueError as error:
                raise ValueError(f"{name} must be located under project root") from error
        return self

    @classmethod
    def from_root(cls, root: Path) -> "Paths":
        root = root.expanduser().resolve()
        return cls(
            root=root,
            raw_data=root / "data" / "raw",
            interim_data=root / "data" / "interim",
            processed_data=root / "data" / "processed",
            models=root / "models",
        )

    @property
    def directory_map(self) -> dict[str, Path]:
        return {
            "raw_data": self.raw_data,
            "interim_data": self.interim_data,
            "processed_data": self.processed_data,
            "models": self.models,
        }

    def ensure_directories(self) -> None:
        """Create only the application-owned directories under the configured root."""

        for path in self.directory_map.values():
            path.mkdir(parents=True, exist_ok=True)

    def resolve_within(self, directory: Path, relative_path: str | Path) -> Path:
        """Resolve a user-supplied relative path without allowing root escape."""

        if directory not in self.directory_map.values():
            raise ValueError("directory must be one of the configured application directories")

        relative_path = Path(relative_path)
        if relative_path.is_absolute():
            raise ValueError("relative_path must not be absolute")

        candidate = (directory / relative_path).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError as error:
            raise ValueError("relative_path must stay within the configured directory") from error
        return candidate


class AudioSettings(BaseModel):
    """Stable preprocessing defaults shared by the first offline pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    target_sample_rate_hz: int = Field(default=DEFAULT_SAMPLE_RATE_HZ, ge=8_000, le=192_000)
    convert_to_mono: bool = True
    normalize_loudness: bool = True
    normalization_method: Literal["rms"] = "rms"
    target_rms_dbfs: float = Field(default=-20.0, ge=-60, lt=0)
    peak_ceiling_dbfs: float = Field(default=-1.0, ge=-20, le=0)
    max_gain_db: float = Field(default=20.0, ge=0, le=60)
    silence_floor_dbfs: float = Field(default=-60.0, ge=-120, lt=0)
    noise_reduction: NoiseReductionSettings = Field(default_factory=NoiseReductionSettings)
    accepted_formats: tuple[Literal["WAV", "FLAC"], ...] = ("WAV", "FLAC")
    max_input_bytes: int = Field(default=268_435_456, gt=0)
    max_decoded_bytes: int = Field(default=268_435_456, gt=0)
    max_duration_seconds: float = Field(default=600.0, gt=0, le=600)
    max_input_channels: int = Field(default=8, ge=1, le=32)
    output_format: Literal["WAV"] = "WAV"
    output_subtype: Literal["PCM_16"] = "PCM_16"
    window_seconds: tuple[WindowSeconds, ...] = DEFAULT_WINDOW_SECONDS
    window_overlap_ratio: float = Field(default=0.5, ge=0, lt=1)
    tail_policy: Literal["pad", "drop"] = "pad"

    @field_validator("window_seconds")
    @classmethod
    def validate_window_seconds(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            raise ValueError("window_seconds must include at least one window")
        if any(value <= 0 for value in values):
            raise ValueError("window_seconds values must be positive")
        if tuple(sorted(values)) != values or len(set(values)) != len(values):
            raise ValueError("window_seconds must be unique and ordered from shortest to longest")
        return values

    @model_validator(mode="after")
    def validate_preprocessing(self) -> "AudioSettings":
        if not self.accepted_formats or len(set(self.accepted_formats)) != len(self.accepted_formats):
            raise ValueError("accepted_formats must be nonempty and unique")
        if not self.silence_floor_dbfs < self.target_rms_dbfs <= self.peak_ceiling_dbfs:
            raise ValueError("require silence_floor_dbfs < target_rms_dbfs <= peak_ceiling_dbfs")
        for seconds in self.window_seconds:
            length, hop = self.window_sample_counts(seconds)
            if not 1 <= hop <= length:
                raise ValueError("window length and hop must each span at least one sample")
        return self

    def window_sample_counts(self, seconds: float) -> tuple[int, int]:
        """Use Python round (ties to even), then derive hop from rounded length."""
        length = round(seconds * self.target_sample_rate_hz)
        return length, round(length * (1 - self.window_overlap_ratio))


class AudioSentinelSettings(BaseModel):
    """Top-level settings object supplied to all future pipeline modules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paths: Paths
    audio: AudioSettings = Field(default_factory=AudioSettings)

    @classmethod
    def from_project_root(cls, root: Path) -> "AudioSentinelSettings":
        return cls(paths=Paths.from_root(root))

    def ensure_directories(self) -> None:
        self.paths.ensure_directories()


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest project root by walking upward to ``pyproject.toml``."""

    candidate = (start or Path.cwd()).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise FileNotFoundError(f"Could not find a project root from {candidate}")


def load_settings(
    project_root: Path | None = None, *, audio_config_path: Path | None = None
) -> AudioSentinelSettings:
    """Build default settings for the supplied or automatically discovered root."""

    root = project_root or find_project_root(Path(__file__))
    audio = AudioSettings()
    if audio_config_path is not None:
        config_path = Path(audio_config_path)
        if not config_path.is_absolute():
            config_path = root / config_path
        audio = AudioSettings.model_validate_json(config_path.read_text(encoding="utf-8"))
    return AudioSentinelSettings(paths=Paths.from_root(root), audio=audio)
