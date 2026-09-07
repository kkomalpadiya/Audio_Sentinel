"""A2.1 Log-Mel recipe and sample-grid contract; no feature extraction."""

from numbers import Integral
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LogMelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    recipe_version: Literal["1.0"] = "1.0"
    sample_rate_hz: int = Field(default=16_000, ge=8_000, le=192_000, strict=True)
    n_fft: int = Field(default=512, ge=32, le=8192, strict=True)
    win_length: int = Field(default=400, ge=2, strict=True)
    hop_length: int = Field(default=160, ge=1, strict=True)
    window: Literal["hann_periodic"] = "hann_periodic"
    center: Literal[False] = False
    short_input_policy: Literal["right_zero_pad_to_n_fft"] = "right_zero_pad_to_n_fft"
    frame_tail_policy: Literal["drop"] = "drop"
    n_mels: int = Field(default=64, ge=1, le=256, strict=True)
    fmin_hz: float = Field(default=0.0, ge=0)
    fmax_hz: float = Field(default=8_000.0, gt=0)
    mel_scale: Literal["slaney"] = "slaney"
    mel_norm: Literal["slaney"] = "slaney"
    power: Literal[2.0] = 2.0
    log_scale: Literal["power_db"] = "power_db"
    reference_power: float = Field(default=1.0, gt=0)
    amin: float = Field(default=1e-10, gt=0)
    top_db: float = Field(default=80.0, gt=0, le=200)
    max_feature_bytes: int = Field(default=268_435_456, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_recipe(self) -> "LogMelSettings":
        if self.n_fft & (self.n_fft - 1):
            raise ValueError("n_fft must be a power of two")
        if not 1 <= self.hop_length <= self.win_length <= self.n_fft:
            raise ValueError("require hop_length <= win_length <= n_fft")
        if not self.fmin_hz < self.fmax_hz <= self.sample_rate_hz / 2:
            raise ValueError("require fmin_hz < fmax_hz <= Nyquist")
        if self.n_mels > self.n_fft // 2 + 1:
            raise ValueError("n_mels must not exceed the number of FFT frequency bins")
        if self.amin > self.reference_power:
            raise ValueError("amin must not exceed reference_power")
        return self

    def frame_count(self, num_samples: int) -> int:
        """Count complete FFT frames, padding only inputs shorter than one FFT."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, Integral) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        return 1 + (max(int(num_samples), self.n_fft) - self.n_fft) // self.hop_length

    def expected_shape(self, num_samples: int) -> tuple[int, int]:
        shape = (self.n_mels, self.frame_count(num_samples))
        if shape[0] * shape[1] * 4 > self.max_feature_bytes:
            raise ValueError("feature array exceeds max_feature_bytes")
        return shape
