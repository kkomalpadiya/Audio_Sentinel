"""A1.2 in-memory audio conversion. No file persistence or segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import math
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from audio_sentinel.audio_loader import LoadedAudio, validate_processing_consent
from audio_sentinel.config import AudioSettings
from audio_sentinel.contracts import ConsentRecord
from audio_sentinel.preparation import SourceAudioMetadata


class AudioTransformError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class NormalizationStats:
    reason: Literal["disabled", "silence", "target", "gain_limit", "peak_limit"]
    gain_db: float
    input_rms_dbfs: float | None
    output_rms_dbfs: float | None
    input_peak_dbfs: float | None
    output_peak_dbfs: float | None


@dataclass(frozen=True)
class NormalizedAudio:
    samples: NDArray[np.float32] = field(repr=False, compare=False)
    stats: NormalizationStats


@dataclass(frozen=True)
class PreparedSignal:
    """A complete transformed signal in memory; source metadata stays original."""

    clip_id: str
    consent: ConsentRecord
    source: SourceAudioMetadata
    settings: AudioSettings
    sample_rate_hz: int
    samples: NDArray[np.float32] = field(repr=False, compare=False)
    normalization: NormalizationStats

    @property
    def num_frames(self) -> int:
        return self.samples.shape[0]

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.sample_rate_hz


def _validate_samples(samples: NDArray[np.float32]) -> None:
    if not isinstance(samples, np.ndarray) or samples.dtype != np.float32:
        raise AudioTransformError("invalid_samples", "Expected a float32 NumPy array from the audio loader.")
    if samples.ndim != 2 or samples.shape[0] == 0 or not 1 <= samples.shape[1] <= 32:
        raise AudioTransformError("invalid_samples", "Expected nonempty audio shaped (frames, channels), with 1–32 channels.")
    for start in range(0, len(samples), 65_536):
        if not np.isfinite(samples[start:start + 65_536]).all():
            raise AudioTransformError("non_finite_audio", "Samples must not contain NaN or infinity.")


def _check_memory(num_frames: int, channels: int, max_bytes: int) -> None:
    if max_bytes <= 0 or num_frames * channels * 4 > max_bytes:
        raise AudioTransformError("decoded_audio_too_large", "Signal exceeds max_decoded_bytes.")


def convert_to_mono(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Average channels without modifying the caller's array; keep a channel axis."""
    _validate_samples(samples)
    mono = np.empty((len(samples), 1), dtype=np.float32)
    for start in range(0, len(samples), 65_536):
        # Float64 accumulation avoids overflow when averaging large finite inputs.
        mono[start:start + 65_536, 0] = samples[start:start + 65_536].mean(axis=1, dtype=np.float64)
    return mono


def resample_audio(
    samples: NDArray[np.float32], source_rate_hz: int, target_rate_hz: int, *, max_decoded_bytes: int
) -> NDArray[np.float32]:
    """Polyphase FIR resampling along time, with A1.1's exact rounded frame count."""
    _validate_samples(samples)
    for rate in (source_rate_hz, target_rate_hz):
        if isinstance(rate, bool) or not isinstance(rate, Integral) or not 8_000 <= rate <= 192_000:
            raise AudioTransformError("invalid_sample_rate", "Sample rates must be integers from 8,000 to 192,000 Hz.")
    expected = round(len(samples) * int(target_rate_hz) / int(source_rate_hz))
    if expected < 1:
        raise AudioTransformError("audio_too_short", "Source rounds to zero frames at the target rate.")
    _check_memory(expected, samples.shape[1], max_decoded_bytes)
    if source_rate_hz == target_rate_hz:
        return samples.copy()
    divisor = math.gcd(int(source_rate_hz), int(target_rate_hz))
    with np.errstate(over="ignore", invalid="ignore"):
        output = resample_poly(
            samples, int(target_rate_hz) // divisor, int(source_rate_hz) // divisor,
            axis=0, window=("kaiser", 5.0), padtype="constant", cval=0,
        )
    # SciPy normally rounds up. The public contract rounds to nearest, ties to even.
    if len(output) < expected:
        output = np.pad(output, ((0, expected - len(output)), (0, 0)))
    output = np.array(output[:expected], dtype=np.float32, order="C", copy=True)
    _validate_samples(output)
    return output


def _levels(samples: NDArray[np.float32]) -> tuple[float, float]:
    sum_squares, peak = 0.0, 0.0
    for start in range(0, len(samples), 65_536):
        block = samples[start:start + 65_536]
        sum_squares += float(np.sum(np.square(block, dtype=np.float64)))
        peak = max(peak, float(np.max(np.abs(block))))
    return math.sqrt(sum_squares / samples.size), peak


def _dbfs(level: float) -> float | None:
    # Zero has no finite dBFS value; use None rather than non-JSON infinity.
    return 20 * math.log10(level) if level > 0 else None


def normalize_loudness(samples: NDArray[np.float32], settings: AudioSettings) -> NormalizedAudio:
    """RMS normalization using one gain shared by all frames and channels."""
    _validate_samples(samples)
    _check_memory(len(samples), samples.shape[1], settings.max_decoded_bytes)
    rms, peak = _levels(samples)
    gain = 1.0
    if not settings.normalize_loudness:
        reason = "disabled"
    elif rms <= float(np.float32(10 ** (settings.silence_floor_dbfs / 20))):
        reason = "silence"
    else:
        ceiling = 10 ** (settings.peak_ceiling_dbfs / 20)
        safe_ceiling = np.float32(ceiling)
        if float(safe_ceiling) > ceiling:
            safe_ceiling = np.nextafter(safe_ceiling, np.float32(0))
        # The representable ceiling prevents float32 rounding from exceeding the limit.
        gain, reason = min(
            (10 ** (settings.target_rms_dbfs / 20) / rms, "target"),
            (10 ** (settings.max_gain_db / 20), "gain_limit"),
            (float(safe_ceiling) / peak, "peak_limit"),
            key=lambda item: item[0],
        )
    output = np.empty_like(samples)
    np.multiply(samples, np.float64(gain), out=output, casting="unsafe")
    output_rms, output_peak = _levels(output)
    return NormalizedAudio(output, NormalizationStats(
        reason=reason, gain_db=20 * math.log10(gain), input_rms_dbfs=_dbfs(rms),
        output_rms_dbfs=_dbfs(output_rms), input_peak_dbfs=_dbfs(peak), output_peak_dbfs=_dbfs(output_peak),
    ))


def prepare_signal(loaded: LoadedAudio, settings: AudioSettings, *, now: datetime | None = None) -> PreparedSignal:
    """Apply A1.2 to loader output. Noise reduction must remain disabled until B1.2."""
    consent = validate_processing_consent(loaded.consent, now if now is not None else datetime.now(UTC))
    if settings.noise_reduction.enabled:
        raise AudioTransformError("noise_reduction_unavailable", "Noise reduction belongs to B1.2 and is not implemented yet.")
    _validate_samples(loaded.samples)
    if loaded.samples.shape != (loaded.source.num_frames, loaded.source.channels):
        raise AudioTransformError("source_mismatch", "Loaded samples do not match source metadata.")
    _check_memory(len(loaded.samples), loaded.source.channels, settings.max_decoded_bytes)
    try:
        samples = convert_to_mono(loaded.samples) if settings.convert_to_mono else loaded.samples
        samples = resample_audio(samples, loaded.source.sample_rate_hz, settings.target_sample_rate_hz,
                                 max_decoded_bytes=settings.max_decoded_bytes)
        # B1.2 will insert noise reduction here, before normalization.
        normalized = normalize_loudness(samples, settings)
    except MemoryError as error:
        raise AudioTransformError("insufficient_memory", "Not enough memory to transform this recording.") from error
    consent = validate_processing_consent(consent, now if now is not None else datetime.now(UTC))
    return PreparedSignal(
        clip_id=loaded.clip_id, consent=consent, source=loaded.source.model_copy(deep=True),
        settings=settings, sample_rate_hz=settings.target_sample_rate_hz,
        samples=normalized.samples, normalization=normalized.stats,
    )
