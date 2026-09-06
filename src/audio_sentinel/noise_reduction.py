"""Optional stationary spectral gating for the offline audio preparation path."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter
from scipy.signal import ShortTimeFFT

from audio_sentinel.audio_arrays import AudioTransformError, check_memory, validate_samples
from audio_sentinel.config import NoiseReductionSettings


@dataclass(frozen=True)
class NoiseReductionStats:
    reason: Literal["disabled", "too_short", "silence", "processed"]
    algorithm_version: str = "1.0"
    profile_frames: int = 0
    mean_spectral_gain: float = 1.0


@dataclass(frozen=True)
class NoiseReducedAudio:
    samples: NDArray[np.float32] = field(repr=False, compare=False)
    stats: NoiseReductionStats


def reduce_noise(
    samples: NDArray[np.float32], sample_rate_hz: int, settings: NoiseReductionSettings,
    *, max_decoded_bytes: int,
) -> NoiseReducedAudio:
    """Reduce noise per channel without changing length, rate, or input samples.

    The profile is a per-frequency temporal quantile of fully contained STFT frames.
    No external noise recording or learned model is needed.
    """
    validate_samples(samples)
    check_memory(len(samples), samples.shape[1], max_decoded_bytes)
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, Integral) or not 8_000 <= sample_rate_hz <= 192_000:
        raise AudioTransformError("invalid_sample_rate", "Sample rate must be an integer from 8,000 to 192,000 Hz.")
    if not settings.enabled:
        return NoiseReducedAudio(samples.copy(), NoiseReductionStats("disabled"))
    if len(samples) < settings.fft_size:
        return NoiseReducedAudio(samples.copy(), NoiseReductionStats("too_short"))
    if not np.any(samples):
        return NoiseReducedAudio(samples.copy(), NoiseReductionStats("silence"))

    transform = ShortTimeFFT.from_window(
        "hann", fs=sample_rate_hz, nperseg=settings.fft_size,
        noverlap=3 * settings.fft_size // 4, scale_to="magnitude",
    )
    frame_count = transform.p_num(len(samples))
    # Budget per-channel complex spectra, magnitude/masks/quantile copies and output.
    estimated_working_bytes = (settings.fft_size // 2 + 1) * frame_count * 64 + len(samples) * 24
    if estimated_working_bytes > settings.max_working_bytes:
        raise AudioTransformError("noise_working_memory_exceeded", "Spectral working-memory estimate exceeds noise_reduction.max_working_bytes.")
    centers = np.rint(transform.t(len(samples)) * sample_rate_hz).astype(np.int64)
    interior = (centers >= settings.fft_size // 2) & (centers <= len(samples) - settings.fft_size // 2)
    profile_frames = int(np.count_nonzero(interior))
    if profile_frames < 4:
        return NoiseReducedAudio(samples.copy(), NoiseReductionStats("too_short"))

    try:
        output = np.empty_like(samples)
        gain_total = 0.0
        for channel in range(samples.shape[1]):
            signal = samples[:, channel]
            if not np.any(signal):
                output[:, channel] = signal
                gain_total += 1.0
                continue
            spectrum = transform.stft(signal.astype(np.float64), padding="zeros")
            magnitude = np.abs(spectrum)
            threshold = settings.threshold_multiplier * np.quantile(
                magnitude[:, interior], settings.noise_quantile, axis=1, keepdims=True,
            )
            # No estimated noise means no suppression at that frequency.
            activity = np.ones_like(magnitude)
            np.divide(magnitude - threshold, threshold, out=activity, where=threshold > 0)
            np.clip(activity, 0, 1, out=activity)
            activity = uniform_filter(activity, size=(3, 3), mode="nearest")
            np.clip(activity, 0, 1, out=activity)
            gain = 1 - settings.reduction_strength * (1 - activity)
            gain_total += float(np.mean(gain))
            spectrum *= gain
            reconstructed = transform.istft(spectrum, k0=0, k1=len(samples))
            with np.errstate(over="ignore", invalid="ignore"):
                output[:, channel] = reconstructed
        validate_samples(output)
    except MemoryError as error:
        raise AudioTransformError("insufficient_memory", "Not enough memory for noise reduction; shorten the clip or disable this optional step.") from error
    return NoiseReducedAudio(output, NoiseReductionStats(
        "processed", profile_frames=profile_frames, mean_spectral_gain=gain_total / samples.shape[1],
    ))
