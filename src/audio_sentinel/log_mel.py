"""B2.1 in-memory Log-Mel generation from decoded prepared mono audio."""

from dataclasses import dataclass, field
from numbers import Integral
import warnings

import librosa
import numpy as np
from numpy.typing import NDArray

from audio_sentinel.audio_arrays import AudioTransformError, validate_samples
from audio_sentinel.feature_settings import LogMelSettings


class LogMelError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class LogMelFeatures:
    """Owned numeric array and recipe; file identity is added by A2.2 storage."""

    values: NDArray[np.float32] = field(repr=False, compare=False)
    settings: LogMelSettings
    num_samples: int
    analysis_padding_samples: int
    extractor_version: str
    extractor_name: str = "librosa"
    implementation_version: str = "1.0"


def estimate_log_mel_working_bytes(num_samples: int, settings: LogMelSettings) -> int:
    """Conservative array-workspace estimate, excluding caller input/library overhead."""
    frames = settings.frame_count(num_samples)
    bins = settings.n_fft // 2 + 1
    # Signal conversion/padding, complex FFT plus power/temporaries, Mel/log/output
    # arrays, filter construction, analysis window, and fixed allocation slack.
    return (16 * max(num_samples, settings.n_fft) + 64 * bins * frames
            + 48 * settings.n_mels * frames + 48 * settings.n_mels * bins
            + 32 * settings.n_fft + 1_048_576)


def generate_log_mel(
    samples: NDArray[np.float32], sample_rate_hz: int, settings: LogMelSettings,
    *, max_working_bytes: int = 268_435_456,
) -> LogMelFeatures:
    """Transform one float32 (samples, 1) window without file I/O or source edits.

    Supply decoded, already prepared audio. No mixing, resampling, normalization,
    permission handling, or persistence happens here. The caller owns permission
    and source-file verification, and must not modify samples during this call.
    """
    # Revalidate even if a caller used Pydantic model_copy/update to bypass checks.
    settings = LogMelSettings.model_validate(settings.model_dump())
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, Integral):
        raise LogMelError("invalid_sample_rate", "Sample rate must be an integer.")
    if sample_rate_hz != settings.sample_rate_hz:
        raise LogMelError("sample_rate_mismatch", "Prepared audio must match the Log-Mel sample rate.")
    if (isinstance(max_working_bytes, bool) or not isinstance(max_working_bytes, Integral)
            or max_working_bytes <= 0):
        raise LogMelError("invalid_memory_limit", "max_working_bytes must be a positive integer.")
    if (not isinstance(samples, np.ndarray) or samples.dtype != np.float32
            or samples.ndim != 2 or len(samples) == 0):
        raise LogMelError("invalid_samples", "Expected nonempty float32 audio shaped (samples, 1).")
    if samples.shape[1] != 1:
        raise LogMelError("mono_required", "Log-Mel version 1 requires prepared mono audio.")
    try:
        shape = settings.expected_shape(len(samples))
    except ValueError as error:
        raise LogMelError("feature_too_large", str(error)) from error
    if estimate_log_mel_working_bytes(len(samples), settings) > max_working_bytes:
        raise LogMelError("working_memory_exceeded", "Estimated Log-Mel workspace exceeds max_working_bytes.")
    try:
        validate_samples(samples)
        # Float64 intermediates preserve tiny positive floors and avoid squaring
        # overflow for large but finite float32 samples. Input is always copied.
        signal = np.array(samples[:, 0], dtype=np.float64, copy=True)
        padding = max(0, settings.n_fft - len(signal))
        if padding:
            signal = np.pad(signal, (0, padding), mode="constant")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Empty filters detected in mel frequency basis.*", category=UserWarning)
            basis = librosa.filters.mel(
                sr=settings.sample_rate_hz, n_fft=settings.n_fft, n_mels=settings.n_mels,
                fmin=settings.fmin_hz, fmax=settings.fmax_hz, htk=False,
                norm="slaney", dtype=np.float64,
            )
        if not np.isfinite(basis).all() or np.any(np.max(basis, axis=1) <= 0):
            raise LogMelError("invalid_mel_filters", "Recipe produces empty or non-finite Mel filters.")
        spectrum = librosa.stft(
            signal, n_fft=settings.n_fft, hop_length=settings.hop_length,
            win_length=settings.win_length, window="hann", center=False,
            dtype=np.complex128,
        )
        power = np.abs(spectrum)
        del spectrum
        np.square(power, out=power)
        mel_power = basis @ power
        del basis, power
        db = librosa.power_to_db(mel_power, ref=settings.reference_power,
                                 amin=settings.amin, top_db=settings.top_db)
        values = np.array(db, dtype=np.float32, order="C", copy=True)
        if values.shape != shape or not np.isfinite(values).all():
            raise LogMelError("invalid_features", "Generated features have an invalid shape or non-finite values.")
        return LogMelFeatures(values, settings, len(samples), padding, librosa.__version__)
    except AudioTransformError as error:
        raise LogMelError(error.code, str(error)) from error
    except MemoryError as error:
        raise LogMelError("insufficient_memory", "Not enough memory to generate Log-Mel features.") from error
