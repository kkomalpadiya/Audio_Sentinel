"""Shared validation for in-memory float32 audio operations."""

import numpy as np
from numpy.typing import NDArray


class AudioTransformError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def validate_samples(samples: NDArray[np.float32]) -> None:
    if not isinstance(samples, np.ndarray) or samples.dtype != np.float32:
        raise AudioTransformError("invalid_samples", "Expected a float32 NumPy array from the audio loader.")
    if samples.ndim != 2 or samples.shape[0] == 0 or not 1 <= samples.shape[1] <= 32:
        raise AudioTransformError("invalid_samples", "Expected nonempty audio shaped (frames, channels), with 1–32 channels.")
    for start in range(0, len(samples), 65_536):
        if not np.isfinite(samples[start:start + 65_536]).all():
            raise AudioTransformError("non_finite_audio", "Samples must not contain NaN or infinity.")


def check_memory(num_frames: int, channels: int, max_bytes: int) -> None:
    if max_bytes <= 0 or num_frames * channels * 4 > max_bytes:
        raise AudioTransformError("decoded_audio_too_large", "Signal exceeds max_decoded_bytes.")
