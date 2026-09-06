"""B1.3 deterministic windows from a prepared in-memory signal."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import re

import numpy as np
from numpy.typing import NDArray

from audio_sentinel.audio_arrays import AudioTransformError, check_memory, validate_samples
from audio_sentinel.audio_loader import validate_processing_consent
from audio_sentinel.audio_transforms import PreparedSignal
from audio_sentinel.preparation import PreparedWindowRecord


@dataclass(frozen=True)
class AudioWindow:
    """Independent samples and a planned persistence record; no file is written."""

    clip_id: str
    sample_rate_hz: int
    record: PreparedWindowRecord
    samples: NDArray[np.float32] = field(repr=False, compare=False)


def prepared_output_key(signal: PreparedSignal) -> str:
    """Shared deterministic namespace for full clips, windows, and manifests."""
    identity = json.dumps({
        "clip_id": signal.clip_id,
        "source_sha256": signal.source.sha256,
        "settings": signal.settings.model_dump(mode="json"),
        "preprocessing_version": "1.0",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def iter_windows(signal: PreparedSignal, *, now: datetime | None = None) -> Iterator[AudioWindow]:
    """Validate eagerly, then yield windows ordered by duration and start sample.

    Keep the source signal unchanged while consuming the iterator. Every returned
    window owns its samples; modifying a window never modifies the source.
    """
    consent = validate_processing_consent(signal.consent, now if now is not None else datetime.now(UTC))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}", signal.clip_id):
        raise AudioTransformError("invalid_clip_id", "clip_id must be a valid opaque identifier of 3–128 characters.")
    validate_samples(signal.samples)
    settings = signal.settings
    if signal.sample_rate_hz != settings.target_sample_rate_hz:
        raise AudioTransformError("sample_rate_mismatch", "Prepared sample rate must match the window settings.")
    expected_frames = round(signal.source.num_frames * signal.sample_rate_hz / signal.source.sample_rate_hz)
    expected_channels = 1 if settings.convert_to_mono else signal.source.channels
    if signal.samples.shape != (expected_frames, expected_channels):
        raise AudioTransformError("source_mismatch", "Prepared dimensions must match the source conversion policy.")
    check_memory(signal.num_frames, signal.channels, settings.max_decoded_bytes)

    # A small plan per duration, never a list of all windows or sample arrays.
    groups = []
    for seconds in settings.window_seconds:
        length, hop = settings.window_sample_counts(seconds)
        remainder = signal.num_frames - length
        if settings.tail_policy == "pad":
            count = 1 + max(0, (remainder + hop - 1) // hop)
        else:
            count = 0 if remainder < 0 else 1 + remainder // hop
        if count:
            # A short input can still require a large padded output. Check first.
            check_memory(length, signal.channels, settings.max_decoded_bytes)
        groups.append((seconds, length, hop, count))

    # Opaque, filesystem-safe identity, including full settings and source content.
    key = prepared_output_key(signal)
    samples, rate, clip_id = signal.samples, signal.sample_rate_hz, signal.clip_id

    def generate() -> Iterator[AudioWindow]:
        for group_index, (seconds, length, hop, count) in enumerate(groups):
            for index in range(count):
                # A consumer may pause between windows; recheck permission on resume.
                validate_processing_consent(consent, now if now is not None else datetime.now(UTC))
                start = index * hop
                end = min(start + length, len(samples))
                padding = length - (end - start)
                name = f"g{group_index:04d}-s{start:012d}"
                record = PreparedWindowRecord(
                    window_id=f"{key}-{name}",
                    audio_path=f"prepared/{key}/windows/{name}.wav",
                    window_seconds=seconds, start_sample=start, end_sample=end,
                    padding_samples=padding,
                )
                try:
                    if padding:
                        window = np.zeros((length, samples.shape[1]), dtype=np.float32)
                        window[:end - start] = samples[start:end]
                    else:
                        window = samples[start:end].copy()
                except MemoryError as error:
                    raise AudioTransformError("insufficient_memory", "Not enough memory to allocate this audio window.") from error
                yield AudioWindow(clip_id, rate, record, window)

    return generate()
