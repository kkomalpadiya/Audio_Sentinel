"""Load authorized local recordings for the offline preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError
import soundfile as sf

from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import ConsentRecord
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.preparation import SourceAudioMetadata, validate_relative_audio_path


class AudioLoadError(ValueError):
    """A rejected input; code is stable for callers, message explains the failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class LoadedAudio:
    """Float32 samples shaped (frames, channels), still at the source rate."""

    clip_id: str
    consent: ConsentRecord
    source: SourceAudioMetadata
    samples: NDArray[np.float32] = field(repr=False, compare=False)


def _validated_consent(consent: ConsentRecord, now: datetime) -> ConsentRecord:
    timestamps = (now, consent.granted_at, consent.expires_at)
    if any(value is not None and value.utcoffset() is None for value in timestamps):
        raise AudioLoadError("invalid_consent", "Consent timestamps and current time must include a timezone.")
    try:
        # Revalidate a copy: a caller may have mutated the original Pydantic record.
        consent = ConsentRecord.model_validate(consent.model_dump())
    except ValidationError as error:
        raise AudioLoadError("invalid_consent", "Consent record is inconsistent.") from error
    if not consent.permits_acoustic_processing:
        raise AudioLoadError("consent_denied", "Active acoustic-processing permission is required.")
    if consent.granted_at > now:
        raise AudioLoadError("consent_not_yet_active", "Consent has not taken effect yet.")
    if consent.expires_at is not None and now >= consent.expires_at:
        raise AudioLoadError("consent_expired", "Consent has expired.")
    return consent


def _resolve_source(path: Path, raw_root: Path) -> tuple[Path, str]:
    try:
        path = Path(path)
        root = raw_root.resolve(strict=True)
        if not root.is_dir():
            raise AudioLoadError("invalid_path", "The raw data root must be a directory.")
        if not path.is_absolute():
            # Reject traversal even if it happens to resolve back inside the root.
            validate_relative_audio_path(path.as_posix())
            path = root / path
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
        validate_relative_audio_path(relative)
        return resolved, relative
    except FileNotFoundError as error:
        raise AudioLoadError("file_not_found", "The source file or raw data directory does not exist.") from error
    except (ValueError, RuntimeError) as error:
        if isinstance(error, AudioLoadError):
            raise
        raise AudioLoadError("invalid_path", "Source must be a file contained in data/raw, including symlink targets.") from error
    except OSError as error:
        raise AudioLoadError("file_unreadable", "Cannot resolve the source file.") from error


def _validate_wav_container(stream: BinaryIO, size_bytes: int) -> None:
    """Reject truncated RIFF chunks which a forgiving decoder may silently shorten."""
    stream.seek(0)
    header = stream.read(12)
    if len(header) != 12 or header[:4] not in (b"RIFF", b"RIFX") or header[8:] != b"WAVE":
        raise AudioLoadError("invalid_audio", "Invalid WAV container header.")
    byteorder = "little" if header[:4] == b"RIFF" else "big"
    end = int.from_bytes(header[4:8], byteorder) + 8
    if end > size_bytes or end < 12:
        raise AudioLoadError("invalid_audio", "WAV container is truncated or has an invalid length.")
    offset = 12
    pcm_block_align = None
    data_lengths = []
    while offset < end:
        stream.seek(offset)
        chunk = stream.read(8)
        if len(chunk) != 8 or offset + 8 > end:
            raise AudioLoadError("invalid_audio", "WAV chunk header is truncated.")
        length = int.from_bytes(chunk[4:8], byteorder)
        next_offset = offset + 8 + length
        if next_offset > end:
            raise AudioLoadError("invalid_audio", "WAV chunk payload is truncated.")
        if chunk[:4] == b"fmt " and length >= 16:
            format_header = stream.read(16)
            if int.from_bytes(format_header[:2], byteorder) in (1, 3):
                pcm_block_align = int.from_bytes(format_header[12:14], byteorder)
        elif chunk[:4] == b"data":
            data_lengths.append(length)
        # RIFF chunks align to two bytes; tolerate omitted padding at physical EOF.
        offset = next_offset + (length % 2 if next_offset < end else 0)
    if pcm_block_align is not None and (
        pcm_block_align == 0 or any(length % pcm_block_align for length in data_lengths)
    ):
        raise AudioLoadError("invalid_audio", "WAV data contains an incomplete PCM frame.")
    stream.seek(0)


def _file_signature(details: os.stat_result) -> tuple[int, int, int]:
    return details.st_size, details.st_mtime_ns, details.st_ctime_ns


def load_audio(
    clip: InputAudio, settings: AudioSentinelSettings, *, now: datetime | None = None
) -> LoadedAudio:
    """Read one local file without resampling, normalization, or disk writes.

    Relative paths start at settings.paths.raw_data. Absolute paths must resolve
    inside that root. `now` is an optional clock override for reproducible tests.
    """
    consent = _validated_consent(clip.consent, now if now is not None else datetime.now(UTC))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}", clip.clip_id):
        raise AudioLoadError("invalid_clip_id", "clip_id must be a valid opaque identifier of 3–128 characters.")
    path, relative_path = _resolve_source(clip.audio_path, settings.paths.raw_data)
    policy = settings.audio
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise AudioLoadError("invalid_path", "Source must be a regular file.")
        with path.open("rb") as stream:
            initial = os.fstat(stream.fileno())
            if initial.st_size == 0:
                raise AudioLoadError("empty_audio", "Source file is empty.")
            if initial.st_size > policy.max_input_bytes:
                raise AudioLoadError("input_too_large", "Source exceeds max_input_bytes.")
            header = stream.read(12)
            if header[:4] in (b"RIFF", b"RIFX") and header[8:] == b"WAVE":
                _validate_wav_container(stream, initial.st_size)
            stream.seek(0)
            with sf.SoundFile(stream, mode="r") as audio:
                # WAVE_FORMAT_EXTENSIBLE is still a WAV container.
                container = "WAV" if audio.format == "WAVEX" else audio.format
                if container not in policy.accepted_formats:
                    raise AudioLoadError("unsupported_format", "Decoded container is not an accepted WAV or FLAC format.")
                frames, channels, rate = audio.frames, audio.channels, audio.samplerate
                if frames <= 0:
                    raise AudioLoadError("empty_audio", "Source contains no audio frames.")
                if not 8_000 <= rate <= 192_000:
                    raise AudioLoadError("invalid_sample_rate", "Source sample rate must be 8,000–192,000 Hz.")
                if not 1 <= channels <= policy.max_input_channels:
                    raise AudioLoadError("too_many_channels", "Source exceeds max_input_channels.")
                if frames / rate > policy.max_duration_seconds:
                    raise AudioLoadError("audio_too_long", "Source exceeds max_duration_seconds.")
                if frames * channels * np.dtype(np.float32).itemsize > policy.max_decoded_bytes:
                    raise AudioLoadError("decoded_audio_too_large", "Decoded samples exceed max_decoded_bytes.")
                if round(frames * policy.target_sample_rate_hz / rate) < 1:
                    raise AudioLoadError("audio_too_short", "Source would contain no frames at the configured target rate.")

                samples = np.empty((frames, channels), dtype=np.float32)
                # Bound temporary allocations and detect NaN/Infinity per block.
                for start in range(0, frames, 65_536):
                    count = min(65_536, frames - start)
                    block = audio.read(count, dtype="float32", always_2d=True)
                    if block.shape != (count, channels):
                        raise AudioLoadError("invalid_audio", "Decoded frame count is shorter than the file header declares.")
                    if not np.isfinite(block).all():
                        raise AudioLoadError("non_finite_audio", "Source contains NaN or infinite samples.")
                    samples[start:start + count] = block
                if len(audio.read(1, dtype="float32", always_2d=True)):
                    raise AudioLoadError("invalid_audio", "Decoded frame count exceeds the file header.")

            stream.seek(0)
            digest = hashlib.sha256()
            remaining = initial.st_size
            while remaining:
                chunk = stream.read(min(1_048_576, remaining))
                if not chunk:
                    raise AudioLoadError("source_changed", "Source changed while it was being loaded.")
                digest.update(chunk)
                remaining -= len(chunk)
            if stream.read(1) or _file_signature(os.fstat(stream.fileno())) != _file_signature(initial):
                raise AudioLoadError("source_changed", "Source changed while it was being loaded.")
    except AudioLoadError:
        raise
    except sf.SoundFileError as error:
        raise AudioLoadError("invalid_audio", "Source is corrupt, unsupported, or cannot be decoded.") from error
    except (OSError, EOFError) as error:
        raise AudioLoadError("file_unreadable", "Cannot read the source file.") from error
    except MemoryError as error:
        raise AudioLoadError("insufficient_memory", "Not enough memory to load this clip; use a shorter recording.") from error

    # A long decode must not return audio after its permission has expired.
    consent = _validated_consent(consent, now if now is not None else datetime.now(UTC))
    source = SourceAudioMetadata(
        audio_path=relative_path, sha256=digest.hexdigest(), format=container,
        sample_rate_hz=rate, channels=channels, num_frames=frames, size_bytes=initial.st_size,
    )
    return LoadedAudio(clip_id=clip.clip_id, consent=consent, source=source, samples=samples)
