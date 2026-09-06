"""A1.3 verified, staged persistence of prepared audio and its manifest."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import errno
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from tempfile import mkdtemp
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import soundfile as sf

from audio_sentinel.audio_loader import validate_processing_consent
from audio_sentinel.audio_transforms import PreparedSignal
from audio_sentinel.config import Paths, PersistenceSettings
from audio_sentinel.contracts import EventAnnotation, PreparedClipRecord
from audio_sentinel.interfaces import PreprocessedAudio
from audio_sentinel.preparation import PreparedAudioManifest, SourceAudioMetadata
from audio_sentinel.segmentation import iter_windows, prepared_output_key


class AudioPersistenceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class PersistedAudio:
    directory: Path
    manifest_path: Path
    manifest: PreparedAudioManifest
    reused: bool

    @property
    def audio(self) -> PreprocessedAudio:
        """The file-backed interface consumed by later pipeline stages."""
        clip = self.manifest.clip
        return PreprocessedAudio(clip.clip_id, self.directory / "audio.wav", clip.sample_rate_hz,
                                 clip.duration_seconds, clip.consent)


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return getattr(path.lstat(), "st_reparse_tag", None) == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)
    except FileNotFoundError:
        return False


def _output_parent(paths: Paths) -> Path:
    root = paths.root.resolve(strict=True)
    interim = Path(os.path.abspath(paths.interim_data))
    raw = paths.raw_data.resolve()
    try:
        parts = interim.relative_to(root).parts
    except ValueError as error:
        raise AudioPersistenceError("invalid_output_path", "Interim data must stay inside the project root.") from error
    if not parts or interim.is_relative_to(raw) or raw.is_relative_to(interim):
        raise AudioPersistenceError("invalid_output_path", "Prepared output must be separate from raw data.")
    current = root
    for part in (*parts, "prepared"):
        current = current / part
        if _is_link(current):
            raise AudioPersistenceError("invalid_output_path", "Prepared output directories must not be symlinks or junctions.")
        current.mkdir(exist_ok=True)
    return current


def _pcm16(samples: NDArray[np.float32]) -> NDArray[np.int16]:
    if not np.isfinite(samples).all():
        raise AudioPersistenceError("invalid_samples", "Samples changed or became non-finite before encoding.")
    # Explicit rounding and saturation avoid platform-dependent float conversion.
    return np.rint(np.clip(samples, -1.0, 32767 / 32768) * 32768).astype(np.int16)


def _write_verified_wav(path: Path, samples: NDArray[np.float32], rate: int) -> int:
    with path.open("x+b") as stream:
        with sf.SoundFile(stream, mode="w", samplerate=rate, channels=samples.shape[1],
                          format="WAV", subtype="PCM_16", endian="LITTLE") as audio:
            for start in range(0, len(samples), 65_536):
                audio.write(_pcm16(samples[start:start + 65_536]))
        stream.flush()
        os.fsync(stream.fileno())
    with sf.SoundFile(path, mode="r") as audio:
        if (audio.format, audio.subtype, audio.samplerate, audio.channels, audio.frames) != (
            "WAV", "PCM_16", rate, samples.shape[1], len(samples)
        ):
            raise AudioPersistenceError("verification_failed", "Written WAV properties differ from the prepared signal.")
        for start in range(0, len(samples), 65_536):
            block = samples[start:start + 65_536]
            actual = audio.read(len(block), dtype="int16", always_2d=True)
            if not np.array_equal(actual, _pcm16(block)):
                raise AudioPersistenceError("verification_failed", "Written WAV samples differ from the encoded signal.")
    return path.stat().st_size


def _preflight(signal: PreparedSignal, clip: PreparedClipRecord, policy: PersistenceSettings) -> None:
    count, frames = 0, signal.num_frames
    for seconds in signal.settings.window_seconds:
        length, hop = signal.settings.window_sample_counts(seconds)
        remaining = signal.num_frames - length
        if signal.settings.tail_policy == "pad":
            group_count = 1 + max(0, (remaining + hop - 1) // hop)
        else:
            group_count = 0 if remaining < 0 else 1 + remaining // hop
        count += group_count
        frames += group_count * length
    if count > policy.max_windows:
        raise AudioPersistenceError("too_many_windows", "Planned window count exceeds persistence.max_windows.")
    # Conservative headers/JSON allowance, plus exact planned PCM payload size.
    estimate = frames * signal.channels * 2 + (count + 1) * 128 + count * 1024 + 16_384
    estimate += len(clip.model_dump_json().encode("utf-8"))
    estimate += len(signal.settings.model_dump_json().encode("utf-8"))
    if estimate > policy.max_output_bytes:
        raise AudioPersistenceError("output_too_large", "Estimated bundle size exceeds persistence.max_output_bytes.")


def _tree_files(directory: Path) -> set[str]:
    if _is_link(directory) or not directory.is_dir():
        raise AudioPersistenceError("output_conflict", "An existing output path is not a regular directory.")
    names = set()
    pending = [directory]
    while pending:
        for path in pending.pop().iterdir():
            if _is_link(path):
                raise AudioPersistenceError("output_conflict", "Existing output contains a link or junction.")
            name = path.relative_to(directory).as_posix()
            if path.is_file():
                names.add(name)
            elif path.is_dir():
                names.add(name + "/")
                pending.append(path)
            else:
                raise AudioPersistenceError("output_conflict", "Existing output contains a non-regular entry.")
    return names


def _compare_existing(stage: Path, destination: Path) -> None:
    names = _tree_files(stage)
    if names != _tree_files(destination):
        raise AudioPersistenceError("output_conflict", "Existing output has missing or unexpected files; it was not overwritten.")
    for name in names:
        if name.endswith("/"):
            continue
        with (stage / name).open("rb") as expected, (destination / name).open("rb") as actual:
            while True:
                block = expected.read(1_048_576)
                if actual.read(1_048_576) != block:
                    raise AudioPersistenceError("output_conflict", "Existing output differs from the requested result; it was not overwritten.")
                if not block:
                    break


def _remove_stage(stage: Path, parent: Path) -> None:
    # Only remove the exact private staging directory created by this operation.
    if stage.parent != parent or not stage.name.startswith(".pending-") or _is_link(stage):
        raise AudioPersistenceError("cleanup_failed", "Refusing to clean an unexpected staging path.")
    if stage.resolve() != stage or not stage.is_relative_to(parent.resolve()):
        raise AudioPersistenceError("cleanup_failed", "Staging path no longer resolves inside the output directory.")
    shutil.rmtree(stage)


def save_prepared_audio(
    signal: PreparedSignal, paths: Paths, *, source_dataset: str,
    annotations: Sequence[EventAnnotation] = (), policy: PersistenceSettings | None = None,
    now: datetime | None = None,
) -> PersistedAudio:
    """Save a complete verified bundle, reusing identical existing output only.

    The caller must keep signal.samples unchanged during this synchronous call.
    Only prepared audio is saved; the original raw file is never copied or deleted.
    """
    policy = policy or PersistenceSettings()
    consent = validate_processing_consent(signal.consent, now if now is not None else datetime.now(UTC))
    signal = replace(signal, consent=consent, source=SourceAudioMetadata.model_validate(signal.source.model_dump()))
    windows = iter_windows(signal, now=now)  # Eager validation, lazy sample allocation.
    key = prepared_output_key(signal)
    relative_root = PurePosixPath("prepared") / key
    clip = PreparedClipRecord(
        clip_id=signal.clip_id, source_dataset=source_dataset,
        audio_path=str(relative_root / "audio.wav"), sample_rate_hz=signal.sample_rate_hz,
        duration_seconds=signal.duration_seconds, consent=consent,
        annotations=[EventAnnotation.model_validate(item.model_dump()) for item in annotations],
    )
    _preflight(signal, clip, policy)
    stage = None
    try:
        parent = _output_parent(paths)
        destination = parent / key
        if _is_link(destination) or (destination.exists() and not destination.is_dir()):
            raise AudioPersistenceError("output_conflict", "Existing destination is not a regular output directory.")
        stage = Path(mkdtemp(prefix=".pending-", dir=parent)).resolve()
        used_bytes = _write_verified_wav(stage / "audio.wav", signal.samples, signal.sample_rate_hz)
        records = []
        for window in windows:
            relative = PurePosixPath(window.record.audio_path).relative_to(relative_root)
            path = stage.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            used_bytes += _write_verified_wav(path, window.samples, signal.sample_rate_hz)
            if used_bytes > policy.max_output_bytes:
                raise AudioPersistenceError("output_too_large", "Written WAV files exceed persistence.max_output_bytes.")
            records.append(window.record)
        manifest = PreparedAudioManifest(
            source=signal.source, settings=signal.settings, clip=clip,
            channels=signal.channels, num_frames=signal.num_frames, windows=tuple(records),
        )
        document = (json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        if used_bytes + len(document) > policy.max_output_bytes:
            raise AudioPersistenceError("output_too_large", "Complete bundle exceeds persistence.max_output_bytes.")
        manifest_path = stage / "manifest.json"
        with manifest_path.open("xb") as output:
            output.write(document)
            output.flush()
            os.fsync(output.fileno())
        if PreparedAudioManifest.model_validate_json(manifest_path.read_bytes()) != manifest:
            raise AudioPersistenceError("verification_failed", "Saved manifest did not round-trip correctly.")
        validate_processing_consent(consent, now if now is not None else datetime.now(UTC))
        reused = False
        if destination.exists():
            _compare_existing(stage, destination)
            reused = True
        else:
            try:
                stage.rename(destination)
                stage = None
            except OSError as error:
                # Another cooperating writer may have published its complete bundle.
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) and not isinstance(error, FileExistsError):
                    raise
                _compare_existing(stage, destination)
                reused = True
        if reused:
            validate_processing_consent(consent, now if now is not None else datetime.now(UTC))
        return PersistedAudio(destination, destination / "manifest.json", manifest, reused)
    except AudioPersistenceError:
        raise
    except (OSError, sf.SoundFileError) as error:
        raise AudioPersistenceError("write_failed", "Prepared output could not be written or verified.") from error
    finally:
        windows.close()
        if stage is not None:
            _remove_stage(stage, parent)
