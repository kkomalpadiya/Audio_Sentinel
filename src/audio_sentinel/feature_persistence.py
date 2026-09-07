"""A2.2 verified feature bundles for individual saved preparation windows."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import errno
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
import soundfile as sf

from audio_sentinel.audio_loader import validate_processing_consent
from audio_sentinel.config import Paths
from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.features import LogMelFeatureMetadata, LogMelSource
from audio_sentinel.log_mel import generate_log_mel
from audio_sentinel.persistence import _is_link, _remove_stage
from audio_sentinel.preparation import PreparedAudioManifest, validate_relative_audio_path


class FeaturePersistenceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class FeaturePersistenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_metadata_bytes: int = Field(default=16_777_216, gt=0, strict=True)
    max_window_bytes: int = Field(default=268_435_456, gt=0, strict=True)
    max_decoded_bytes: int = Field(default=268_435_456, gt=0, strict=True)
    max_output_bytes: int = Field(default=536_870_912, gt=0, strict=True)
    max_working_bytes: int = Field(default=268_435_456, gt=0, strict=True)


@dataclass(frozen=True)
class SavedLogMel:
    directory: Path
    metadata_path: Path
    metadata: LogMelFeatureMetadata
    reused: bool

    @property
    def feature_path(self) -> Path:
        return self.directory / "features.npy"


@dataclass(frozen=True)
class LoadedLogMel:
    metadata: LogMelFeatureMetadata
    values: NDArray[np.float32] = field(repr=False, compare=False)


def _clock(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(UTC)


def _safe_file(root: Path, managed: Path, relative: str | Path) -> Path:
    """Resolve an existing regular file; reject links along every managed component."""
    relative = str(relative).replace(os.sep, "/") if isinstance(relative, Path) else relative
    try:
        validate_relative_audio_path(relative)
        root = root.resolve(strict=True)
        base = Path(os.path.abspath(managed))
        parts = base.relative_to(root).parts + tuple(relative.split("/"))
        current = root
        for part in parts:
            current = current / part
            if _is_link(current):
                raise ValueError("linked path")
        resolved = current.resolve(strict=True)
        if resolved != current or not resolved.is_relative_to(base) or not resolved.is_file():
            raise ValueError("not a contained regular file")
        return resolved
    except (ValueError, OSError, RuntimeError) as error:
        raise FeaturePersistenceError("invalid_path", "Expected a regular file inside the managed directory, without links.") from error


def _read_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size > limit:
            raise FeaturePersistenceError("file_too_large", "File exceeds its configured byte limit.")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise FeaturePersistenceError("file_too_large", "File exceeds its configured byte limit.")
        return data


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(paths: Paths, manifest_path: str | Path, window_id: str,
            policy: FeaturePersistenceSettings, now: datetime | None,
            *, decode: bool) -> tuple[LogMelSource, NDArray[np.float32] | None]:
    path = _safe_file(paths.root, paths.interim_data, manifest_path)
    document = _read_bytes(path, policy.max_metadata_bytes)
    manifest = PreparedAudioManifest.model_validate_json(document)
    validate_processing_consent(manifest.clip.consent, _clock(now))
    window = next((w for w in manifest.windows if w.window_id == window_id), None)
    if window is None:
        raise FeaturePersistenceError("window_not_found", "Window is not listed in the preparation manifest.")
    if manifest.channels != 1:
        raise FeaturePersistenceError("mono_required", "Feature storage requires prepared mono windows.")
    audio_path = _safe_file(paths.root, paths.interim_data, window.audio_path)
    audio_bytes = _read_bytes(audio_path, policy.max_window_bytes)
    source = LogMelSource(
        preparation_manifest_path=path.relative_to(Path(os.path.abspath(paths.interim_data))).as_posix(),
        preparation_manifest_sha256=_hash(document), raw_audio_sha256=manifest.source.sha256,
        window_audio_sha256=_hash(audio_bytes), clip_id=manifest.clip.clip_id,
        sample_rate_hz=manifest.clip.sample_rate_hz, window=window,
    )
    source.validate_against(manifest)
    samples = None
    if decode:
        if source.num_samples * 4 > policy.max_decoded_bytes:
            raise FeaturePersistenceError("decoded_audio_too_large", "Window exceeds the decoded sample budget.")
        with sf.SoundFile(BytesIO(audio_bytes)) as audio:
            if (audio.format, audio.subtype, audio.samplerate, audio.channels, audio.frames) != (
                "WAV", "PCM_16", source.sample_rate_hz, 1, source.num_samples
            ):
                raise FeaturePersistenceError("source_mismatch", "Window WAV properties differ from its preparation record.")
            samples = audio.read(source.num_samples, dtype="float32", always_2d=True)
            if samples.shape != (source.num_samples, 1) or len(audio.read(1)):
                raise FeaturePersistenceError("source_mismatch", "Window decoded length differs from its record.")
        if window.padding_samples and np.any(samples[-window.padding_samples:]):
            raise FeaturePersistenceError("source_mismatch", "Prepared tail padding contains nonzero samples.")
    validate_processing_consent(manifest.clip.consent, _clock(now))
    return source, samples


def _assert_source_current(paths: Paths, expected: LogMelSource,
                           policy: FeaturePersistenceSettings, now: datetime | None) -> None:
    current, _ = _source(paths, expected.preparation_manifest_path, expected.window.window_id,
                         policy, now, decode=False)
    if current != expected:
        raise FeaturePersistenceError("source_changed", "Preparation manifest or window changed; features cannot be reused.")


def _output_parent(paths: Paths) -> Path:
    root = paths.root.resolve(strict=True)
    processed = Path(os.path.abspath(paths.processed_data))
    try:
        parts = processed.relative_to(root).parts
        if not parts:
            raise ValueError("root is not an output directory")
        for other in (paths.raw_data, paths.interim_data):
            other = other.resolve()
            if processed.is_relative_to(other) or other.is_relative_to(processed):
                raise ValueError("output overlaps source data")
        current = root
        for part in (*parts, "log-mel"):
            current = current / part
            if _is_link(current):
                raise ValueError("linked output")
            current.mkdir(exist_ok=True)
        return current.resolve(strict=True)
    except (ValueError, OSError) as error:
        raise FeaturePersistenceError("invalid_output_path", "Feature output must be separate from raw/interim data without linked directories.") from error


def _key(source: LogMelSource, recipe: LogMelSettings, extractor_version: str) -> str:
    identity = dict(source=source.model_dump(mode="json"), settings=recipe.model_dump(mode="json"),
                    extractor_name="librosa", extractor_version=extractor_version, implementation_version="1.0")
    return _hash(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _read_npy(path: Path, metadata: LogMelFeatureMetadata, policy: FeaturePersistenceSettings) -> NDArray[np.float32]:
    # Parse and validate the bounded header before allocating from untrusted shape.
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > policy.max_output_bytes:
            raise FeaturePersistenceError("file_too_large", "Feature file exceeds output byte limit.")
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise FeaturePersistenceError("invalid_features", "Unsupported NPY version.")
        if shape != metadata.shape or fortran or dtype != np.dtype("<f4"):
            raise FeaturePersistenceError("invalid_features", "NPY shape, order or dtype differs from metadata.")
        payload = shape[0] * shape[1] * 4
        if payload > metadata.settings.max_feature_bytes or size != stream.tell() + payload:
            raise FeaturePersistenceError("invalid_features", "NPY payload is oversized, truncated or has trailing bytes.")
        values = np.frombuffer(stream.read(payload), dtype="<f4").reshape(shape).copy(order="C")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata.feature_sha256 or not np.isfinite(values).all():
        raise FeaturePersistenceError("invalid_features", "Feature hash differs or samples are non-finite.")
    return values


def _load(paths: Paths, metadata_path: str | Path, policy: FeaturePersistenceSettings,
          now: datetime | None) -> LoadedLogMel:
    path = _safe_file(paths.root, paths.processed_data, metadata_path)
    metadata = LogMelFeatureMetadata.model_validate_json(_read_bytes(path, policy.max_metadata_bytes))
    expected_key = _key(metadata.source, metadata.settings, metadata.extractor_version)
    expected_relative = f"log-mel/{expected_key}/features.npy"
    if metadata.feature_path != expected_relative or path != paths.processed_data / f"log-mel/{expected_key}/metadata.json":
        raise FeaturePersistenceError("output_conflict", "Metadata identity does not match the bundle path.")
    feature_path = _safe_file(paths.root, paths.processed_data, metadata.feature_path)
    if {p.name for p in path.parent.iterdir()} != {"features.npy", "metadata.json"}:
        raise FeaturePersistenceError("output_conflict", "Bundle contains missing or unexpected entries.")
    if path.stat().st_size + feature_path.stat().st_size > policy.max_output_bytes:
        raise FeaturePersistenceError("file_too_large", "Bundle exceeds output byte limit.")
    _assert_source_current(paths, metadata.source, policy, now)
    values = _read_npy(feature_path, metadata, policy)
    _assert_source_current(paths, metadata.source, policy, now)
    return LoadedLogMel(metadata, values)


def load_log_mel(paths: Paths, metadata_path: str | Path, *,
                 policy: FeaturePersistenceSettings | None = None, now: datetime | None = None) -> LoadedLogMel:
    """Reload a bundle by metadata path relative to processed_data, with source checks."""
    policy = FeaturePersistenceSettings.model_validate((policy or FeaturePersistenceSettings()).model_dump())
    try:
        return _load(paths, metadata_path, policy, now)
    except (OSError, EOFError) as error:
        raise FeaturePersistenceError("read_failed", "Feature bundle could not be read.") from error
    except MemoryError as error:
        raise FeaturePersistenceError("insufficient_memory", "Not enough memory to reload features.") from error


def save_window_log_mel(paths: Paths, manifest_path: str | Path, window_id: str, settings: LogMelSettings,
                        *, policy: FeaturePersistenceSettings | None = None,
                        now: datetime | None = None) -> SavedLogMel:
    """Decode a listed window, generate and save verified NPY + JSON as one bundle.

    Input manifest path is relative to interim_data. No raw audio is modified.
    Repeats regenerate and compare, preserving original metadata creation time.
    """
    policy = FeaturePersistenceSettings.model_validate((policy or FeaturePersistenceSettings()).model_dump())
    stage = None
    try:
        source, samples = _source(paths, manifest_path, window_id, policy, now, decode=True)
        result = generate_log_mel(samples, source.sample_rate_hz, settings, max_working_bytes=policy.max_working_bytes)
        del samples
        if result.values.nbytes + 10_000 > policy.max_output_bytes:
            raise FeaturePersistenceError("output_too_large", "Feature payload plus header allowance exceeds output budget.")
        key = _key(source, result.settings, result.extractor_version)
        parent = _output_parent(paths)
        destination = parent / key
        relative = f"log-mel/{key}/metadata.json"
        if _is_link(destination) or (destination.exists() and not destination.is_dir()):
            raise FeaturePersistenceError("output_conflict", "Destination is not a regular bundle directory.")
        stage = Path(mkdtemp(prefix=".pending-", dir=parent)).resolve()
        feature_path = stage / "features.npy"
        with feature_path.open("xb") as stream:
            np.save(stream, result.values.astype("<f4", copy=False), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        with feature_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = LogMelFeatureMetadata(
            source=source, settings=result.settings, feature_path=f"log-mel/{key}/features.npy",
            feature_sha256=digest, shape=result.values.shape,
            analysis_padding_samples=result.analysis_padding_samples,
            extractor_version=result.extractor_version, created_at=_clock(now),
        )
        if not np.array_equal(_read_npy(feature_path, metadata, policy), result.values):
            raise FeaturePersistenceError("verification_failed", "Saved array differs from generated features.")
        document = (metadata.model_dump_json(indent=2) + "\n").encode()
        if len(document) > policy.max_metadata_bytes or len(document) + feature_path.stat().st_size > policy.max_output_bytes:
            raise FeaturePersistenceError("output_too_large", "Complete feature bundle exceeds output limits.")
        with (stage / "metadata.json").open("xb") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        if LogMelFeatureMetadata.model_validate_json((stage / "metadata.json").read_bytes()) != metadata:
            raise FeaturePersistenceError("verification_failed", "Metadata failed readback.")
        _assert_source_current(paths, source, policy, now)
        reused = destination.exists()
        if not reused:
            try:
                stage.rename(destination)
                stage = None
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) and not isinstance(error, FileExistsError):
                    raise
                reused = True
        if reused:
            existing = _load(paths, relative, policy, now)
            comparable = existing.metadata.model_copy(update={"created_at": metadata.created_at})
            if comparable != metadata or not np.array_equal(existing.values, result.values):
                raise FeaturePersistenceError("output_conflict", "Existing output differs; it was not overwritten.")
            metadata = existing.metadata
        return SavedLogMel(destination, destination / "metadata.json", metadata, reused)
    except (OSError, sf.SoundFileError) as error:
        raise FeaturePersistenceError("write_failed", "Source or feature output could not be read, written or verified.") from error
    except MemoryError as error:
        raise FeaturePersistenceError("insufficient_memory", "Not enough memory to persist features.") from error
    finally:
        if stage is not None:
            _remove_stage(stage, parent)
