"""A3.2 bounded YAMNet inference over verified prepared waveform windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
from io import BytesIO
import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
import soundfile as sf

from audio_sentinel.acoustic_loader import AcousticModelMetadata, LoadedAcousticModel, TensorContract
from audio_sentinel.acoustic_model import LABEL_MAPPING_VERSION, YAMNET, load_class_map
from audio_sentinel.audio_loader import validate_processing_consent
from audio_sentinel.config import Paths
from audio_sentinel.feature_persistence import _read_bytes, _safe_file
from audio_sentinel.preparation import PreparedAudioManifest, PreparedWindowRecord


class AcousticInferenceError(RuntimeError):
    """A stable inference failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class AcousticInferenceSettings(BaseModel):
    """Resource limits for one manifest inference call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_manifest_bytes: int = Field(default=16_777_216, gt=0, strict=True)
    max_window_bytes: int = Field(default=268_435_456, gt=0, strict=True)
    max_decoded_bytes: int = Field(default=268_435_456, gt=0, strict=True)
    max_model_output_bytes: int = Field(default=536_870_912, gt=0, strict=True)
    max_windows: int = Field(default=10_000, ge=0, strict=True)


@dataclass(frozen=True)
class AcousticWindowInference:
    """Raw scores and provenance for one prepared window; no event decision."""

    window: PreparedWindowRecord
    window_audio_sha256: str
    input_num_samples: int
    scores: NDArray[np.float32] = field(repr=False, compare=False)
    embeddings_shape: tuple[int, int]
    spectrogram_shape: tuple[int, int]

    @property
    def patch_count(self) -> int:
        return self.scores.shape[0]

    @property
    def patch_spans_samples(self) -> tuple[tuple[int, int], ...]:
        """Return absolute prepared-clip support, clipped to real audio."""
        return tuple(
            (
                self.window.start_sample + index * YAMNET.patch_hop_samples,
                min(
                    self.window.start_sample + index * YAMNET.patch_hop_samples
                    + YAMNET.patch_support_samples,
                    self.window.end_sample,
                ),
            )
            for index in range(self.patch_count)
        )


@dataclass(frozen=True)
class AcousticInferenceResult:
    """In-memory raw model evidence for one immutable manifest snapshot."""

    model: AcousticModelMetadata
    preparation_manifest_path: str
    preparation_manifest_sha256: str
    raw_audio_sha256: str
    clip_id: str
    windows: tuple[AcousticWindowInference, ...]

    @property
    def patch_count(self) -> int:
        return sum(window.patch_count for window in self.windows)

    def to_summary(self) -> dict[str, object]:
        """Return JSON-ready inventory data without copying score payloads."""
        return {
            "clip_id": self.clip_id,
            "model": self.model.as_dict(),
            "preparation_manifest_path": self.preparation_manifest_path,
            "preparation_manifest_sha256": self.preparation_manifest_sha256,
            "raw_audio_sha256": self.raw_audio_sha256,
            "window_count": len(self.windows),
            "patch_count": self.patch_count,
            "windows": [
                {
                    "window_id": item.window.window_id,
                    "window_audio_sha256": item.window_audio_sha256,
                    "input_num_samples": item.input_num_samples,
                    "score_shape": item.scores.shape,
                    "embeddings_shape": item.embeddings_shape,
                    "spectrogram_shape": item.spectrogram_shape,
                    "patch_spans_samples": item.patch_spans_samples,
                }
                for item in self.windows
            ],
        }


def expected_yamnet_patches(num_samples: int) -> int:
    """Return the v1 patch count after YAMNet's documented right padding."""
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    additional = max(0, num_samples - YAMNET.patch_support_samples)
    return 1 + (additional + YAMNET.patch_hop_samples - 1) // YAMNET.patch_hop_samples


def _clock(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(UTC)


def _manifest(paths: Paths, relative: str | Path,
              policy: AcousticInferenceSettings) -> tuple[Path, bytes, PreparedAudioManifest]:
    try:
        path = _safe_file(paths.root, paths.interim_data, relative)
        document = _read_bytes(path, policy.max_manifest_bytes)
        return path, document, PreparedAudioManifest.model_validate_json(document)
    except AcousticInferenceError:
        raise
    except Exception as error:
        code = getattr(error, "code", "invalid_manifest")
        raise AcousticInferenceError(code, "The prepared-audio manifest could not be verified.") from error


def _window_bytes(paths: Paths, window: PreparedWindowRecord,
                  policy: AcousticInferenceSettings) -> tuple[bytes, str]:
    try:
        path = _safe_file(paths.root, paths.interim_data, window.audio_path)
        document = _read_bytes(path, policy.max_window_bytes)
        return document, hashlib.sha256(document).hexdigest()
    except Exception as error:
        code = getattr(error, "code", "invalid_window")
        raise AcousticInferenceError(code, "A listed prepared window could not be verified.") from error


def _decode_window(document: bytes, manifest: PreparedAudioManifest,
                   window: PreparedWindowRecord, policy: AcousticInferenceSettings) -> NDArray[np.float32]:
    num_samples = window.end_sample - window.start_sample + window.padding_samples
    if num_samples * 4 > policy.max_decoded_bytes:
        raise AcousticInferenceError("decoded_audio_too_large", "A window exceeds the decoded-sample budget.")
    try:
        with sf.SoundFile(BytesIO(document)) as audio:
            if (audio.format, audio.subtype, audio.samplerate, audio.channels, audio.frames) != (
                "WAV", "PCM_16", manifest.clip.sample_rate_hz, 1, num_samples
            ):
                raise AcousticInferenceError(
                    "source_mismatch", "Window WAV properties differ from the preparation manifest."
                )
            samples = audio.read(num_samples, dtype="float32", always_2d=False)
            if samples.shape != (num_samples,) or len(audio.read(1)):
                raise AcousticInferenceError("source_mismatch", "Window length differs from the preparation manifest.")
    except AcousticInferenceError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise AcousticInferenceError("decode_failed", "A prepared window could not be decoded.") from error
    if not np.isfinite(samples).all() or np.any(samples < -1) or np.any(samples > 1):
        raise AcousticInferenceError("invalid_samples", "Decoded window samples must be finite and within [-1, 1].")
    if window.padding_samples:
        if np.any(samples[-window.padding_samples:]):
            raise AcousticInferenceError("source_mismatch", "Prepared tail padding contains nonzero samples.")
        samples = samples[:-window.padding_samples]
    if samples.shape != (window.end_sample - window.start_sample,):
        raise AcousticInferenceError("source_mismatch", "Real window length differs from its recorded span.")
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    samples.setflags(write=False)
    return samples


def _array(value: object, name: str) -> NDArray[np.float32]:
    try:
        raw = value.numpy() if hasattr(value, "numpy") else value
        result = np.asarray(raw)
    except (MemoryError, TypeError, ValueError) as error:
        raise AcousticInferenceError("invalid_output", f"YAMNet {name} output could not be read.") from error
    if result.dtype != np.dtype("float32") or result.ndim != 2 or not np.isfinite(result).all():
        raise AcousticInferenceError("invalid_output", f"YAMNet {name} must be a finite float32 matrix.")
    return result


def _run_model(model: object, waveform: NDArray[np.float32], policy: AcousticInferenceSettings,
               remaining_bytes: int) -> tuple[NDArray[np.float32], tuple[int, int], tuple[int, int], int]:
    patches = expected_yamnet_patches(len(waveform))
    frames = YAMNET.patch_frames + (patches - 1) * (YAMNET.patch_frames // 2)
    expected_bytes = 4 * (patches * (YAMNET.num_classes + 1024) + frames * YAMNET.mel_bands)
    if expected_bytes > min(policy.max_model_output_bytes, remaining_bytes):
        raise AcousticInferenceError("output_too_large", "Predicted YAMNet outputs exceed the inference budget.")
    try:
        outputs = model(waveform)
    except AcousticInferenceError:
        raise
    except Exception as error:
        raise AcousticInferenceError("model_failed", "YAMNet failed while processing a prepared window.") from error
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise AcousticInferenceError("invalid_output", "YAMNet must return scores, embeddings, and a spectrogram.")
    scores, embeddings, spectrogram = (
        _array(outputs[0], "scores"), _array(outputs[1], "embeddings"), _array(outputs[2], "spectrogram")
    )
    if scores.shape != (patches, YAMNET.num_classes):
        raise AcousticInferenceError("invalid_output", "YAMNet score shape differs from the expected patch grid.")
    if embeddings.shape != (patches, 1024):
        raise AcousticInferenceError("invalid_output", "YAMNet embedding shape differs from its score grid.")
    if spectrogram.shape != (frames, YAMNET.mel_bands):
        raise AcousticInferenceError("invalid_output", "YAMNet spectrogram shape differs from its padded frame grid.")
    if np.any(scores < 0) or np.any(scores > 1):
        raise AcousticInferenceError("invalid_output", "YAMNet sigmoid scores must stay within [0, 1].")
    actual_bytes = scores.nbytes + embeddings.nbytes + spectrogram.nbytes
    if actual_bytes != expected_bytes:
        raise AcousticInferenceError("invalid_output", "YAMNet output byte size differs from its verified shapes.")
    owned_scores = np.array(scores, dtype=np.float32, order="C", copy=True)
    owned_scores.setflags(write=False)
    return owned_scores, embeddings.shape, spectrogram.shape, actual_bytes


def _validate_loaded_model(loaded: LoadedAcousticModel) -> None:
    metadata = loaded.metadata
    expected_outputs = (
        TensorContract("output_0", (None, YAMNET.num_classes), "float32"),
        TensorContract("output_1", (None, 1024), "float32"),
        TensorContract("output_2", (None, YAMNET.mel_bands), "float32"),
    )
    if (
        metadata.model_id != YAMNET.model_id
        or metadata.model_version != "1"
        or metadata.model_handle != YAMNET.model_handle
        or metadata.artifact_sha256 != YAMNET.artifact_sha256
        or metadata.class_map_sha256 != YAMNET.class_map_sha256
        or metadata.label_mapping_version != LABEL_MAPPING_VERSION
        or metadata.runtime_distribution != YAMNET.runtime_distribution
        or metadata.runtime_version != YAMNET.runtime_version
        or metadata.signature_name != "serving_default"
        or metadata.input != TensorContract("waveform", (None,), "float32")
        or metadata.outputs != expected_outputs
        or metadata.num_classes != YAMNET.num_classes
        or loaded.classes != load_class_map()
        or not callable(loaded.model)
    ):
        raise AcousticInferenceError("model_mismatch", "Inference requires the verified pinned YAMNet v1 model.")


def infer_prepared_audio(
    paths: Paths, manifest_path: str | Path, loaded: LoadedAcousticModel, *,
    policy: AcousticInferenceSettings | None = None, now: datetime | None = None,
) -> AcousticInferenceResult:
    """Run raw YAMNet inference in manifest order without aggregating events."""
    policy = AcousticInferenceSettings.model_validate((policy or AcousticInferenceSettings()).model_dump())
    _validate_loaded_model(loaded)
    path, manifest_bytes, manifest = _manifest(paths, manifest_path, policy)
    validate_processing_consent(manifest.clip.consent, _clock(now))
    if manifest.clip.sample_rate_hz != YAMNET.sample_rate_hz:
        raise AcousticInferenceError("sample_rate_mismatch", "YAMNet requires prepared 16 kHz audio.")
    if manifest.channels != 1:
        raise AcousticInferenceError("mono_required", "YAMNet requires prepared mono audio.")
    if len(manifest.windows) > policy.max_windows:
        raise AcousticInferenceError("too_many_windows", "Manifest window count exceeds the inference limit.")

    relative_manifest = path.relative_to(Path(os.path.abspath(paths.interim_data))).as_posix()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    results: list[AcousticWindowInference] = []
    used_output_bytes = 0
    try:
        for window in manifest.windows:
            validate_processing_consent(manifest.clip.consent, _clock(now))
            audio_bytes, audio_sha256 = _window_bytes(paths, window, policy)
            waveform = _decode_window(audio_bytes, manifest, window, policy)
            scores, embedding_shape, spectrogram_shape, output_bytes = _run_model(
                loaded.model, waveform, policy, policy.max_model_output_bytes - used_output_bytes
            )
            used_output_bytes += output_bytes
            validate_processing_consent(manifest.clip.consent, _clock(now))
            results.append(AcousticWindowInference(
                window=window,
                window_audio_sha256=audio_sha256,
                input_num_samples=len(waveform),
                scores=scores,
                embeddings_shape=embedding_shape,
                spectrogram_shape=spectrogram_shape,
            ))
    except MemoryError as error:
        raise AcousticInferenceError("insufficient_memory", "Not enough memory to run YAMNet inference.") from error

    # Do not return a mixed snapshot if manifest/window files changed mid-run.
    final_path, final_manifest_bytes, _ = _manifest(paths, relative_manifest, policy)
    if final_path != path or final_manifest_bytes != manifest_bytes:
        raise AcousticInferenceError("source_changed", "Preparation manifest changed during inference.")
    for result in results:
        _, current_sha256 = _window_bytes(paths, result.window, policy)
        if current_sha256 != result.window_audio_sha256:
            raise AcousticInferenceError("source_changed", "A prepared window changed during inference.")
    validate_processing_consent(manifest.clip.consent, _clock(now))
    return AcousticInferenceResult(
        model=loaded.metadata,
        preparation_manifest_path=relative_manifest,
        preparation_manifest_sha256=manifest_sha256,
        raw_audio_sha256=manifest.source.sha256,
        clip_id=manifest.clip.clip_id,
        windows=tuple(results),
    )
