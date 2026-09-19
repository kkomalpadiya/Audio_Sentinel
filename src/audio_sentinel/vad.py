"""B4.1 verified Silero VAD v6 loader and raw frame-probability wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from importlib import import_module
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from audio_sentinel.config import Paths
from audio_sentinel.persistence import _is_link
from audio_sentinel.speech_contracts import SpeechModelDescriptor


MODEL_RELATIVE_PATH = PurePosixPath("silero-vad/6")
MODEL_FILENAME = "silero_vad_v6.onnx"
INSTALL_MARKER = ".complete.json"
MAX_MARKER_BYTES = 16_384
MAX_MODEL_BYTES = 5_000_000


@dataclass(frozen=True)
class SileroVadSpec:
    model_id: str = "silero-vad"
    model_version: str = "6.0"
    model_source: str = "https://github.com/snakers4/silero-vad/releases/tag/v6.0"
    source_distribution: str = "faster-whisper"
    source_distribution_version: str = "1.2.1"
    source_asset: str = "faster_whisper/assets/silero_vad_v6.onnx"
    source_distribution_sha256: str = (
        "79a66ad50688c0b794dd501dc340a736992a6342f7f95e5811be60b5224a26a7"
    )
    artifact_sha256: str = "4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2"
    artifact_size_bytes: int = 1_245_151
    runtime_distribution: str = "onnxruntime"
    runtime_version: str = "1.23.2"
    sample_rate_hz: int = 16_000
    frame_samples: int = 512
    context_samples: int = 64
    state_size: int = 128

    def marker(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "source_asset": self.source_asset,
            "source_distribution": self.source_distribution,
            "source_distribution_sha256": self.source_distribution_sha256,
            "source_distribution_version": self.source_distribution_version,
        }


SILERO_VAD = SileroVadSpec()


class VadError(RuntimeError):
    """Stable VAD setup/inference failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class VadTensorContract:
    name: str
    shape: tuple[int | None, ...]
    dtype: str


@dataclass(frozen=True)
class VadModelMetadata:
    model_id: str
    model_version: str
    model_source: str
    model_path: str
    artifact_sha256: str
    source_distribution: str
    source_distribution_version: str
    runtime_distribution: str
    runtime_version: str
    providers: tuple[str, ...]
    inputs: tuple[VadTensorContract, ...]
    outputs: tuple[VadTensorContract, ...]
    sample_rate_hz: int
    frame_samples: int
    context_samples: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def as_speech_descriptor(self) -> SpeechModelDescriptor:
        return SpeechModelDescriptor(
            model_id=self.model_id,
            model_version=self.model_version,
            artifact_sha256=self.artifact_sha256,
            runtime_distribution=self.runtime_distribution,
            runtime_version=self.runtime_version,
        )


@dataclass(frozen=True)
class LoadedVadModel:
    metadata: VadModelMetadata
    session: object = field(repr=False, compare=False)


class VadInferenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_input_samples: int = Field(default=9_600_000, gt=0, strict=True)
    max_frames_per_call: int = Field(default=10_000, gt=0, le=100_000, strict=True)
    max_output_bytes: int = Field(default=16_777_216, gt=0, strict=True)


@dataclass(frozen=True)
class VadFrameProbability:
    frame_index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    speech_probability: float
    padding_samples: int


@dataclass(frozen=True)
class VadInferenceResult:
    """Raw VAD evidence for one independent waveform; no segment decision."""

    model: VadModelMetadata
    input_num_samples: int
    padded_num_samples: int
    frames: tuple[VadFrameProbability, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def to_summary(self) -> dict[str, object]:
        return {
            "model": self.model.as_dict(),
            "input_num_samples": self.input_num_samples,
            "padded_num_samples": self.padded_num_samples,
            "frame_count": self.frame_count,
            "frames": [asdict(frame) for frame in self.frames],
        }


def expected_vad_frames(num_samples: int) -> int:
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    return (num_samples + SILERO_VAD.frame_samples - 1) // SILERO_VAD.frame_samples


def _safe_model_directory(paths: Paths, relative_path: str | Path) -> Path:
    relative = PurePosixPath(str(relative_path).replace(os.sep, "/"))
    if relative.is_absolute() or not relative.parts or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise VadError("invalid_path", "VAD model path must stay inside models/ without traversal.")
    try:
        root = paths.root.resolve(strict=True)
        models = Path(os.path.abspath(paths.models))
        current = root
        for part in models.relative_to(root).parts + relative.parts:
            current = current / part
            if _is_link(current):
                raise ValueError("linked model path")
        resolved = current.resolve(strict=True)
        if resolved != current or not resolved.is_relative_to(models) or not resolved.is_dir():
            raise ValueError("model directory is not contained")
        return resolved
    except FileNotFoundError as error:
        raise VadError(
            "model_not_found", "Silero VAD v6 is missing; run scripts/setup_speech.ps1."
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, VadError):
            raise
        raise VadError(
            "invalid_path", "VAD model must be a regular local directory without links or junctions."
        ) from error


def _regular_file(directory: Path, relative: str, max_bytes: int) -> tuple[Path, os.stat_result]:
    try:
        current = directory
        for part in PurePosixPath(relative).parts:
            current = current / part
            if _is_link(current):
                raise ValueError("linked payload")
        resolved = current.resolve(strict=True)
        details = resolved.stat()
        if resolved != current or not stat.S_ISREG(details.st_mode) or details.st_size > max_bytes:
            raise ValueError("invalid payload")
        return resolved, details
    except (OSError, RuntimeError, ValueError) as error:
        raise VadError("invalid_artifact", "Silero VAD files are missing, linked, or invalid.") from error


def _artifact_inventory(directory: Path) -> tuple[Path, os.stat_result, dict[str, object]]:
    try:
        entries = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        }
    except (OSError, RuntimeError, ValueError) as error:
        raise VadError("invalid_artifact", "Silero VAD files could not be inventoried.") from error
    if entries != {MODEL_FILENAME, INSTALL_MARKER}:
        raise VadError("invalid_artifact", "Silero VAD must contain exactly its model and marker files.")
    model_path, model_stat = _regular_file(directory, MODEL_FILENAME, MAX_MODEL_BYTES)
    marker_path, _ = _regular_file(directory, INSTALL_MARKER, MAX_MARKER_BYTES)
    if model_stat.st_size != SILERO_VAD.artifact_size_bytes:
        raise VadError("invalid_artifact", "Silero VAD model size differs from the pinned v6 artifact.")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VadError("invalid_artifact", "Silero VAD installation marker is invalid.") from error
    if marker != SILERO_VAD.marker():
        raise VadError("invalid_artifact", "Silero VAD installation marker differs from the pinned source.")
    return model_path, model_stat, marker


def model_artifact_sha256(directory: Path) -> str:
    """Hash the verified ONNX file and detect changes during the read."""

    model_path, _, _ = _artifact_inventory(directory)
    digest = hashlib.sha256()
    try:
        with model_path.open("rb") as stream:
            initial = os.fstat(stream.fileno())
            while chunk := stream.read(1_048_576):
                digest.update(chunk)
            final = os.fstat(stream.fileno())
    except OSError as error:
        raise VadError("invalid_artifact", "Silero VAD model could not be read.") from error
    if (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) != (
        final.st_size,
        final.st_mtime_ns,
        final.st_ctime_ns,
    ):
        raise VadError("artifact_changed", "Silero VAD changed while it was verified.")
    return digest.hexdigest()


def _shape(value: object) -> tuple[int | None, ...]:
    return tuple(item if isinstance(item, int) else None for item in (getattr(value, "shape", ()) or ()))


def _tensor(value: object) -> VadTensorContract:
    return VadTensorContract(
        name=str(getattr(value, "name", "")),
        shape=_shape(value),
        dtype=str(getattr(value, "type", "")),
    )


def _verify_session(session: object) -> tuple[
    tuple[str, ...], tuple[VadTensorContract, ...], tuple[VadTensorContract, ...]
]:
    try:
        providers = tuple(session.get_providers())
        inputs = tuple(_tensor(item) for item in session.get_inputs())
        outputs = tuple(_tensor(item) for item in session.get_outputs())
    except (AttributeError, TypeError, ValueError) as error:
        raise VadError("signature_mismatch", "ONNX Runtime could not inspect Silero VAD v6.") from error
    expected_inputs = (
        VadTensorContract("input", (None, 576), "tensor(float)"),
        VadTensorContract("h", (1, 1, 128), "tensor(float)"),
        VadTensorContract("c", (1, 1, 128), "tensor(float)"),
    )
    expected_outputs = (
        VadTensorContract("speech_probs", (None,), "tensor(float)"),
        VadTensorContract("hn", (1, 1, 128), "tensor(float)"),
        VadTensorContract("cn", (1, 1, 128), "tensor(float)"),
    )
    if providers != ("CPUExecutionProvider",) or inputs != expected_inputs or outputs != expected_outputs:
        raise VadError("signature_mismatch", "Silero VAD provider or tensor contract differs from v6.")
    if not callable(getattr(session, "run", None)):
        raise VadError("signature_mismatch", "Silero VAD session has no callable inference entry point.")
    return providers, inputs, outputs


def load_silero_vad(
    paths: Paths, relative_path: str | Path = MODEL_RELATIVE_PATH
) -> LoadedVadModel:
    """Load and inspect the pinned local CPU model without running inference."""

    directory = _safe_model_directory(paths, relative_path)
    artifact_sha256 = model_artifact_sha256(directory)
    if artifact_sha256 != SILERO_VAD.artifact_sha256:
        raise VadError("artifact_mismatch", "Silero VAD digest differs from the pinned v6 artifact.")
    try:
        onnxruntime = import_module("onnxruntime")
    except (ImportError, OSError) as error:
        raise VadError(
            "runtime_missing", "ONNX Runtime is unavailable; run scripts/setup_speech.ps1."
        ) from error
    runtime_version = str(getattr(onnxruntime, "__version__", ""))
    if runtime_version != SILERO_VAD.runtime_version:
        raise VadError(
            "runtime_mismatch",
            f"Silero VAD requires {SILERO_VAD.runtime_distribution} {SILERO_VAD.runtime_version}.",
        )
    try:
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 4
        session = onnxruntime.InferenceSession(
            str(directory / MODEL_FILENAME),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        providers, inputs, outputs = _verify_session(session)
    except VadError:
        raise
    except Exception as error:
        raise VadError("model_load_failed", "ONNX Runtime could not load Silero VAD v6.") from error
    if model_artifact_sha256(directory) != artifact_sha256:
        raise VadError("artifact_changed", "Silero VAD changed while ONNX Runtime loaded it.")
    metadata = VadModelMetadata(
        model_id=SILERO_VAD.model_id,
        model_version=SILERO_VAD.model_version,
        model_source=SILERO_VAD.model_source,
        model_path=PurePosixPath(relative_path).as_posix(),
        artifact_sha256=artifact_sha256,
        source_distribution=SILERO_VAD.source_distribution,
        source_distribution_version=SILERO_VAD.source_distribution_version,
        runtime_distribution=SILERO_VAD.runtime_distribution,
        runtime_version=runtime_version,
        providers=providers,
        inputs=inputs,
        outputs=outputs,
        sample_rate_hz=SILERO_VAD.sample_rate_hz,
        frame_samples=SILERO_VAD.frame_samples,
        context_samples=SILERO_VAD.context_samples,
    )
    return LoadedVadModel(metadata=metadata, session=session)


def _validate_loaded_model(loaded: LoadedVadModel) -> None:
    expected_inputs = (
        VadTensorContract("input", (None, 576), "tensor(float)"),
        VadTensorContract("h", (1, 1, 128), "tensor(float)"),
        VadTensorContract("c", (1, 1, 128), "tensor(float)"),
    )
    expected_outputs = (
        VadTensorContract("speech_probs", (None,), "tensor(float)"),
        VadTensorContract("hn", (1, 1, 128), "tensor(float)"),
        VadTensorContract("cn", (1, 1, 128), "tensor(float)"),
    )
    metadata = loaded.metadata
    if (
        metadata.model_id != SILERO_VAD.model_id
        or metadata.model_version != SILERO_VAD.model_version
        or metadata.model_source != SILERO_VAD.model_source
        or metadata.model_path != MODEL_RELATIVE_PATH.as_posix()
        or metadata.artifact_sha256 != SILERO_VAD.artifact_sha256
        or metadata.source_distribution != SILERO_VAD.source_distribution
        or metadata.source_distribution_version != SILERO_VAD.source_distribution_version
        or metadata.runtime_distribution != SILERO_VAD.runtime_distribution
        or metadata.runtime_version != SILERO_VAD.runtime_version
        or metadata.providers != ("CPUExecutionProvider",)
        or metadata.inputs != expected_inputs
        or metadata.outputs != expected_outputs
        or metadata.sample_rate_hz != SILERO_VAD.sample_rate_hz
        or metadata.frame_samples != SILERO_VAD.frame_samples
        or metadata.context_samples != SILERO_VAD.context_samples
        or not callable(getattr(loaded.session, "run", None))
    ):
        raise VadError("model_mismatch", "VAD inference requires the verified pinned Silero v6 model.")


def _float_array(value: object, name: str, shape: tuple[int, ...]) -> NDArray[np.float32]:
    try:
        result = np.asarray(value)
    except (MemoryError, TypeError, ValueError) as error:
        raise VadError("invalid_output", f"Silero VAD {name} output could not be read.") from error
    if result.dtype != np.dtype("float32") or result.shape != shape or not np.isfinite(result).all():
        raise VadError("invalid_output", f"Silero VAD {name} must be finite float32 with shape {shape}.")
    return result


def infer_vad_probabilities(
    loaded: LoadedVadModel,
    waveform: NDArray[np.float32],
    *,
    settings: VadInferenceSettings | None = None,
) -> VadInferenceResult:
    """Return one raw speech probability per 512 samples for an independent 16 kHz waveform."""

    settings = VadInferenceSettings.model_validate((settings or VadInferenceSettings()).model_dump())
    _validate_loaded_model(loaded)
    if not isinstance(waveform, np.ndarray) or waveform.dtype != np.float32 or waveform.ndim != 1:
        raise VadError("invalid_samples", "VAD expects a one-dimensional float32 waveform.")
    if len(waveform) == 0:
        raise VadError("invalid_samples", "VAD waveform must contain at least one sample.")
    if len(waveform) > settings.max_input_samples:
        raise VadError("input_too_large", "VAD waveform exceeds max_input_samples.")
    if not np.isfinite(waveform).all() or np.any(waveform < -1) or np.any(waveform > 1):
        raise VadError("invalid_samples", "VAD waveform samples must be finite and within [-1, 1].")

    frame_count = expected_vad_frames(len(waveform))
    if frame_count * 4 > settings.max_output_bytes:
        raise VadError("output_too_large", "VAD probabilities exceed max_output_bytes.")
    padded_num_samples = frame_count * SILERO_VAD.frame_samples
    try:
        padded = np.zeros(padded_num_samples, dtype=np.float32)
        padded[: len(waveform)] = waveform
        blocks = padded.reshape(frame_count, SILERO_VAD.frame_samples)
        contexts = np.zeros((frame_count, SILERO_VAD.context_samples), dtype=np.float32)
        if frame_count > 1:
            contexts[1:] = blocks[:-1, -SILERO_VAD.context_samples :]
        model_input = np.concatenate((contexts, blocks), axis=1)
        h = np.zeros((1, 1, SILERO_VAD.state_size), dtype=np.float32)
        c = np.zeros((1, 1, SILERO_VAD.state_size), dtype=np.float32)
        probabilities: list[NDArray[np.float32]] = []
        for start in range(0, frame_count, settings.max_frames_per_call):
            batch = model_input[start : start + settings.max_frames_per_call]
            try:
                outputs = loaded.session.run(None, {"input": batch, "h": h, "c": c})
            except Exception as error:
                raise VadError("model_failed", "Silero VAD failed while scoring a waveform.") from error
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
                raise VadError("invalid_output", "Silero VAD must return probabilities and two states.")
            probabilities.append(_float_array(outputs[0], "speech_probs", (len(batch),)))
            h = _float_array(outputs[1], "hn", (1, 1, SILERO_VAD.state_size))
            c = _float_array(outputs[2], "cn", (1, 1, SILERO_VAD.state_size))
        scores = np.concatenate(probabilities)
    except VadError:
        raise
    except MemoryError as error:
        raise VadError("insufficient_memory", "Not enough memory to prepare VAD inference.") from error
    if scores.shape != (frame_count,) or np.any(scores < 0) or np.any(scores > 1):
        raise VadError("invalid_output", "Silero VAD probabilities must stay within [0, 1].")

    frames = []
    for index, probability in enumerate(scores):
        start_sample = index * SILERO_VAD.frame_samples
        end_sample = min(start_sample + SILERO_VAD.frame_samples, len(waveform))
        frames.append(VadFrameProbability(
            frame_index=index,
            start_sample=start_sample,
            end_sample=end_sample,
            start_seconds=start_sample / SILERO_VAD.sample_rate_hz,
            end_seconds=end_sample / SILERO_VAD.sample_rate_hz,
            speech_probability=float(probability),
            padding_samples=SILERO_VAD.frame_samples - (end_sample - start_sample),
        ))
    return VadInferenceResult(
        model=loaded.metadata,
        input_num_samples=len(waveform),
        padded_num_samples=padded_num_samples,
        frames=tuple(frames),
    )
