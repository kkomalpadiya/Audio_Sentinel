"""B3.1 local-only YAMNet SavedModel loading and version verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from importlib import import_module
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping

from audio_sentinel.acoustic_model import (
    LABEL_MAPPING_VERSION,
    YAMNET,
    AcousticClass,
    load_class_map,
    validate_label_mapping,
)
from audio_sentinel.config import Paths
from audio_sentinel.persistence import _is_link


MODEL_RELATIVE_PATH = PurePosixPath("yamnet/1")
MODEL_FILES = (
    "assets/yamnet_class_map.csv",
    "saved_model.pb",
    "variables/variables.data-00000-of-00001",
    "variables/variables.index",
)
DOWNLOAD_MARKER = ".complete/models/google/yamnet/tensorFlow2/yamnet/1/bundle.complete"
MAX_MODEL_BYTES = 100_000_000


class AcousticModelLoadError(RuntimeError):
    """A stable loader failure code plus a safe, actionable explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: tuple[int | None, ...]
    dtype: str


@dataclass(frozen=True)
class AcousticModelMetadata:
    model_id: str
    model_version: str
    model_handle: str
    model_path: str
    artifact_sha256: str
    class_map_sha256: str
    label_mapping_version: str
    runtime_distribution: str
    runtime_version: str
    exported_with_tensorflow: str
    exported_with_tensorflow_git: str
    signature_name: str
    input: TensorContract
    outputs: tuple[TensorContract, ...]
    num_classes: int

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready metadata without exposing an absolute local path."""
        return asdict(self)


@dataclass(frozen=True)
class LoadedAcousticModel:
    metadata: AcousticModelMetadata
    classes: tuple[AcousticClass, ...]
    model: object = field(repr=False, compare=False)


def _safe_model_directory(paths: Paths, relative_path: str | Path) -> Path:
    relative = PurePosixPath(str(relative_path).replace(os.sep, "/"))
    if relative.is_absolute() or not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise AcousticModelLoadError("invalid_path", "Model path must be relative to models/ without traversal.")
    try:
        root = paths.root.resolve(strict=True)
        models = Path(os.path.abspath(paths.models))
        model_parts = models.relative_to(root).parts + relative.parts
        current = root
        for part in model_parts:
            current = current / part
            if _is_link(current):
                raise ValueError("linked model path")
        resolved = current.resolve(strict=True)
        if resolved != current or not resolved.is_relative_to(models) or not resolved.is_dir():
            raise ValueError("model directory is not contained")
        return resolved
    except FileNotFoundError as error:
        raise AcousticModelLoadError(
            "model_not_found", "The local YAMNet SavedModel is missing; run scripts/setup_yamnet.ps1."
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, AcousticModelLoadError):
            raise
        raise AcousticModelLoadError(
            "invalid_path", "Model must be a regular local directory inside models/ without links or junctions."
        ) from error


def _model_files(directory: Path) -> tuple[Path, ...]:
    try:
        entries = tuple(sorted(
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*") if path.is_file()
        ))
        allowed = set(MODEL_FILES) | {DOWNLOAD_MARKER}
        if not set(entries).issubset(allowed) or not set(MODEL_FILES).issubset(entries):
            raise AcousticModelLoadError(
                "invalid_artifact", "YAMNet SavedModel contains missing or unexpected files."
            )
        marker = directory / Path(*PurePosixPath(DOWNLOAD_MARKER).parts)
        if marker.exists() and (marker.stat().st_size != 0 or _is_link(marker)):
            raise AcousticModelLoadError("invalid_artifact", "YAMNet download marker is invalid.")
        paths = []
        total = 0
        for relative in MODEL_FILES:
            current = directory
            for part in PurePosixPath(relative).parts:
                current = current / part
                if _is_link(current):
                    raise AcousticModelLoadError(
                        "invalid_artifact", "YAMNet SavedModel must not contain links or junctions."
                    )
            resolved = current.resolve(strict=True)
            details = resolved.stat()
            if resolved != current or not stat.S_ISREG(details.st_mode):
                raise AcousticModelLoadError("invalid_artifact", "YAMNet payload entries must be regular files.")
            total += details.st_size
            if total > MAX_MODEL_BYTES:
                raise AcousticModelLoadError("model_too_large", "YAMNet SavedModel exceeds its byte limit.")
            paths.append(resolved)
        return tuple(paths)
    except AcousticModelLoadError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise AcousticModelLoadError("invalid_artifact", "YAMNet SavedModel files could not be verified.") from error


def model_artifact_sha256(directory: Path) -> str:
    """Hash the complete four-file YAMNet payload with paths and byte lengths."""
    digest = hashlib.sha256()
    for relative, path in zip(MODEL_FILES, _model_files(directory), strict=True):
        name = relative.encode("utf-8")
        with path.open("rb") as stream:
            initial = os.fstat(stream.fileno())
            digest.update(len(name).to_bytes(4, "big"))
            digest.update(name)
            digest.update(initial.st_size.to_bytes(8, "big"))
            while chunk := stream.read(1_048_576):
                digest.update(chunk)
            final = os.fstat(stream.fileno())
        if (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) != (
            final.st_size, final.st_mtime_ns, final.st_ctime_ns
        ):
            raise AcousticModelLoadError("artifact_changed", "YAMNet SavedModel changed while it was verified.")
    return digest.hexdigest()


def _shape(spec: object) -> tuple[int | None, ...]:
    shape = getattr(spec, "shape", None)
    values = shape.as_list() if hasattr(shape, "as_list") else tuple(shape or ())
    return tuple(None if value is None else int(value) for value in values)


def _tensor(name: str, spec: object) -> TensorContract:
    dtype = getattr(getattr(spec, "dtype", None), "name", str(getattr(spec, "dtype", "")))
    return TensorContract(name=name, shape=_shape(spec), dtype=dtype)


def _verify_signature(model: object) -> tuple[str, TensorContract, tuple[TensorContract, ...]]:
    signatures = getattr(model, "signatures", None)
    if not isinstance(signatures, Mapping) or set(signatures) != {"serving_default"} or not callable(model):
        raise AcousticModelLoadError("signature_mismatch", "YAMNet callable/signature inventory differs from v1.")
    signature = signatures["serving_default"]
    positional, inputs = signature.structured_input_signature
    outputs = signature.structured_outputs
    if positional or set(inputs) != {"waveform"} or set(outputs) != {"output_0", "output_1", "output_2"}:
        raise AcousticModelLoadError("signature_mismatch", "YAMNet input or output names differ from v1.")
    input_contract = _tensor("waveform", inputs["waveform"])
    output_contracts = tuple(_tensor(name, outputs[name]) for name in ("output_0", "output_1", "output_2"))
    expected_outputs = ((None, YAMNET.num_classes), (None, 1024), (None, YAMNET.mel_bands))
    if input_contract != TensorContract("waveform", (None,), "float32") or any(
        contract.shape != shape or contract.dtype != "float32"
        for contract, shape in zip(output_contracts, expected_outputs, strict=True)
    ):
        raise AcousticModelLoadError("signature_mismatch", "YAMNet tensor shape or dtype differs from v1.")
    return "serving_default", input_contract, output_contracts


def load_yamnet(paths: Paths, relative_path: str | Path = MODEL_RELATIVE_PATH) -> LoadedAcousticModel:
    """Load a pinned local YAMNet SavedModel without downloading or running inference."""
    validate_label_mapping()
    directory = _safe_model_directory(paths, relative_path)
    artifact_sha256 = model_artifact_sha256(directory)
    if artifact_sha256 != YAMNET.artifact_sha256:
        raise AcousticModelLoadError("artifact_mismatch", "YAMNet SavedModel digest differs from the pinned v1 artifact.")
    try:
        tensorflow = import_module("tensorflow")
    except (ImportError, OSError) as error:
        raise AcousticModelLoadError(
            "runtime_missing", "TensorFlow runtime is unavailable; run scripts/setup_yamnet.ps1."
        ) from error
    runtime_version = str(getattr(tensorflow, "__version__", ""))
    if runtime_version != YAMNET.runtime_version:
        raise AcousticModelLoadError(
            "runtime_mismatch", f"YAMNet requires {YAMNET.runtime_distribution} {YAMNET.runtime_version}."
        )
    try:
        model = tensorflow.saved_model.load(str(directory))
        class_map_value = model.class_map_path()
        class_map_raw = class_map_value.numpy() if hasattr(class_map_value, "numpy") else class_map_value
        if isinstance(class_map_raw, bytes):
            class_map_raw = class_map_raw.decode("utf-8")
        class_map_path = Path(str(class_map_raw)).resolve(strict=True)
        expected_class_map = (directory / "assets/yamnet_class_map.csv").resolve(strict=True)
        if class_map_path != expected_class_map:
            raise AcousticModelLoadError("class_map_mismatch", "Loaded YAMNet points to an unexpected vocabulary asset.")
        signature_name, input_contract, outputs = _verify_signature(model)
    except AcousticModelLoadError:
        raise
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise AcousticModelLoadError("model_load_failed", "TensorFlow could not load or inspect YAMNet v1.") from error
    if model_artifact_sha256(directory) != artifact_sha256:
        raise AcousticModelLoadError("artifact_changed", "YAMNet SavedModel changed while TensorFlow loaded it.")
    classes = load_class_map()
    metadata = AcousticModelMetadata(
        model_id=YAMNET.model_id,
        model_version="1",
        model_handle=YAMNET.model_handle,
        model_path=PurePosixPath(relative_path).as_posix(),
        artifact_sha256=artifact_sha256,
        class_map_sha256=YAMNET.class_map_sha256,
        label_mapping_version=LABEL_MAPPING_VERSION,
        runtime_distribution=YAMNET.runtime_distribution,
        runtime_version=runtime_version,
        exported_with_tensorflow=str(getattr(model, "tensorflow_version", "unknown")),
        exported_with_tensorflow_git=str(getattr(model, "tensorflow_git_version", "unknown")),
        signature_name=signature_name,
        input=input_contract,
        outputs=outputs,
        num_classes=len(classes),
    )
    return LoadedAcousticModel(metadata=metadata, classes=classes, model=model)
