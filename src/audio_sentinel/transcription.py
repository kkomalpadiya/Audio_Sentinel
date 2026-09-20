"""B4.2 verified local Faster-Whisper wrapper for one speech segment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from importlib import import_module, metadata as importlib_metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterable
import unicodedata

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from audio_sentinel.config import Paths
from audio_sentinel.persistence import _is_link
from audio_sentinel.speech_contracts import (
    SpeechModelDescriptor,
    TranscriptCandidate,
    TranscriptConfidenceKind,
)


MODEL_RELATIVE_PATH = PurePosixPath("faster-whisper-tiny.en/1")
INSTALL_MARKER = ".complete.json"
MAX_MARKER_BYTES = 32_768


@dataclass(frozen=True)
class TranscriptionArtifactSpec:
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class WhisperTinyEnglishSpec:
    model_id: str = "faster-whisper-tiny.en"
    model_version: str = "1"
    model_source: str = "https://huggingface.co/Systran/faster-whisper-tiny.en"
    source_repository: str = "Systran/faster-whisper-tiny.en"
    source_revision: str = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
    artifact_sha256: str = "c811dde420696b8ed6e745c3e1142a850c77f4c2f1c8244d9f1a8748f6093a1a"
    faster_whisper_version: str = "1.2.1"
    ctranslate2_version: str = "4.8.2"
    tokenizers_version: str = "0.23.2"
    huggingface_hub_version: str = "1.32.0"
    device: str = "cpu"
    requested_compute_type: str = "int8"
    resolved_compute_type: str = "int8_float32"
    language: str = "en"
    sample_rate_hz: int = 16_000
    artifacts: tuple[TranscriptionArtifactSpec, ...] = (
        TranscriptionArtifactSpec(
            "config.json",
            2_317,
            "14b1b421a90349bc551b881461426b561a874049cb9e4c4864f2ca384f6a7cc5",
        ),
        TranscriptionArtifactSpec(
            "model.bin",
            75_537_502,
            "1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a",
        ),
        TranscriptionArtifactSpec(
            "tokenizer.json",
            2_128_466,
            "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
        ),
        TranscriptionArtifactSpec(
            "vocabulary.txt",
            422_309,
            "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
        ),
    )

    def marker(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "model_id": self.model_id,
            "model_version": self.model_version,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
        }


WHISPER_TINY_EN = WhisperTinyEnglishSpec()


class TranscriptionError(RuntimeError):
    """Stable setup/inference failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TranscriptionArtifactMetadata:
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class TranscriptionModelMetadata:
    model_id: str
    model_version: str
    model_source: str
    model_path: str
    source_repository: str
    source_revision: str
    artifact_sha256: str
    artifacts: tuple[TranscriptionArtifactMetadata, ...]
    runtime_distribution: str
    runtime_version: str
    ctranslate2_version: str
    tokenizers_version: str
    device: str
    compute_type: str
    language: str
    sample_rate_hz: int

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
class LoadedTranscriptionModel:
    metadata: TranscriptionModelMetadata
    model: object = field(repr=False, compare=False)


class TranscriptionSettings(BaseModel):
    """Resource limits only; decoding choices are fixed for reproducibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_input_samples: int = Field(default=9_600_000, gt=0, strict=True)
    max_segments: int = Field(default=10_000, gt=0, le=100_000, strict=True)
    max_tokens: int = Field(default=100_000, gt=0, le=1_000_000, strict=True)
    max_output_bytes: int = Field(default=1_048_576, gt=0, le=16_777_216, strict=True)


@dataclass(frozen=True)
class TranscriptionChunk:
    segment_id: int
    start_seconds: float
    end_seconds: float
    text: str
    token_count: int
    average_log_probability: float
    no_speech_probability: float


@dataclass(frozen=True)
class TranscriptionResult:
    """One model hypothesis; reliability policy application belongs to A4.3."""

    model: TranscriptionModelMetadata
    input_num_samples: int
    input_duration_seconds: float
    candidate: TranscriptCandidate | None
    chunks: tuple[TranscriptionChunk, ...]

    def to_summary(self) -> dict[str, object]:
        return {
            "model": self.model.as_dict(),
            "input_num_samples": self.input_num_samples,
            "input_duration_seconds": self.input_duration_seconds,
            "candidate": None if self.candidate is None else self.candidate.model_dump(mode="json"),
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }


def _safe_model_directory(paths: Paths, relative_path: str | Path) -> Path:
    relative = PurePosixPath(str(relative_path).replace(os.sep, "/"))
    if relative.is_absolute() or not relative.parts or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise TranscriptionError(
            "invalid_path", "Transcription model path must stay inside models/ without traversal."
        )
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
        raise TranscriptionError(
            "model_not_found", "Faster-Whisper tiny.en is missing; run scripts/setup_speech.ps1."
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise TranscriptionError(
            "invalid_path",
            "Transcription model must be a regular local directory without links or junctions.",
        ) from error


def _regular_file(directory: Path, artifact: TranscriptionArtifactSpec) -> Path:
    try:
        current = directory / artifact.filename
        if _is_link(current):
            raise ValueError("linked payload")
        resolved = current.resolve(strict=True)
        details = resolved.stat()
        if (
            resolved != current
            or not stat.S_ISREG(details.st_mode)
            or details.st_size != artifact.size_bytes
        ):
            raise ValueError("invalid payload")
        return resolved
    except (OSError, RuntimeError, ValueError) as error:
        raise TranscriptionError(
            "invalid_artifact", "Transcription model files are missing, linked, or invalid."
        ) from error


def _read_marker(directory: Path) -> dict[str, object]:
    try:
        marker_path = directory / INSTALL_MARKER
        if _is_link(marker_path):
            raise ValueError("linked marker")
        resolved = marker_path.resolve(strict=True)
        details = resolved.stat()
        if resolved != marker_path or not stat.S_ISREG(details.st_mode) or details.st_size > MAX_MARKER_BYTES:
            raise ValueError("invalid marker")
        marker = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise TranscriptionError("invalid_artifact", "Transcription installation marker is invalid.") from error
    if marker != WHISPER_TINY_EN.marker():
        raise TranscriptionError(
            "invalid_artifact", "Transcription marker differs from the pinned model source."
        )
    return marker


def _hash_regular_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            initial = os.fstat(stream.fileno())
            while chunk := stream.read(1_048_576):
                digest.update(chunk)
            final = os.fstat(stream.fileno())
    except OSError as error:
        raise TranscriptionError("invalid_artifact", "Transcription model file could not be read.") from error
    if (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) != (
        final.st_size,
        final.st_mtime_ns,
        final.st_ctime_ns,
    ):
        raise TranscriptionError("artifact_changed", "Transcription model changed while verified.")
    return digest.hexdigest()


def model_artifact_sha256(directory: Path) -> str:
    """Verify every required file and return a digest of the canonical inventory."""

    expected_entries = {artifact.filename for artifact in WHISPER_TINY_EN.artifacts} | {
        INSTALL_MARKER
    }
    try:
        entries = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        }
    except (OSError, RuntimeError, ValueError) as error:
        raise TranscriptionError("invalid_artifact", "Transcription files could not be inventoried.") from error
    if entries != expected_entries:
        raise TranscriptionError(
            "invalid_artifact", "Transcription directory must contain only the pinned files and marker."
        )
    _read_marker(directory)
    inventory: dict[str, dict[str, object]] = {}
    for artifact in WHISPER_TINY_EN.artifacts:
        path = _regular_file(directory, artifact)
        digest = _hash_regular_file(path)
        if digest != artifact.sha256:
            raise TranscriptionError(
                "artifact_mismatch", f"Pinned transcription artifact failed verification: {artifact.filename}."
            )
        inventory[artifact.filename] = {
            "sha256": digest,
            "size_bytes": artifact.size_bytes,
        }
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError as error:
        raise TranscriptionError(
            "runtime_missing", "Pinned transcription runtime is unavailable; run scripts/setup_speech.ps1."
        ) from error


def _verify_runtime_versions() -> None:
    expected = {
        "faster-whisper": WHISPER_TINY_EN.faster_whisper_version,
        "ctranslate2": WHISPER_TINY_EN.ctranslate2_version,
        "tokenizers": WHISPER_TINY_EN.tokenizers_version,
    }
    for distribution, version in expected.items():
        if _distribution_version(distribution) != version:
            raise TranscriptionError(
                "runtime_mismatch", f"Transcription requires {distribution} {version}."
            )


def load_transcription_model(
    paths: Paths, relative_path: str | Path = MODEL_RELATIVE_PATH
) -> LoadedTranscriptionModel:
    """Load the pinned English-only model from disk with networking disabled."""

    directory = _safe_model_directory(paths, relative_path)
    artifact_sha256 = model_artifact_sha256(directory)
    if artifact_sha256 != WHISPER_TINY_EN.artifact_sha256:
        raise TranscriptionError("artifact_mismatch", "Transcription inventory digest is not pinned.")
    _verify_runtime_versions()
    try:
        faster_whisper = import_module("faster_whisper")
        model = faster_whisper.WhisperModel(
            str(directory),
            device=WHISPER_TINY_EN.device,
            compute_type=WHISPER_TINY_EN.requested_compute_type,
            cpu_threads=1,
            num_workers=1,
            local_files_only=True,
        )
        engine = model.model
        device = str(engine.device)
        compute_type = str(engine.compute_type)
        multilingual = bool(engine.is_multilingual)
    except TranscriptionError:
        raise
    except Exception as error:
        raise TranscriptionError(
            "model_load_failed", "Faster-Whisper could not load the pinned tiny.en model."
        ) from error
    if (
        device != WHISPER_TINY_EN.device
        or compute_type != WHISPER_TINY_EN.resolved_compute_type
        or multilingual
        or not callable(getattr(model, "transcribe", None))
    ):
        raise TranscriptionError(
            "signature_mismatch", "Loaded transcription model has an unexpected device or capability."
        )
    if model_artifact_sha256(directory) != artifact_sha256:
        raise TranscriptionError("artifact_changed", "Transcription model changed while it loaded.")
    artifacts = tuple(
        TranscriptionArtifactMetadata(item.filename, item.size_bytes, item.sha256)
        for item in WHISPER_TINY_EN.artifacts
    )
    return LoadedTranscriptionModel(
        metadata=TranscriptionModelMetadata(
            model_id=WHISPER_TINY_EN.model_id,
            model_version=WHISPER_TINY_EN.model_version,
            model_source=WHISPER_TINY_EN.model_source,
            model_path=PurePosixPath(relative_path).as_posix(),
            source_repository=WHISPER_TINY_EN.source_repository,
            source_revision=WHISPER_TINY_EN.source_revision,
            artifact_sha256=artifact_sha256,
            artifacts=artifacts,
            runtime_distribution="faster-whisper",
            runtime_version=WHISPER_TINY_EN.faster_whisper_version,
            ctranslate2_version=WHISPER_TINY_EN.ctranslate2_version,
            tokenizers_version=WHISPER_TINY_EN.tokenizers_version,
            device=device,
            compute_type=compute_type,
            language=WHISPER_TINY_EN.language,
            sample_rate_hz=WHISPER_TINY_EN.sample_rate_hz,
        ),
        model=model,
    )


def _expected_metadata() -> dict[str, object]:
    return {
        "model_id": WHISPER_TINY_EN.model_id,
        "model_version": WHISPER_TINY_EN.model_version,
        "model_source": WHISPER_TINY_EN.model_source,
        "model_path": MODEL_RELATIVE_PATH.as_posix(),
        "source_repository": WHISPER_TINY_EN.source_repository,
        "source_revision": WHISPER_TINY_EN.source_revision,
        "artifact_sha256": WHISPER_TINY_EN.artifact_sha256,
        "runtime_distribution": "faster-whisper",
        "runtime_version": WHISPER_TINY_EN.faster_whisper_version,
        "ctranslate2_version": WHISPER_TINY_EN.ctranslate2_version,
        "tokenizers_version": WHISPER_TINY_EN.tokenizers_version,
        "device": WHISPER_TINY_EN.device,
        "compute_type": WHISPER_TINY_EN.resolved_compute_type,
        "language": WHISPER_TINY_EN.language,
        "sample_rate_hz": WHISPER_TINY_EN.sample_rate_hz,
    }


def validate_loaded_transcription_model(loaded: LoadedTranscriptionModel) -> None:
    metadata = loaded.metadata
    values = metadata.as_dict()
    values.pop("artifacts")
    if values != _expected_metadata() or metadata.artifacts != tuple(
        TranscriptionArtifactMetadata(item.filename, item.size_bytes, item.sha256)
        for item in WHISPER_TINY_EN.artifacts
    ) or not callable(getattr(loaded.model, "transcribe", None)):
        raise TranscriptionError(
            "model_mismatch", "Transcription requires the verified pinned tiny.en model."
        )


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TranscriptionError("invalid_output", f"Transcription {name} must be finite numeric data.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TranscriptionError(
            "invalid_output", f"Transcription {name} must be finite numeric data."
        ) from error
    if not math.isfinite(result):
        raise TranscriptionError("invalid_output", f"Transcription {name} must be finite numeric data.")
    return result


def _normalized_text(value: object) -> str:
    if not isinstance(value, str):
        raise TranscriptionError("invalid_output", "Transcription text must be a string.")
    normalized = " ".join(value.split())
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise TranscriptionError("invalid_output", "Transcription text contains a control character.")
    return normalized


def _tokens(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TranscriptionError("invalid_output", "Transcription tokens must be integers.")
    try:
        tokens = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TranscriptionError("invalid_output", "Transcription tokens must be integers.") from error
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise TranscriptionError("invalid_output", "Transcription tokens must be non-negative integers.")
    return tokens


def _read_chunks(
    segments: Iterable[object], *, duration: float, settings: TranscriptionSettings
) -> tuple[TranscriptionChunk, ...]:
    chunks: list[TranscriptionChunk] = []
    total_tokens = 0
    total_bytes = 0
    previous_start = 0.0
    try:
        for index, segment in enumerate(segments):
            if index >= settings.max_segments:
                raise TranscriptionError("output_too_large", "Transcription exceeded max_segments.")
            segment_id = getattr(segment, "id", None)
            if isinstance(segment_id, bool) or not isinstance(segment_id, int) or segment_id < 0:
                raise TranscriptionError("invalid_output", "Transcription segment ID is invalid.")
            start = _number(getattr(segment, "start", None), "segment start")
            end = _number(getattr(segment, "end", None), "segment end")
            average_log_probability = _number(
                getattr(segment, "avg_logprob", None), "average log probability"
            )
            no_speech_probability = _number(
                getattr(segment, "no_speech_prob", None), "no-speech probability"
            )
            text = _normalized_text(getattr(segment, "text", None))
            tokens = _tokens(getattr(segment, "tokens", None))
            if (
                start < 0
                or end < start
                or end > duration + 1e-6
                or start < previous_start
                or average_log_probability > 0
                or not 0 <= no_speech_probability <= 1
            ):
                raise TranscriptionError(
                    "invalid_output", "Transcription segment timing or scores are outside valid bounds."
                )
            previous_start = start
            total_tokens += len(tokens)
            total_bytes += len(text.encode("utf-8"))
            if total_tokens > settings.max_tokens:
                raise TranscriptionError("output_too_large", "Transcription exceeded max_tokens.")
            if total_bytes > settings.max_output_bytes:
                raise TranscriptionError("output_too_large", "Transcription exceeded max_output_bytes.")
            if text:
                chunks.append(TranscriptionChunk(
                    segment_id=segment_id,
                    start_seconds=start,
                    end_seconds=end,
                    text=text,
                    token_count=len(tokens),
                    average_log_probability=average_log_probability,
                    no_speech_probability=no_speech_probability,
                ))
    except TranscriptionError:
        raise
    except Exception as error:
        raise TranscriptionError("invalid_output", "Transcription segments could not be read.") from error
    return tuple(chunks)


def _candidate(chunks: tuple[TranscriptionChunk, ...]) -> TranscriptCandidate | None:
    if not chunks:
        return None
    text = " ".join(chunk.text for chunk in chunks)
    weights = [max(chunk.token_count, 1) for chunk in chunks]
    mean_log_probability = sum(
        chunk.average_log_probability * weight for chunk, weight in zip(chunks, weights, strict=True)
    ) / sum(weights)
    confidence = min(1.0, max(0.0, math.exp(mean_log_probability)))
    try:
        return TranscriptCandidate(
            text=text,
            confidence_score=confidence,
            confidence_kind=TranscriptConfidenceKind.DERIVED_SCORE,
            language=WHISPER_TINY_EN.language,
            language_confidence=1.0,
        )
    except Exception as error:
        raise TranscriptionError("invalid_output", "Transcription candidate violates its contract.") from error


def transcribe_segment(
    loaded: LoadedTranscriptionModel,
    waveform: NDArray[np.float32],
    *,
    settings: TranscriptionSettings | None = None,
) -> TranscriptionResult:
    """Transcribe one independent 16 kHz segment without applying reliability gates."""

    settings = TranscriptionSettings.model_validate((settings or TranscriptionSettings()).model_dump())
    validate_loaded_transcription_model(loaded)
    if not isinstance(waveform, np.ndarray) or waveform.dtype != np.float32 or waveform.ndim != 1:
        raise TranscriptionError(
            "invalid_samples", "Transcription expects a one-dimensional float32 waveform."
        )
    if len(waveform) == 0:
        raise TranscriptionError("invalid_samples", "Transcription waveform must contain samples.")
    if len(waveform) > settings.max_input_samples:
        raise TranscriptionError("input_too_large", "Transcription waveform exceeds max_input_samples.")
    if not np.isfinite(waveform).all() or np.any(waveform < -1) or np.any(waveform > 1):
        raise TranscriptionError(
            "invalid_samples", "Transcription samples must be finite and within [-1, 1]."
        )
    duration = len(waveform) / WHISPER_TINY_EN.sample_rate_hz
    model_input = np.array(waveform, dtype=np.float32, copy=True, order="C")
    try:
        segments, info = loaded.model.transcribe(
            model_input,
            language=WHISPER_TINY_EN.language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
            without_timestamps=False,
            initial_prompt=None,
            hotwords=None,
            suppress_blank=True,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        chunks = _read_chunks(segments, duration=duration, settings=settings)
        language = str(getattr(info, "language", ""))
        language_probability = _number(
            getattr(info, "language_probability", None), "language probability"
        )
        reported_duration = _number(getattr(info, "duration", None), "duration")
    except TranscriptionError:
        raise
    except MemoryError as error:
        raise TranscriptionError(
            "insufficient_memory", "Not enough memory to run transcription."
        ) from error
    except Exception as error:
        raise TranscriptionError("model_failed", "Faster-Whisper failed to transcribe the segment.") from error
    if (
        language != WHISPER_TINY_EN.language
        or language_probability != 1.0
        or not math.isclose(reported_duration, duration, rel_tol=0, abs_tol=1e-6)
    ):
        raise TranscriptionError(
            "invalid_output", "English-only transcription metadata differs from the input."
        )
    return TranscriptionResult(
        model=loaded.metadata,
        input_num_samples=len(waveform),
        input_duration_seconds=duration,
        candidate=_candidate(chunks),
        chunks=chunks,
    )
