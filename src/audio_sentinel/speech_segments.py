"""A4.2 verified speech-segment extraction and absolute timestamp handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
import soundfile as sf

from audio_sentinel.audio_loader import validate_processing_consent
from audio_sentinel.config import Paths
from audio_sentinel.contracts import ConsentRecord, ProcessingScope
from audio_sentinel.feature_persistence import _read_bytes, _safe_file
from audio_sentinel.preparation import PreparedAudioManifest, PreparedWindowRecord
from audio_sentinel.speech_contracts import (
    SpeechEvidenceDocument,
    SpeechEvidenceSource,
    SpeechReliabilityPolicy,
    SpeechSegmentEvidence,
    SpeechWindowReference,
)
from audio_sentinel.vad import (
    LoadedVadModel,
    VadError,
    VadInferenceResult,
    VadInferenceSettings,
    expected_vad_frames,
    infer_vad_probabilities,
    validate_loaded_vad,
)


class SpeechSegmentationError(RuntimeError):
    """Stable segment-extraction failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class SpeechSegmentationSettings(BaseModel):
    """Reproducible window selection, merge behavior, and resource limits."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    window_seconds: float | None = Field(default=None, gt=0, le=600)
    merge_gap_samples: int = Field(default=0, ge=0, strict=True)
    minimum_speech_samples: int = Field(default=1, gt=0, strict=True)
    max_manifest_bytes: int = Field(default=16_777_216, gt=0, strict=True)
    max_window_bytes: int = Field(default=268_435_456, gt=0, strict=True)
    max_source_bytes: int = Field(default=1_073_741_824, gt=0, strict=True)
    max_decoded_bytes: int = Field(default=268_435_456, gt=0, strict=True)
    max_windows: int = Field(default=10_000, gt=0, strict=True)
    max_frames: int = Field(default=1_000_000, gt=0, strict=True)
    max_segments: int = Field(default=100_000, gt=0, strict=True)


@dataclass(frozen=True)
class SpeechWindowVadInference:
    """Raw B4.1 output plus the exact prepared-window bytes that produced it."""

    window: PreparedWindowRecord
    window_audio_sha256: str
    vad: VadInferenceResult


@dataclass(frozen=True)
class SpeechFrameContribution:
    """One threshold-qualified frame expressed on the prepared-clip sample grid."""

    window_id: str
    window_audio_sha256: str
    frame_index: int
    start_sample: int
    end_sample: int
    speech_probability: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedSpeechSegment:
    """A non-overlapping union of qualifying VAD frame support."""

    segment_id: str
    source_window_ids: tuple[str, ...]
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    vad_score: float
    contributions: tuple[SpeechFrameContribution, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "source_window_ids": self.source_window_ids,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "vad_score": self.vad_score,
            "contributions": [item.as_dict() for item in self.contributions],
        }


@dataclass(frozen=True)
class SpeechSegmentationResult:
    """Transcript-free A4.1 evidence plus inspectable extraction provenance."""

    evidence: SpeechEvidenceDocument
    settings: SpeechSegmentationSettings
    selected_window_seconds: float
    input_frame_count: int
    windows: tuple[SpeechWindowVadInference, ...]
    segments: tuple[ExtractedSpeechSegment, ...]

    def to_summary(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence.evidence_id,
            "clip_id": self.evidence.source.clip_id,
            "model": self.evidence.vad_model.model_dump(mode="json"),
            "reliability_policy": self.evidence.reliability_policy.model_dump(mode="json"),
            "settings": self.settings.model_dump(mode="json"),
            "selected_window_seconds": self.selected_window_seconds,
            "input_window_count": len(self.windows),
            "input_frame_count": self.input_frame_count,
            "segment_count": len(self.segments),
            "segments": [item.as_dict() for item in self.segments],
        }


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise SpeechSegmentationError(
            "invalid_time", "Speech evidence creation time must be timezone-aware."
        )
    return value.astimezone(UTC)


def _validate_speech_consent(consent: ConsentRecord, now: datetime) -> ConsentRecord:
    consent = validate_processing_consent(consent, now)
    if consent.processing_scope is not ProcessingScope.ACOUSTIC_AND_SPEECH:
        raise SpeechSegmentationError(
            "scope_not_allowed", "Active acoustic-and-speech processing permission is required."
        )
    return consent


def _manifest(
    paths: Paths,
    relative: str | Path,
    settings: SpeechSegmentationSettings,
) -> tuple[Path, bytes, PreparedAudioManifest]:
    try:
        path = _safe_file(paths.root, paths.interim_data, relative)
        document = _read_bytes(path, settings.max_manifest_bytes)
        return path, document, PreparedAudioManifest.model_validate_json(document)
    except SpeechSegmentationError:
        raise
    except Exception as error:
        code = getattr(error, "code", "invalid_manifest")
        raise SpeechSegmentationError(
            code, "The prepared-audio manifest could not be verified."
        ) from error


def _window_bytes(
    paths: Paths,
    window: PreparedWindowRecord,
    settings: SpeechSegmentationSettings,
) -> tuple[bytes, str]:
    try:
        path = _safe_file(paths.root, paths.interim_data, window.audio_path)
        document = _read_bytes(path, settings.max_window_bytes)
        return document, hashlib.sha256(document).hexdigest()
    except Exception as error:
        code = getattr(error, "code", "invalid_window")
        raise SpeechSegmentationError(
            code, "A listed prepared speech window could not be verified."
        ) from error


def _decode_window(
    document: bytes,
    manifest: PreparedAudioManifest,
    window: PreparedWindowRecord,
    settings: SpeechSegmentationSettings,
) -> NDArray[np.float32]:
    stored_samples = window.end_sample - window.start_sample + window.padding_samples
    if stored_samples * 4 > settings.max_decoded_bytes:
        raise SpeechSegmentationError(
            "decoded_audio_too_large", "A speech window exceeds the decoded-sample budget."
        )
    try:
        with sf.SoundFile(BytesIO(document)) as audio:
            if (audio.format, audio.subtype, audio.samplerate, audio.channels, audio.frames) != (
                "WAV",
                "PCM_16",
                manifest.clip.sample_rate_hz,
                1,
                stored_samples,
            ):
                raise SpeechSegmentationError(
                    "source_mismatch",
                    "Speech window WAV properties differ from the preparation manifest.",
                )
            samples = audio.read(stored_samples, dtype="float32", always_2d=False)
            if samples.shape != (stored_samples,) or len(audio.read(1)):
                raise SpeechSegmentationError(
                    "source_mismatch", "Speech window length differs from the preparation manifest."
                )
    except SpeechSegmentationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise SpeechSegmentationError(
            "decode_failed", "A prepared speech window could not be decoded."
        ) from error
    if not np.isfinite(samples).all() or np.any(samples < -1) or np.any(samples > 1):
        raise SpeechSegmentationError(
            "invalid_samples", "Decoded speech-window samples must be finite and within [-1, 1]."
        )
    if window.padding_samples:
        if np.any(samples[-window.padding_samples:]):
            raise SpeechSegmentationError(
                "source_mismatch", "Prepared speech-window padding contains nonzero samples."
            )
        samples = samples[:-window.padding_samples]
    if samples.shape != (window.end_sample - window.start_sample,):
        raise SpeechSegmentationError(
            "source_mismatch", "Real speech-window length differs from its recorded span."
        )
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    samples.setflags(write=False)
    return samples


def _select_windows(
    manifest: PreparedAudioManifest,
    settings: SpeechSegmentationSettings,
) -> tuple[float, tuple[PreparedWindowRecord, ...]]:
    selected_seconds = (
        manifest.settings.window_seconds[0]
        if settings.window_seconds is None
        else settings.window_seconds
    )
    if selected_seconds not in manifest.settings.window_seconds:
        raise SpeechSegmentationError(
            "window_duration_unavailable",
            "The requested speech window duration is absent from the preparation manifest.",
        )
    selected = tuple(
        window for window in manifest.windows if window.window_seconds == selected_seconds
    )
    if not selected:
        raise SpeechSegmentationError(
            "incomplete_window_coverage",
            "The selected prepared-window duration does not cover the source clip.",
        )
    ordered = tuple(sorted(selected, key=lambda item: (item.start_sample, item.end_sample, item.window_id)))
    coverage_end = 0
    for window in ordered:
        if window.start_sample > coverage_end:
            raise SpeechSegmentationError(
                "incomplete_window_coverage",
                "Selected prepared speech windows contain an uncovered sample gap.",
            )
        coverage_end = max(coverage_end, window.end_sample)
    if coverage_end != manifest.num_frames:
        raise SpeechSegmentationError(
            "incomplete_window_coverage",
            "Selected prepared speech windows do not cover the complete source clip.",
        )
    return selected_seconds, ordered


def _validate_window_vad(item: SpeechWindowVadInference) -> None:
    window = item.window
    expected_samples = window.end_sample - window.start_sample
    if item.vad.input_num_samples != expected_samples:
        raise SpeechSegmentationError(
            "invalid_vad_result", "VAD input length differs from its prepared-window span."
        )
    if item.vad.frame_count != expected_vad_frames(expected_samples):
        raise SpeechSegmentationError(
            "invalid_vad_result", "VAD frame count differs from the pinned frame grid."
        )
    for index, frame in enumerate(item.vad.frames):
        start = index * item.vad.model.frame_samples
        end = min(start + item.vad.model.frame_samples, expected_samples)
        if (
            frame.frame_index != index
            or frame.start_sample != start
            or frame.end_sample != end
            or frame.padding_samples != item.vad.model.frame_samples - (end - start)
            or frame.start_seconds != start / item.vad.model.sample_rate_hz
            or frame.end_seconds != end / item.vad.model.sample_rate_hz
            or not np.isfinite(frame.speech_probability)
            or not 0 <= frame.speech_probability <= 1
        ):
            raise SpeechSegmentationError(
                "invalid_vad_result", "VAD frame metadata differs from the pinned sample grid."
            )


def _contributions(
    windows: tuple[SpeechWindowVadInference, ...],
    threshold: float,
) -> list[SpeechFrameContribution]:
    output: list[SpeechFrameContribution] = []
    for item in windows:
        _validate_window_vad(item)
        for frame in item.vad.frames:
            if frame.speech_probability < threshold:
                continue
            output.append(
                SpeechFrameContribution(
                    window_id=item.window.window_id,
                    window_audio_sha256=item.window_audio_sha256,
                    frame_index=frame.frame_index,
                    start_sample=item.window.start_sample + frame.start_sample,
                    end_sample=item.window.start_sample + frame.end_sample,
                    speech_probability=frame.speech_probability,
                )
            )
    return sorted(
        output,
        key=lambda item: (
            item.start_sample,
            item.end_sample,
            item.window_id,
            item.frame_index,
        ),
    )


def _source_window_ids(
    windows: tuple[SpeechWindowVadInference, ...], start: int, end: int
) -> tuple[str, ...]:
    references = tuple(
        item.window.window_id
        for item in windows
        if item.window.start_sample < end and item.window.end_sample > start
    )
    coverage_end = start
    for item in windows:
        if item.window.window_id not in references:
            continue
        if item.window.start_sample > coverage_end:
            raise SpeechSegmentationError(
                "incomplete_segment_coverage",
                "Referenced prepared windows do not cover an extracted speech segment.",
            )
        coverage_end = max(coverage_end, item.window.end_sample)
    if coverage_end < end:
        raise SpeechSegmentationError(
            "incomplete_segment_coverage",
            "Referenced prepared windows do not cover an extracted speech segment.",
        )
    return references


def _merge_contributions(
    contributions: list[SpeechFrameContribution],
    windows: tuple[SpeechWindowVadInference, ...],
    sample_rate_hz: int,
    settings: SpeechSegmentationSettings,
) -> tuple[ExtractedSpeechSegment, ...]:
    groups: list[tuple[int, int, float, tuple[SpeechFrameContribution, ...]]] = []
    current: list[SpeechFrameContribution] = []
    start = end = 0
    peak = 0.0
    for contribution in contributions:
        if current and contribution.start_sample > end + settings.merge_gap_samples:
            groups.append((start, end, peak, tuple(current)))
            current = []
        if not current:
            start = contribution.start_sample
            end = contribution.end_sample
            peak = contribution.speech_probability
        else:
            end = max(end, contribution.end_sample)
            peak = max(peak, contribution.speech_probability)
        current.append(contribution)
    if current:
        groups.append((start, end, peak, tuple(current)))

    segments: list[ExtractedSpeechSegment] = []
    for start, end, peak, group in groups:
        if end - start < settings.minimum_speech_samples:
            continue
        if len(segments) >= settings.max_segments:
            raise SpeechSegmentationError(
                "too_many_segments", "Extracted speech segments exceed the configured limit."
            )
        segment_id = f"speech-{len(segments):04d}"
        segments.append(
            ExtractedSpeechSegment(
                segment_id=segment_id,
                source_window_ids=_source_window_ids(windows, start, end),
                start_sample=start,
                end_sample=end,
                start_seconds=start / sample_rate_hz,
                end_seconds=end / sample_rate_hz,
                vad_score=peak,
                contributions=group,
            )
        )
    return tuple(segments)


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _evidence(
    manifest: PreparedAudioManifest,
    relative_manifest: str,
    manifest_sha256: str,
    loaded: LoadedVadModel,
    policy: SpeechReliabilityPolicy,
    windows: tuple[SpeechWindowVadInference, ...],
    segments: tuple[ExtractedSpeechSegment, ...],
    now: datetime,
) -> SpeechEvidenceDocument:
    window_documents = tuple(
        SpeechWindowReference(
            window_id=item.window.window_id,
            audio_path=item.window.audio_path,
            window_audio_sha256=item.window_audio_sha256,
            start_sample=item.window.start_sample,
            end_sample=item.window.end_sample,
            padding_samples=item.window.padding_samples,
        )
        for item in windows
    )
    segment_documents = tuple(
        SpeechSegmentEvidence(
            segment_id=item.segment_id,
            source_window_ids=item.source_window_ids,
            start_sample=item.start_sample,
            end_sample=item.end_sample,
            start_seconds=item.start_seconds,
            end_seconds=item.end_seconds,
            vad_score=item.vad_score,
            transcript=None,
            assessment=policy.assess(None),
        )
        for item in segments
    )
    payload: dict[str, object] = {
        "source": SpeechEvidenceSource(
            preparation_manifest_path=relative_manifest,
            preparation_manifest_sha256=manifest_sha256,
            raw_audio_sha256=manifest.source.sha256,
            clip_id=manifest.clip.clip_id,
            consent_id=manifest.clip.consent.consent_id,
            processing_scope="acoustic_and_speech",
            sample_rate_hz=manifest.clip.sample_rate_hz,
            num_samples=manifest.num_frames,
        ),
        "reliability_policy": policy,
        "vad_model": loaded.metadata.as_speech_descriptor(),
        "transcription_model": None,
        "input_window_count": len(window_documents),
        "segment_count": len(segment_documents),
        "transcribed_segment_count": 0,
        "windows": window_documents,
        "segments": segment_documents,
    }
    identity = {
        key: (
            value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else [item.model_dump(mode="json") for item in value]
            if isinstance(value, tuple)
            else value
        )
        for key, value in payload.items()
    }
    return SpeechEvidenceDocument(
        evidence_id=_canonical_hash(identity), created_at=now, **payload
    )


def extract_prepared_speech_segments(
    paths: Paths,
    manifest_path: str | Path,
    loaded: LoadedVadModel,
    *,
    reliability_policy: SpeechReliabilityPolicy | None = None,
    settings: SpeechSegmentationSettings | None = None,
    inference_settings: VadInferenceSettings | None = None,
    now: datetime | None = None,
) -> SpeechSegmentationResult:
    """Verify prepared windows, score them, and emit sample-exact VAD-positive segments."""

    resolved_settings = SpeechSegmentationSettings.model_validate(
        (settings or SpeechSegmentationSettings()).model_dump()
    )
    resolved_policy = SpeechReliabilityPolicy.model_validate(
        (reliability_policy or SpeechReliabilityPolicy()).model_dump()
    )
    created_at = _clock(now)
    try:
        validate_loaded_vad(loaded)
    except Exception as error:
        code = getattr(error, "code", "model_mismatch")
        raise SpeechSegmentationError(
            code, "Speech segmentation requires the verified pinned Silero VAD model."
        ) from error

    path, manifest_bytes, manifest = _manifest(paths, manifest_path, resolved_settings)
    _validate_speech_consent(manifest.clip.consent, _clock(now))
    if manifest.clip.sample_rate_hz != loaded.metadata.sample_rate_hz:
        raise SpeechSegmentationError(
            "sample_rate_mismatch", "Silero VAD requires prepared 16 kHz audio."
        )
    if manifest.channels != 1:
        raise SpeechSegmentationError(
            "mono_required", "Silero VAD requires prepared mono audio."
        )
    selected_seconds, selected_windows = _select_windows(manifest, resolved_settings)
    if len(selected_windows) > resolved_settings.max_windows:
        raise SpeechSegmentationError(
            "too_many_windows", "Selected speech windows exceed the configured limit."
        )
    predicted_frames = sum(
        expected_vad_frames(item.end_sample - item.start_sample)
        for item in selected_windows
    )
    if predicted_frames > resolved_settings.max_frames:
        raise SpeechSegmentationError(
            "too_many_frames", "Selected speech windows exceed the configured VAD frame limit."
        )

    relative_manifest = path.relative_to(Path(os.path.abspath(paths.interim_data))).as_posix()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    results: list[SpeechWindowVadInference] = []
    used_source_bytes = len(manifest_bytes)
    try:
        for window in selected_windows:
            _validate_speech_consent(manifest.clip.consent, _clock(now))
            audio_bytes, audio_sha256 = _window_bytes(paths, window, resolved_settings)
            used_source_bytes += len(audio_bytes)
            if used_source_bytes > resolved_settings.max_source_bytes:
                raise SpeechSegmentationError(
                    "source_too_large", "Prepared speech sources exceed the configured byte limit."
                )
            waveform = _decode_window(audio_bytes, manifest, window, resolved_settings)
            try:
                vad = infer_vad_probabilities(
                    loaded, waveform, settings=inference_settings
                )
            except VadError as error:
                raise SpeechSegmentationError(
                    error.code, "Silero VAD could not score a prepared speech window."
                ) from error
            _validate_speech_consent(manifest.clip.consent, _clock(now))
            results.append(
                SpeechWindowVadInference(
                    window=window,
                    window_audio_sha256=audio_sha256,
                    vad=vad,
                )
            )
    except SpeechSegmentationError:
        raise
    except MemoryError as error:
        raise SpeechSegmentationError(
            "insufficient_memory", "Not enough memory to extract speech segments."
        ) from error

    final_path, final_manifest_bytes, _ = _manifest(
        paths, relative_manifest, resolved_settings
    )
    if final_path != path or final_manifest_bytes != manifest_bytes:
        raise SpeechSegmentationError(
            "source_changed", "Preparation manifest changed during speech segmentation."
        )
    for item in results:
        _, current_sha256 = _window_bytes(paths, item.window, resolved_settings)
        if current_sha256 != item.window_audio_sha256:
            raise SpeechSegmentationError(
                "source_changed", "A prepared speech window changed during segmentation."
            )
    _validate_speech_consent(manifest.clip.consent, _clock(now))

    window_results = tuple(results)
    contributions = _contributions(
        window_results, resolved_policy.vad_speech_threshold
    )
    segments = _merge_contributions(
        contributions,
        window_results,
        manifest.clip.sample_rate_hz,
        resolved_settings,
    )
    evidence = _evidence(
        manifest,
        relative_manifest,
        manifest_sha256,
        loaded,
        resolved_policy,
        window_results,
        segments,
        created_at,
    )
    return SpeechSegmentationResult(
        evidence=evidence,
        settings=resolved_settings,
        selected_window_seconds=selected_seconds,
        input_frame_count=predicted_frames,
        windows=window_results,
        segments=segments,
    )
