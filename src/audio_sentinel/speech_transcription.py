"""A4.3 verified segment transcription and low-confidence orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
import soundfile as sf

from audio_sentinel.audio_loader import validate_processing_consent
from audio_sentinel.config import Paths
from audio_sentinel.contracts import ProcessingScope
from audio_sentinel.feature_persistence import _read_bytes, _safe_file
from audio_sentinel.preparation import PreparedAudioManifest, PreparedWindowRecord
from audio_sentinel.speech_contracts import (
    SpeechEvidenceDocument,
    SpeechReliabilityAssessment,
    SpeechSegmentEvidence,
    TranscriptConfidenceKind,
    TranscriptReliability,
)
from audio_sentinel.speech_segments import (
    ExtractedSpeechSegment,
    SpeechSegmentationResult,
    SpeechWindowVadInference,
)
from audio_sentinel.transcription import (
    LoadedTranscriptionModel,
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSettings,
    transcribe_segment,
    validate_loaded_transcription_model,
)


class SpeechOrchestrationError(RuntimeError):
    """Stable orchestration failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class SpeechOrchestrationSettings(BaseModel):
    """Resource limits for verified source reconstruction and transcript handoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_manifest_bytes: int = Field(default=16_777_216, gt=0, strict=True)
    max_window_bytes: int = Field(default=268_435_456, gt=0, strict=True)
    max_source_bytes: int = Field(default=1_073_741_824, gt=0, strict=True)
    max_decoded_bytes: int = Field(default=268_435_456, gt=0, strict=True)
    max_segments: int = Field(default=100_000, gt=0, strict=True)
    max_total_segment_samples: int = Field(default=9_600_000, gt=0, strict=True)
    max_downstream_text_bytes: int = Field(default=1_048_576, gt=0, strict=True)


@dataclass(frozen=True)
class DownstreamTranscript:
    """Accepted text that may proceed automatically to language analysis."""

    segment_id: str
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    text: str
    confidence_score: float
    confidence_kind: TranscriptConfidenceKind

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentTranscription:
    """Inspectable model result and its policy-derived handling outcome."""

    segment_id: str
    transcription: TranscriptionResult
    assessment: SpeechReliabilityAssessment

    def to_summary(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "transcription": self.transcription.to_summary(),
            "assessment": self.assessment.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class SpeechTranscriptionResult:
    """Completed evidence plus an acceptance-only downstream text boundary."""

    evidence: SpeechEvidenceDocument
    settings: SpeechOrchestrationSettings
    transcriptions: tuple[SegmentTranscription, ...]
    downstream_transcripts: tuple[DownstreamTranscript, ...]

    @property
    def accepted_segment_ids(self) -> tuple[str, ...]:
        return tuple(item.segment_id for item in self.downstream_transcripts)

    @property
    def review_required_segment_ids(self) -> tuple[str, ...]:
        return tuple(
            item.segment_id
            for item in self.transcriptions
            if item.assessment.reliability is TranscriptReliability.REVIEW_REQUIRED
        )

    @property
    def rejected_segment_ids(self) -> tuple[str, ...]:
        return tuple(
            item.segment_id
            for item in self.transcriptions
            if item.assessment.reliability is TranscriptReliability.REJECTED_LOW_CONFIDENCE
        )

    @property
    def untranscribed_segment_ids(self) -> tuple[str, ...]:
        return tuple(
            item.segment_id
            for item in self.transcriptions
            if item.assessment.reliability is TranscriptReliability.NOT_TRANSCRIBED
        )

    def to_summary(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence.evidence_id,
            "clip_id": self.evidence.source.clip_id,
            "settings": self.settings.model_dump(mode="json"),
            "segment_count": self.evidence.segment_count,
            "transcribed_segment_count": self.evidence.transcribed_segment_count,
            "accepted_segment_ids": self.accepted_segment_ids,
            "review_required_segment_ids": self.review_required_segment_ids,
            "rejected_segment_ids": self.rejected_segment_ids,
            "untranscribed_segment_ids": self.untranscribed_segment_ids,
            "transcriptions": [item.to_summary() for item in self.transcriptions],
            "downstream_transcripts": [item.as_dict() for item in self.downstream_transcripts],
        }


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise SpeechOrchestrationError(
            "invalid_time", "Speech transcription time must be timezone-aware."
        )
    return value.astimezone(UTC)


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _identity_payload(evidence: SpeechEvidenceDocument) -> dict[str, object]:
    return {
        "source": evidence.source.model_dump(mode="json"),
        "reliability_policy": evidence.reliability_policy.model_dump(mode="json"),
        "vad_model": evidence.vad_model.model_dump(mode="json"),
        "transcription_model": (
            None
            if evidence.transcription_model is None
            else evidence.transcription_model.model_dump(mode="json")
        ),
        "input_window_count": evidence.input_window_count,
        "segment_count": evidence.segment_count,
        "transcribed_segment_count": evidence.transcribed_segment_count,
        "windows": [item.model_dump(mode="json") for item in evidence.windows],
        "segments": [item.model_dump(mode="json") for item in evidence.segments],
    }


def _validate_segmentation(segmentation: SpeechSegmentationResult) -> None:
    try:
        evidence = SpeechEvidenceDocument.model_validate(
            segmentation.evidence.model_dump(mode="python")
        )
        settings = type(segmentation.settings).model_validate(segmentation.settings.model_dump())
    except Exception as error:
        raise SpeechOrchestrationError(
            "invalid_segmentation", "Speech segmentation result failed contract validation."
        ) from error
    if (
        evidence != segmentation.evidence
        or settings != segmentation.settings
        or evidence.evidence_id != _canonical_hash(_identity_payload(evidence))
        or evidence.transcription_model is not None
        or evidence.transcribed_segment_count != 0
        or any(item.transcript is not None for item in evidence.segments)
        or len(segmentation.windows) != evidence.input_window_count
        or len(segmentation.segments) != evidence.segment_count
        or len(segmentation.windows) > segmentation.settings.max_windows
        or len(segmentation.segments) > segmentation.settings.max_segments
    ):
        raise SpeechOrchestrationError(
            "invalid_segmentation", "A4.3 requires an intact transcript-free A4.2 result."
        )
    expected_frames = 0
    for diagnostic, public in zip(segmentation.windows, evidence.windows, strict=True):
        if (
            diagnostic.window.window_id != public.window_id
            or diagnostic.window.audio_path != public.audio_path
            or diagnostic.window.start_sample != public.start_sample
            or diagnostic.window.end_sample != public.end_sample
            or diagnostic.window.padding_samples != public.padding_samples
            or diagnostic.window.window_seconds != segmentation.selected_window_seconds
            or diagnostic.window_audio_sha256 != public.window_audio_sha256
            or diagnostic.vad.model.as_speech_descriptor() != evidence.vad_model
            or diagnostic.vad.input_num_samples
            != diagnostic.window.end_sample - diagnostic.window.start_sample
        ):
            raise SpeechOrchestrationError(
                "invalid_segmentation", "Speech window diagnostics differ from public evidence."
            )
        expected_frames += diagnostic.vad.frame_count
    if expected_frames != segmentation.input_frame_count:
        raise SpeechOrchestrationError(
            "invalid_segmentation", "Speech VAD frame count differs from its diagnostics."
        )
    for diagnostic, public in zip(segmentation.segments, evidence.segments, strict=True):
        if (
            diagnostic.segment_id != public.segment_id
            or diagnostic.source_window_ids != public.source_window_ids
            or diagnostic.start_sample != public.start_sample
            or diagnostic.end_sample != public.end_sample
            or diagnostic.start_seconds != public.start_seconds
            or diagnostic.end_seconds != public.end_seconds
            or diagnostic.vad_score != public.vad_score
        ):
            raise SpeechOrchestrationError(
                "invalid_segmentation", "Speech segment diagnostics differ from public evidence."
            )


def _validate_consent(manifest: PreparedAudioManifest, now: datetime) -> None:
    try:
        consent = validate_processing_consent(manifest.clip.consent, now)
    except Exception as error:
        code = getattr(error, "code", "consent_invalid")
        raise SpeechOrchestrationError(
            code, "Active acoustic-and-speech processing permission is required."
        ) from error
    if consent.processing_scope is not ProcessingScope.ACOUSTIC_AND_SPEECH:
        raise SpeechOrchestrationError(
            "scope_not_allowed", "Active acoustic-and-speech processing permission is required."
        )


def _read_manifest(
    paths: Paths,
    relative: str,
    settings: SpeechOrchestrationSettings,
) -> tuple[Path, bytes, PreparedAudioManifest]:
    try:
        path = _safe_file(paths.root, paths.interim_data, relative)
        document = _read_bytes(path, settings.max_manifest_bytes)
        manifest = PreparedAudioManifest.model_validate_json(document)
        return path, document, manifest
    except SpeechOrchestrationError:
        raise
    except Exception as error:
        code = getattr(error, "code", "invalid_manifest")
        raise SpeechOrchestrationError(
            code, "The prepared-audio manifest could not be verified for transcription."
        ) from error


def _validate_manifest(
    manifest: PreparedAudioManifest,
    manifest_bytes: bytes,
    segmentation: SpeechSegmentationResult,
    now: datetime,
) -> dict[str, PreparedWindowRecord]:
    evidence = segmentation.evidence
    source = evidence.source
    _validate_consent(manifest, now)
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != source.preparation_manifest_sha256
        or manifest.source.sha256 != source.raw_audio_sha256
        or manifest.clip.clip_id != source.clip_id
        or manifest.clip.consent.consent_id != source.consent_id
        or manifest.clip.sample_rate_hz != source.sample_rate_hz
        or manifest.num_frames != source.num_samples
        or manifest.channels != 1
        or manifest.clip.sample_rate_hz != 16_000
    ):
        raise SpeechOrchestrationError(
            "source_mismatch", "Prepared manifest differs from the A4.2 speech evidence."
        )
    manifest_windows = {item.window_id: item for item in manifest.windows}
    selected: dict[str, PreparedWindowRecord] = {}
    for diagnostic in segmentation.windows:
        current = manifest_windows.get(diagnostic.window.window_id)
        if current != diagnostic.window:
            raise SpeechOrchestrationError(
                "source_mismatch", "Prepared window inventory differs from A4.2 diagnostics."
            )
        selected[current.window_id] = current
    return selected


def _decode_window(
    paths: Paths,
    diagnostic: SpeechWindowVadInference,
    settings: SpeechOrchestrationSettings,
) -> tuple[NDArray[np.float32], int]:
    window = diagnostic.window
    stored_samples = window.end_sample - window.start_sample + window.padding_samples
    if stored_samples * 4 > settings.max_decoded_bytes:
        raise SpeechOrchestrationError(
            "decoded_audio_too_large", "A prepared speech window exceeds the decode budget."
        )
    try:
        path = _safe_file(paths.root, paths.interim_data, window.audio_path)
        document = _read_bytes(path, settings.max_window_bytes)
    except Exception as error:
        code = getattr(error, "code", "invalid_window")
        raise SpeechOrchestrationError(
            code, "A prepared speech window could not be verified for transcription."
        ) from error
    if hashlib.sha256(document).hexdigest() != diagnostic.window_audio_sha256:
        raise SpeechOrchestrationError(
            "source_mismatch", "A prepared speech window differs from A4.2 evidence."
        )
    try:
        with sf.SoundFile(BytesIO(document)) as audio:
            if (audio.format, audio.subtype, audio.samplerate, audio.channels, audio.frames) != (
                "WAV",
                "PCM_16",
                16_000,
                1,
                stored_samples,
            ):
                raise SpeechOrchestrationError(
                    "source_mismatch", "Prepared speech-window WAV properties changed."
                )
            samples = audio.read(stored_samples, dtype="float32", always_2d=False)
            if samples.shape != (stored_samples,) or len(audio.read(1)):
                raise SpeechOrchestrationError(
                    "source_mismatch", "Prepared speech-window length changed."
                )
    except SpeechOrchestrationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise SpeechOrchestrationError(
            "decode_failed", "A prepared speech window could not be decoded."
        ) from error
    if not np.isfinite(samples).all() or np.any(samples < -1) or np.any(samples > 1):
        raise SpeechOrchestrationError(
            "invalid_samples", "Prepared speech samples must be finite and within [-1, 1]."
        )
    if window.padding_samples:
        if np.any(samples[-window.padding_samples:]):
            raise SpeechOrchestrationError(
                "source_mismatch", "Prepared speech-window padding contains nonzero samples."
            )
        samples = samples[:-window.padding_samples]
    if samples.shape != (window.end_sample - window.start_sample,):
        raise SpeechOrchestrationError(
            "source_mismatch", "Prepared speech-window span differs from its samples."
        )
    output = np.ascontiguousarray(samples, dtype=np.float32)
    output.setflags(write=False)
    return output, len(document)


def _segment_waveform(
    segment: ExtractedSpeechSegment,
    windows: dict[str, SpeechWindowVadInference],
    samples: dict[str, NDArray[np.float32]],
) -> NDArray[np.float32]:
    length = segment.end_sample - segment.start_sample
    try:
        output = np.empty(length, dtype=np.float32)
        filled = np.zeros(length, dtype=np.bool_)
    except MemoryError as error:
        raise SpeechOrchestrationError(
            "insufficient_memory", "Not enough memory to reconstruct a speech segment."
        ) from error
    for window_id in segment.source_window_ids:
        diagnostic = windows.get(window_id)
        source = samples.get(window_id)
        if diagnostic is None or source is None:
            raise SpeechOrchestrationError(
                "invalid_segmentation", "Speech segment references an unavailable prepared window."
            )
        window = diagnostic.window
        start = max(segment.start_sample, window.start_sample)
        end = min(segment.end_sample, window.end_sample)
        if end <= start:
            raise SpeechOrchestrationError(
                "invalid_segmentation", "Speech segment does not overlap a referenced window."
            )
        destination_start = start - segment.start_sample
        destination_end = end - segment.start_sample
        source_start = start - window.start_sample
        source_end = end - window.start_sample
        destination = output[destination_start:destination_end]
        mask = filled[destination_start:destination_end]
        piece = source[source_start:source_end]
        if np.any(mask) and not np.array_equal(destination[mask], piece[mask]):
            raise SpeechOrchestrationError(
                "source_mismatch", "Overlapping prepared windows contain different samples."
            )
        destination[~mask] = piece[~mask]
        mask[:] = True
    if not np.all(filled):
        raise SpeechOrchestrationError(
            "invalid_segmentation", "Referenced prepared windows do not cover the speech segment."
        )
    output.setflags(write=False)
    return output


def _build_evidence(
    segmentation: SpeechSegmentationResult,
    loaded: LoadedTranscriptionModel,
    transcriptions: tuple[SegmentTranscription, ...],
    created_at: datetime,
) -> SpeechEvidenceDocument:
    originals = segmentation.evidence.segments
    public_segments = tuple(
        SpeechSegmentEvidence(
            segment_id=original.segment_id,
            source_window_ids=original.source_window_ids,
            start_sample=original.start_sample,
            end_sample=original.end_sample,
            start_seconds=original.start_seconds,
            end_seconds=original.end_seconds,
            vad_score=original.vad_score,
            transcript=diagnostic.transcription.candidate,
            assessment=diagnostic.assessment,
        )
        for original, diagnostic in zip(originals, transcriptions, strict=True)
    )
    payload: dict[str, object] = {
        "source": segmentation.evidence.source,
        "reliability_policy": segmentation.evidence.reliability_policy,
        "vad_model": segmentation.evidence.vad_model,
        "transcription_model": loaded.metadata.as_speech_descriptor(),
        "input_window_count": segmentation.evidence.input_window_count,
        "segment_count": len(public_segments),
        "transcribed_segment_count": sum(item.transcript is not None for item in public_segments),
        "windows": segmentation.evidence.windows,
        "segments": public_segments,
    }
    temporary = SpeechEvidenceDocument(
        evidence_id="0" * 64,
        created_at=created_at,
        **payload,
    )
    return SpeechEvidenceDocument(
        evidence_id=_canonical_hash(_identity_payload(temporary)),
        created_at=created_at,
        **payload,
    )


def orchestrate_speech_transcription(
    paths: Paths,
    segmentation: SpeechSegmentationResult,
    loaded: LoadedTranscriptionModel,
    *,
    settings: SpeechOrchestrationSettings | None = None,
    transcription_settings: TranscriptionSettings | None = None,
    now: datetime | None = None,
) -> SpeechTranscriptionResult:
    """Transcribe verified speech segments and expose only policy-accepted text downstream."""

    resolved = SpeechOrchestrationSettings.model_validate(
        (settings or SpeechOrchestrationSettings()).model_dump()
    )
    created_at = _clock(now)
    try:
        validate_loaded_transcription_model(loaded)
    except Exception as error:
        code = getattr(error, "code", "model_mismatch")
        raise SpeechOrchestrationError(
            code, "Speech orchestration requires the verified pinned transcription model."
        ) from error
    _validate_segmentation(segmentation)
    if len(segmentation.segments) > resolved.max_segments:
        raise SpeechOrchestrationError(
            "too_many_segments", "Speech segments exceed the transcription limit."
        )
    total_segment_samples = sum(
        item.end_sample - item.start_sample for item in segmentation.segments
    )
    if total_segment_samples > resolved.max_total_segment_samples:
        raise SpeechOrchestrationError(
            "input_too_large", "Speech segments exceed the total transcription sample budget."
        )

    path, manifest_bytes, manifest = _read_manifest(
        paths,
        segmentation.evidence.source.preparation_manifest_path,
        resolved,
    )
    selected = _validate_manifest(manifest, manifest_bytes, segmentation, _clock(now))
    if path.relative_to(Path(os.path.abspath(paths.interim_data))).as_posix() != (
        segmentation.evidence.source.preparation_manifest_path
    ):
        raise SpeechOrchestrationError(
            "source_mismatch", "Prepared manifest path differs from A4.2 evidence."
        )

    diagnostics = {item.window.window_id: item for item in segmentation.windows}
    decoded: dict[str, NDArray[np.float32]] = {}
    used_source_bytes = len(manifest_bytes)
    used_decoded_bytes = 0
    for window_id, window in selected.items():
        if window_id not in diagnostics:
            raise SpeechOrchestrationError(
                "source_mismatch", "Selected prepared window is missing from A4.2 diagnostics."
            )
        waveform, source_bytes = _decode_window(paths, diagnostics[window_id], resolved)
        used_source_bytes += source_bytes
        used_decoded_bytes += waveform.nbytes
        if used_source_bytes > resolved.max_source_bytes:
            raise SpeechOrchestrationError(
                "source_too_large", "Prepared speech sources exceed the configured byte limit."
            )
        if used_decoded_bytes > resolved.max_decoded_bytes:
            raise SpeechOrchestrationError(
                "decoded_audio_too_large", "Prepared speech samples exceed the decode budget."
            )
        decoded[window_id] = waveform

    policy = segmentation.evidence.reliability_policy
    segment_results: list[SegmentTranscription] = []
    downstream: list[DownstreamTranscript] = []
    downstream_bytes = 0
    try:
        for segment in segmentation.segments:
            _validate_consent(manifest, _clock(now))
            waveform = _segment_waveform(segment, diagnostics, decoded)
            try:
                transcription = transcribe_segment(
                    loaded,
                    waveform,
                    settings=transcription_settings,
                )
            except TranscriptionError as error:
                raise SpeechOrchestrationError(
                    error.code, "The offline model could not transcribe a verified speech segment."
                ) from error
            if (
                transcription.model != loaded.metadata
                or transcription.input_num_samples != len(waveform)
                or not math.isclose(
                    transcription.input_duration_seconds,
                    len(waveform) / 16_000,
                    rel_tol=0,
                    abs_tol=1e-9,
                )
            ):
                raise SpeechOrchestrationError(
                    "invalid_output", "Transcription result differs from its segment input."
                )
            assessment = policy.assess(transcription.candidate)
            diagnostic = SegmentTranscription(
                segment_id=segment.segment_id,
                transcription=transcription,
                assessment=assessment,
            )
            segment_results.append(diagnostic)
            if assessment.downstream_text_allowed:
                candidate = transcription.candidate
                if candidate is None:
                    raise SpeechOrchestrationError(
                        "invalid_output", "Accepted transcription is missing its candidate text."
                    )
                downstream_bytes += len(candidate.text.encode("utf-8"))
                if downstream_bytes > resolved.max_downstream_text_bytes:
                    raise SpeechOrchestrationError(
                        "output_too_large", "Accepted transcript text exceeds the handoff limit."
                    )
                downstream.append(DownstreamTranscript(
                    segment_id=segment.segment_id,
                    start_sample=segment.start_sample,
                    end_sample=segment.end_sample,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    text=candidate.text,
                    confidence_score=candidate.confidence_score,
                    confidence_kind=candidate.confidence_kind,
                ))
            _validate_consent(manifest, _clock(now))
    except SpeechOrchestrationError:
        raise
    except MemoryError as error:
        raise SpeechOrchestrationError(
            "insufficient_memory", "Not enough memory to orchestrate speech transcription."
        ) from error

    final_path, final_manifest_bytes, final_manifest = _read_manifest(
        paths,
        segmentation.evidence.source.preparation_manifest_path,
        resolved,
    )
    if final_path != path or final_manifest_bytes != manifest_bytes or final_manifest != manifest:
        raise SpeechOrchestrationError(
            "source_changed", "Preparation manifest changed during speech transcription."
        )
    for window_id in decoded:
        _decode_window(paths, diagnostics[window_id], resolved)
    _validate_consent(manifest, _clock(now))

    transcriptions = tuple(segment_results)
    evidence = _build_evidence(segmentation, loaded, transcriptions, created_at)
    return SpeechTranscriptionResult(
        evidence=evidence,
        settings=resolved,
        transcriptions=transcriptions,
        downstream_transcripts=tuple(downstream),
    )
