"""A4.3 tests for verified transcription orchestration and policy handling."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
import soundfile as sf

from audio_sentinel import speech_transcription as orchestration
from audio_sentinel.config import AudioSettings
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.pipeline import AudioPreparationService
from audio_sentinel.speech_contracts import (
    SpeechReliabilityPolicy,
    TranscriptConfidenceKind,
    TranscriptReliability,
)
from audio_sentinel.speech_segments import extract_prepared_speech_segments
from audio_sentinel.transcription import (
    WHISPER_TINY_EN,
    LoadedTranscriptionModel,
    TranscriptionArtifactMetadata,
    TranscriptionModelMetadata,
)
from audio_sentinel.vad import (
    SILERO_VAD,
    LoadedVadModel,
    VadModelMetadata,
    VadTensorContract,
)


NOW = datetime(2026, 9, 20, 12, tzinfo=UTC)


class FakeVadSession:
    def __init__(self, score_sets=()):
        self.score_sets = tuple(score_sets)
        self.calls = []

    def run(self, _output_names, feeds):
        call_index = len(self.calls)
        self.calls.append({name: value.copy() for name, value in feeds.items()})
        scores = np.full(len(feeds["input"]), np.float32(0.1), dtype=np.float32)
        values = self.score_sets[call_index] if call_index < len(self.score_sets) else {}
        for index, score in values.items():
            scores[index] = np.float32(score)
        state = np.zeros((1, 1, SILERO_VAD.state_size), dtype=np.float32)
        return scores, state, state.copy()


def loaded_vad(session):
    metadata = VadModelMetadata(
        model_id=SILERO_VAD.model_id,
        model_version=SILERO_VAD.model_version,
        model_source=SILERO_VAD.model_source,
        model_path="silero-vad/6",
        artifact_sha256=SILERO_VAD.artifact_sha256,
        source_distribution=SILERO_VAD.source_distribution,
        source_distribution_version=SILERO_VAD.source_distribution_version,
        runtime_distribution=SILERO_VAD.runtime_distribution,
        runtime_version=SILERO_VAD.runtime_version,
        providers=("CPUExecutionProvider",),
        inputs=(
            VadTensorContract("input", (None, 576), "tensor(float)"),
            VadTensorContract("h", (1, 1, 128), "tensor(float)"),
            VadTensorContract("c", (1, 1, 128), "tensor(float)"),
        ),
        outputs=(
            VadTensorContract("speech_probs", (None,), "tensor(float)"),
            VadTensorContract("hn", (1, 1, 128), "tensor(float)"),
            VadTensorContract("cn", (1, 1, 128), "tensor(float)"),
        ),
        sample_rate_hz=16_000,
        frame_samples=512,
        context_samples=64,
    )
    return LoadedVadModel(metadata, session)


class FakeTranscriber:
    def __init__(self, responses=(), *, fail=None, on_call=None):
        self.responses = tuple(responses)
        self.fail = fail
        self.on_call = on_call
        self.calls = []

    def transcribe(self, samples, **kwargs):
        call_index = len(self.calls)
        self.calls.append((samples.copy(), kwargs))
        if self.on_call:
            self.on_call(call_index)
        if self.fail:
            raise self.fail
        response = self.responses[call_index] if call_index < len(self.responses) else None
        duration = len(samples) / 16_000
        if response is None:
            output = []
        else:
            text, score = response
            output = [SimpleNamespace(
                id=0,
                start=0.0,
                end=duration,
                text=text,
                tokens=(1,),
                avg_logprob=math.log(score),
                no_speech_prob=0.1,
            )]
        info = SimpleNamespace(language="en", language_probability=1.0, duration=duration)
        return iter(output), info


def loaded_transcriber(model=None, metadata=None):
    spec = WHISPER_TINY_EN
    metadata = metadata or TranscriptionModelMetadata(
        model_id=spec.model_id,
        model_version=spec.model_version,
        model_source=spec.model_source,
        model_path="faster-whisper-tiny.en/1",
        source_repository=spec.source_repository,
        source_revision=spec.source_revision,
        artifact_sha256=spec.artifact_sha256,
        artifacts=tuple(
            TranscriptionArtifactMetadata(item.filename, item.size_bytes, item.sha256)
            for item in spec.artifacts
        ),
        runtime_distribution="faster-whisper",
        runtime_version=spec.faster_whisper_version,
        ctranslate2_version=spec.ctranslate2_version,
        tokenizers_version=spec.tokenizers_version,
        device="cpu",
        compute_type="int8_float32",
        language="en",
        sample_rate_hz=16_000,
    )
    return LoadedTranscriptionModel(metadata, model or FakeTranscriber())


@pytest.fixture
def prepared(temporary_settings, active_consent):
    temporary_settings.paths.raw_data.mkdir(parents=True)
    path = temporary_settings.paths.raw_data / "speech-orchestration.wav"
    timeline = np.arange(25_600, dtype=np.float64) / 16_000
    waveform = 0.2 * np.sin(2 * np.pi * 220 * timeline)
    sf.write(path, waveform, 16_000, subtype="PCM_16")
    bundle = AudioPreparationService(temporary_settings, "speech-orchestration").prepare(
        InputAudio("speech-orchestration-clip", path, active_consent),
        audio_settings=AudioSettings(
            normalize_loudness=False,
            window_seconds=(1.0,),
            window_overlap_ratio=0.5,
        ),
        now=NOW,
    )
    return temporary_settings.paths, bundle


def segmented(prepared, score_sets=({0: 0.8}, {}, {}), *, policy=None):
    paths, bundle = prepared
    return extract_prepared_speech_segments(
        paths,
        bundle.manifest_path.relative_to(paths.interim_data),
        loaded_vad(FakeVadSession(score_sets)),
        reliability_policy=policy,
        now=NOW,
    )


def run(prepared, responses=(("accepted", 0.9),), *, score_sets=({0: 0.8}, {}, {}), **kwargs):
    paths, _ = prepared
    segmentation = segmented(prepared, score_sets)
    model = FakeTranscriber(responses)
    result = orchestration.orchestrate_speech_transcription(
        paths,
        segmentation,
        loaded_transcriber(model),
        now=NOW,
        **kwargs,
    )
    return result, segmentation, model


def test_orchestrates_all_policy_outcomes_and_only_hands_off_accepted_text(prepared):
    scores = ({0: 0.8, 2: 0.8, 4: 0.8, 6: 0.8}, {}, {})
    responses = (("rejected", 0.49), ("review", 0.5), ("accepted", 0.8), None)

    result, _, model = run(prepared, responses, score_sets=scores)

    assert len(model.calls) == 4
    assert result.rejected_segment_ids == ("speech-0000",)
    assert result.review_required_segment_ids == ("speech-0001",)
    assert result.accepted_segment_ids == ("speech-0002",)
    assert result.untranscribed_segment_ids == ("speech-0003",)
    assert [item.text for item in result.downstream_transcripts] == ["accepted"]
    assert [item.assessment.downstream_text_allowed for item in result.transcriptions] == [
        False, False, True, False
    ]
    assert result.evidence.transcribed_segment_count == 3
    assert result.evidence.transcription_model == loaded_transcriber().metadata.as_speech_descriptor()


def test_reconstructs_exact_segment_samples_from_prepared_window(prepared):
    result, segmentation, model = run(prepared)
    segment = segmentation.segments[0]
    paths, _ = prepared
    window = segmentation.windows[0].window
    stored, _ = sf.read(paths.interim_data / window.audio_path, dtype="float32")

    assert np.array_equal(model.calls[0][0], stored[segment.start_sample:segment.end_sample])
    assert model.calls[0][0].shape == (512,)
    assert result.evidence.segments[0].start_sample == segment.start_sample


def test_segment_crossing_overlapping_windows_is_reconstructed_once(prepared):
    scores = ({30: 0.8}, {14: 0.9}, {})
    result, segmentation, model = run(prepared, score_sets=scores)

    assert len(segmentation.segments) == 1
    assert len(segmentation.segments[0].source_window_ids) == 2
    assert len(model.calls) == 1
    assert model.calls[0][0].shape == (
        segmentation.segments[0].end_sample - segmentation.segments[0].start_sample,
    )
    assert result.accepted_segment_ids == ("speech-0000",)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.499, TranscriptReliability.REJECTED_LOW_CONFIDENCE),
        (0.5, TranscriptReliability.REVIEW_REQUIRED),
        (0.799, TranscriptReliability.REVIEW_REQUIRED),
        (0.8, TranscriptReliability.ACCEPTED),
    ],
)
def test_reliability_threshold_boundaries_are_applied(prepared, score, expected):
    result, _, _ = run(prepared, (("candidate", score),))
    assert result.transcriptions[0].assessment.reliability is expected
    assert bool(result.downstream_transcripts) is (expected is TranscriptReliability.ACCEPTED)


def test_empty_model_output_remains_not_transcribed_and_never_means_safe(prepared):
    result, _, _ = run(prepared, (None,))
    assessment = result.evidence.segments[0].assessment

    assert result.evidence.segments[0].transcript is None
    assert assessment.reliability is TranscriptReliability.NOT_TRANSCRIBED
    assert assessment.downstream_text_allowed is False
    assert result.downstream_transcripts == ()


def test_new_evidence_identity_is_stable_across_creation_times(prepared):
    paths, _ = prepared
    segmentation = segmented(prepared)
    first = orchestration.orchestrate_speech_transcription(
        paths, segmentation, loaded_transcriber(FakeTranscriber((("same", 0.9),))), now=NOW
    )
    second = orchestration.orchestrate_speech_transcription(
        paths,
        segmentation,
        loaded_transcriber(FakeTranscriber((("same", 0.9),))),
        now=datetime(2026, 9, 21, tzinfo=UTC),
    )
    assert first.evidence.evidence_id == second.evidence.evidence_id
    assert first.evidence.created_at != second.evidence.created_at


def test_empty_segmentation_records_model_without_calling_it(prepared):
    paths, _ = prepared
    segmentation = segmented(prepared, ({}, {}, {}))
    model = FakeTranscriber()

    result = orchestration.orchestrate_speech_transcription(
        paths, segmentation, loaded_transcriber(model), now=NOW
    )

    assert model.calls == [] and result.transcriptions == ()
    assert result.evidence.transcription_model is not None
    assert result.evidence.segment_count == result.evidence.transcribed_segment_count == 0


def test_model_is_validated_before_source_files_are_read(prepared):
    paths, bundle = prepared
    segmentation = segmented(prepared)
    bundle.manifest_path.unlink()
    metadata = replace(loaded_transcriber().metadata, device="cuda")

    with pytest.raises(orchestration.SpeechOrchestrationError) as error:
        orchestration.orchestrate_speech_transcription(
            paths, segmentation, loaded_transcriber(metadata=metadata), now=NOW
        )

    assert error.value.code == "model_mismatch"


@pytest.mark.parametrize("damage", ["identity", "window", "segment", "existing_transcript"])
def test_rejects_tampered_segmentation_snapshot(prepared, damage):
    paths, _ = prepared
    segmentation = segmented(prepared)
    if damage == "identity":
        evidence = segmentation.evidence.model_copy(update={"evidence_id": "0" * 64})
        segmentation = replace(segmentation, evidence=evidence)
    elif damage == "window":
        changed = replace(segmentation.windows[0], window_audio_sha256="0" * 64)
        segmentation = replace(segmentation, windows=(changed, *segmentation.windows[1:]))
    elif damage == "segment":
        changed = replace(segmentation.segments[0], vad_score=0.99)
        segmentation = replace(segmentation, segments=(changed,))
    else:
        public = segmentation.evidence.segments[0].model_copy(
            update={"transcript": SimpleNamespace(text="bad")}
        )
        evidence = segmentation.evidence.model_copy(update={"segments": (public,)})
        segmentation = replace(segmentation, evidence=evidence)

    with pytest.raises(orchestration.SpeechOrchestrationError) as error:
        orchestration.orchestrate_speech_transcription(
            paths, segmentation, loaded_transcriber(), now=NOW
        )

    assert error.value.code == "invalid_segmentation"


def test_rejects_changed_prepared_window_before_model_call(prepared):
    paths, _ = prepared
    segmentation = segmented(prepared)
    target = paths.interim_data / segmentation.windows[0].window.audio_path
    target.write_bytes(target.read_bytes() + b"changed")
    model = FakeTranscriber()

    with pytest.raises(orchestration.SpeechOrchestrationError) as error:
        orchestration.orchestrate_speech_transcription(
            paths, segmentation, loaded_transcriber(model), now=NOW
        )

    assert error.value.code == "source_mismatch" and model.calls == []


def test_detects_manifest_change_during_transcription(prepared):
    paths, bundle = prepared
    segmentation = segmented(prepared)

    def mutate(_index):
        bundle.manifest_path.write_bytes(bundle.manifest_path.read_bytes() + b"\n")

    with pytest.raises(orchestration.SpeechOrchestrationError) as error:
        orchestration.orchestrate_speech_transcription(
            paths,
            segmentation,
            loaded_transcriber(FakeTranscriber((("text", 0.9),), on_call=mutate)),
            now=NOW,
        )
    assert error.value.code == "source_changed"


def test_detects_window_change_during_transcription(prepared):
    paths, _ = prepared
    segmentation = segmented(prepared)
    target = paths.interim_data / segmentation.windows[0].window.audio_path

    def mutate(_index):
        target.write_bytes(target.read_bytes() + b"changed")

    with pytest.raises(orchestration.SpeechOrchestrationError) as error:
        orchestration.orchestrate_speech_transcription(
            paths,
            segmentation,
            loaded_transcriber(FakeTranscriber((("text", 0.9),), on_call=mutate)),
            now=NOW,
        )
    assert error.value.code == "source_mismatch"


def test_model_failure_is_wrapped_without_private_details(prepared):
    paths, _ = prepared
    segmentation = segmented(prepared)
    with pytest.raises(orchestration.SpeechOrchestrationError) as error:
        orchestration.orchestrate_speech_transcription(
            paths,
            segmentation,
            loaded_transcriber(FakeTranscriber(fail=RuntimeError("private detail"))),
            now=NOW,
        )
    assert error.value.code == "model_failed" and "private" not in str(error.value)


@pytest.mark.parametrize(
    ("settings", "code"),
    [
        (orchestration.SpeechOrchestrationSettings(max_segments=1), "too_many_segments"),
        (orchestration.SpeechOrchestrationSettings(max_total_segment_samples=1), "input_too_large"),
        (orchestration.SpeechOrchestrationSettings(max_source_bytes=1), "source_too_large"),
        (orchestration.SpeechOrchestrationSettings(max_decoded_bytes=1), "decoded_audio_too_large"),
        (orchestration.SpeechOrchestrationSettings(max_downstream_text_bytes=1), "output_too_large"),
    ],
)
def test_resource_limits_fail_without_returning_partial_output(prepared, settings, code):
    score_sets = ({0: 0.8, 2: 0.8}, {}, {}) if code == "too_many_segments" else ({0: 0.8}, {}, {})
    paths, _ = prepared
    segmentation = segmented(prepared, score_sets)
    with pytest.raises(orchestration.SpeechOrchestrationError) as error:
        orchestration.orchestrate_speech_transcription(
            paths,
            segmentation,
            loaded_transcriber(FakeTranscriber((("accepted", 0.9),) * 2)),
            settings=settings,
            now=NOW,
        )
    assert error.value.code == code


def test_summary_is_json_ready_and_contains_no_waveform_or_absolute_path(prepared):
    result, _, _ = run(prepared)
    document = json.dumps(result.to_summary(), sort_keys=True)
    assert json.loads(document)["accepted_segment_ids"] == ["speech-0000"]
    assert "waveform" not in document
    assert str(prepared[0].root) not in document


def test_result_and_handoff_records_are_immutable(prepared):
    result, _, _ = run(prepared)
    with pytest.raises(FrozenInstanceError):
        result.transcriptions = ()
    with pytest.raises(FrozenInstanceError):
        result.downstream_transcripts[0].text = "changed"


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_manifest_bytes": 0},
        {"max_window_bytes": True},
        {"max_source_bytes": 0},
        {"max_decoded_bytes": 0},
        {"max_segments": 0},
        {"max_total_segment_samples": 0},
        {"max_downstream_text_bytes": 0},
        {"unexpected": 1},
    ],
)
def test_invalid_orchestration_settings_are_rejected(overrides):
    with pytest.raises(ValidationError):
        orchestration.SpeechOrchestrationSettings(**overrides)


def test_naive_creation_time_is_rejected_before_io(prepared):
    paths, bundle = prepared
    segmentation = segmented(prepared)
    bundle.manifest_path.unlink()
    with pytest.raises(orchestration.SpeechOrchestrationError) as error:
        orchestration.orchestrate_speech_transcription(
            paths, segmentation, loaded_transcriber(), now=datetime(2026, 9, 20)
        )
    assert error.value.code == "invalid_time"


def test_expired_consent_blocks_transcription(prepared):
    paths, bundle = prepared
    segmentation = segmented(prepared)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    manifest["clip"]["consent"]["expires_at"] = "2026-09-20T11:00:00Z"
    document = json.dumps(manifest, indent=2, sort_keys=True).encode()
    bundle.manifest_path.write_bytes(document)
    evidence = segmentation.evidence.model_copy(update={
        "source": segmentation.evidence.source.model_copy(update={
            "preparation_manifest_sha256": hashlib.sha256(document).hexdigest()
        })
    })
    evidence = evidence.model_copy(update={
        "evidence_id": orchestration._canonical_hash(orchestration._identity_payload(evidence))
    })
    segmentation = replace(segmentation, evidence=evidence)
    with pytest.raises(orchestration.SpeechOrchestrationError) as error:
        orchestration.orchestrate_speech_transcription(
            paths, segmentation, loaded_transcriber(), now=NOW
        )
    assert error.value.code == "consent_expired"


def test_downstream_record_preserves_exact_times_and_confidence_kind(prepared):
    result, segmentation, _ = run(prepared)
    handoff = result.downstream_transcripts[0]
    segment = segmentation.segments[0]
    assert (handoff.start_sample, handoff.end_sample) == (segment.start_sample, segment.end_sample)
    assert (handoff.start_seconds, handoff.end_seconds) == (
        segment.start_seconds, segment.end_seconds
    )
    assert handoff.confidence_kind is TranscriptConfidenceKind.DERIVED_SCORE
