"""A4.4 integration tests for the complete prepared-audio speech branch."""

from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio_sentinel import speech_segments, speech_transcription
from audio_sentinel.config import AudioSettings
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.pipeline import AudioPreparationService
from audio_sentinel.speech_contracts import (
    SpeechEvidenceDocument,
    SpeechReliabilityPolicy,
    TranscriptReliability,
)
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


NOW = datetime(2026, 9, 21, 12, tzinfo=UTC)
LATER = datetime(2026, 9, 22, 12, tzinfo=UTC)


class FakeVadSession:
    def __init__(self, score_sets=(), *, fail=None):
        self.score_sets = tuple(score_sets)
        self.fail = fail
        self.calls = []

    def run(self, _output_names, feeds):
        call_index = len(self.calls)
        self.calls.append({name: value.copy() for name, value in feeds.items()})
        if self.fail is not None:
            raise self.fail
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
    def __init__(self, responses=(), *, fail_at=None):
        self.responses = tuple(responses)
        self.fail_at = fail_at
        self.calls = []

    def transcribe(self, samples, **kwargs):
        call_index = len(self.calls)
        self.calls.append((samples.copy(), kwargs))
        if call_index == self.fail_at:
            raise RuntimeError("private transcription failure")
        response = self.responses[call_index] if call_index < len(self.responses) else None
        duration = len(samples) / 16_000
        if response is None:
            chunks = ()
        else:
            text, confidence = response
            chunks = (SimpleNamespace(
                id=call_index,
                start=0.0,
                end=duration,
                text=text,
                tokens=(call_index + 1,),
                avg_logprob=math.log(confidence),
                no_speech_prob=0.1,
            ),)
        info = SimpleNamespace(language="en", language_probability=1.0, duration=duration)
        return iter(chunks), info


def loaded_transcriber(model):
    spec = WHISPER_TINY_EN
    metadata = TranscriptionModelMetadata(
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
    return LoadedTranscriptionModel(metadata, model)


def prepare_case(
    settings,
    consent,
    *,
    name="speech-integration",
    source_format="WAV",
    sample_rate_hz=48_000,
    channels=2,
    duration_seconds=1.6,
):
    settings.paths.raw_data.mkdir(parents=True, exist_ok=True)
    extension = source_format.lower()
    source_path = settings.paths.raw_data / f"{name}.{extension}"
    frame_count = round(sample_rate_hz * duration_seconds)
    timeline = np.arange(frame_count, dtype=np.float64) / sample_rate_hz
    first = 0.16 * np.sin(2 * np.pi * 220 * timeline)
    samples = first if channels == 1 else np.column_stack(
        (first, 0.12 * np.cos(2 * np.pi * 330 * timeline))
    )
    sf.write(source_path, samples, sample_rate_hz, format=source_format, subtype="PCM_16")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    audio_settings = AudioSettings(
        normalize_loudness=False,
        window_seconds=(1.0,),
        window_overlap_ratio=0.5,
    )
    input_audio = InputAudio(f"{name}-clip", source_path, consent)
    bundle = AudioPreparationService(settings, "speech-integration").prepare(
        input_audio,
        audio_settings=audio_settings,
        now=NOW,
    )
    return SimpleNamespace(
        settings=settings,
        paths=settings.paths,
        source_path=source_path,
        source_sha256=source_sha256,
        audio_settings=audio_settings,
        input_audio=input_audio,
        bundle=bundle,
    )


@pytest.fixture
def prepared_case(temporary_settings, active_consent):
    return prepare_case(temporary_settings, active_consent)


def run_branch(case, score_sets, responses, *, policy=None, now=NOW):
    vad_session = FakeVadSession(score_sets)
    segmentation = speech_segments.extract_prepared_speech_segments(
        case.paths,
        case.bundle.manifest_path.relative_to(case.paths.interim_data),
        loaded_vad(vad_session),
        reliability_policy=policy,
        now=now,
    )
    transcriber = FakeTranscriber(responses)
    result = speech_transcription.orchestrate_speech_transcription(
        case.paths,
        segmentation,
        loaded_transcriber(transcriber),
        now=now,
    )
    return SimpleNamespace(
        segmentation=segmentation,
        result=result,
        vad_session=vad_session,
        transcriber=transcriber,
    )


def tree_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_complete_branch_preserves_exact_overlapping_window_handoffs(prepared_case):
    branch = run_branch(
        prepared_case,
        ({20: 0.75}, {4: 0.9}, {}),
        (("accepted words", 0.9),),
    )
    segment = branch.segmentation.segments[0]
    stored, sample_rate = sf.read(prepared_case.bundle.audio.audio_path, dtype="float32")

    assert sample_rate == 16_000 and stored.ndim == 1
    assert len(branch.vad_session.calls) == 3
    assert len(segment.source_window_ids) == 2
    assert len(branch.transcriber.calls) == 1
    assert np.array_equal(
        branch.transcriber.calls[0][0], stored[segment.start_sample:segment.end_sample]
    )
    assert branch.result.accepted_segment_ids == ("speech-0000",)
    assert branch.result.downstream_transcripts[0].text == "accepted words"
    assert branch.result.evidence.source.clip_id == prepared_case.input_audio.clip_id
    assert branch.result.evidence.vad_model == loaded_vad(FakeVadSession()).metadata.as_speech_descriptor()


@pytest.mark.parametrize(
    "source_format,sample_rate_hz,channels",
    [
        ("WAV", 8_000, 1),
        ("WAV", 48_000, 2),
        ("FLAC", 16_000, 1),
        ("FLAC", 44_100, 2),
    ],
)
def test_format_rate_and_channel_matrix_reaches_accepted_evidence(
    temporary_settings,
    active_consent,
    source_format,
    sample_rate_hz,
    channels,
):
    case = prepare_case(
        temporary_settings,
        active_consent,
        name=f"matrix-{source_format.lower()}-{sample_rate_hz}-{channels}",
        source_format=source_format,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        duration_seconds=1.1,
    )
    branch = run_branch(case, ({0: 0.9}, {}), (("matrix accepted", 0.95),))
    stored, stored_rate = sf.read(
        case.bundle.audio.audio_path, dtype="float32", always_2d=True
    )

    assert case.bundle.manifest.source.format == source_format
    assert case.bundle.manifest.source.sample_rate_hz == sample_rate_hz
    assert case.bundle.manifest.source.channels == channels
    assert stored_rate == 16_000 and stored.shape == (17_600, 1)
    assert branch.result.accepted_segment_ids == ("speech-0000",)
    assert branch.result.evidence.source.processing_scope == "acoustic_and_speech"


def test_all_transcript_outcomes_remain_ordered_and_only_acceptance_flows(prepared_case):
    branch = run_branch(
        prepared_case,
        ({0: 0.9, 2: 0.9, 4: 0.9, 6: 0.9}, {}, {}),
        (("rejected", 0.49), ("review", 0.5), ("accepted", 0.8), None),
    )

    assert [item.segment_id for item in branch.segmentation.segments] == [
        "speech-0000", "speech-0001", "speech-0002", "speech-0003"
    ]
    assert branch.result.rejected_segment_ids == ("speech-0000",)
    assert branch.result.review_required_segment_ids == ("speech-0001",)
    assert branch.result.accepted_segment_ids == ("speech-0002",)
    assert branch.result.untranscribed_segment_ids == ("speech-0003",)
    assert [item.text for item in branch.result.downstream_transcripts] == ["accepted"]


def test_no_speech_skips_transcription_but_records_both_models(prepared_case):
    branch = run_branch(prepared_case, ({}, {}, {}), ())

    assert branch.segmentation.segments == ()
    assert branch.transcriber.calls == []
    assert branch.result.evidence.segment_count == 0
    assert branch.result.evidence.transcribed_segment_count == 0
    assert branch.result.evidence.transcription_model == loaded_transcriber(
        FakeTranscriber()
    ).metadata.as_speech_descriptor()
    assert branch.result.downstream_transcripts == ()


def test_tail_padding_is_removed_before_vad_and_transcription(
    temporary_settings, active_consent
):
    case = prepare_case(
        temporary_settings,
        active_consent,
        name="tail-padding",
        duration_seconds=1.1,
    )
    branch = run_branch(case, ({}, {18: 0.9}), (("tail", 0.9),))
    segment = branch.segmentation.segments[0]
    stored, _ = sf.read(case.bundle.audio.audio_path, dtype="float32")

    assert case.bundle.manifest.windows[-1].padding_samples == 6_400
    assert len(branch.vad_session.calls[-1]["input"]) == 19
    assert segment.end_sample == case.bundle.manifest.num_frames == 17_600
    assert branch.transcriber.calls[0][0].shape == (384,)
    assert np.array_equal(branch.transcriber.calls[0][0], stored[17_216:17_600])


def test_custom_policy_is_preserved_and_applied_across_both_stages(prepared_case):
    policy = SpeechReliabilityPolicy(
        vad_speech_threshold=0.7,
        transcript_review_threshold=0.6,
        transcript_acceptance_threshold=0.9,
    )
    branch = run_branch(
        prepared_case,
        ({0: 0.71}, {}, {}),
        (("needs review", 0.85),),
        policy=policy,
    )

    assert branch.segmentation.evidence.reliability_policy == policy
    assert branch.result.evidence.reliability_policy == policy
    assert branch.result.transcriptions[0].assessment.reliability is (
        TranscriptReliability.REVIEW_REQUIRED
    )
    assert branch.result.downstream_transcripts == ()


def test_fresh_services_and_models_reproduce_semantic_evidence(prepared_case):
    reused_bundle = AudioPreparationService(
        prepared_case.settings, "speech-integration"
    ).prepare(
        prepared_case.input_audio,
        audio_settings=prepared_case.audio_settings,
        now=NOW,
    )
    restarted_case = SimpleNamespace(
        **{**prepared_case.__dict__, "bundle": reused_bundle}
    )
    first = run_branch(
        restarted_case,
        ({20: 0.8}, {4: 0.9}, {}),
        (("stable", 0.9),),
        now=NOW,
    )
    second = run_branch(
        prepared_case,
        ({20: 0.8}, {4: 0.9}, {}),
        (("stable", 0.9),),
        now=LATER,
    )

    assert reused_bundle.reused is True
    assert reused_bundle.manifest_path == prepared_case.bundle.manifest_path
    assert first.segmentation.evidence.evidence_id == second.segmentation.evidence.evidence_id
    assert first.result.evidence.evidence_id == second.result.evidence.evidence_id
    assert first.result.evidence.created_at != second.result.evidence.created_at
    assert np.array_equal(first.transcriber.calls[0][0], second.transcriber.calls[0][0])


def test_branch_uses_persisted_bundle_after_raw_source_is_removed(prepared_case):
    prepared_case.source_path.unlink()

    branch = run_branch(
        prepared_case,
        ({0: 0.9}, {}, {}),
        (("from prepared audio", 0.9),),
    )

    assert branch.result.downstream_transcripts[0].text == "from prepared audio"


def test_branch_does_not_modify_raw_or_prepared_files(prepared_case):
    source_before = prepared_case.source_sha256
    bundle_before = tree_hashes(prepared_case.bundle.directory)

    run_branch(prepared_case, ({0: 0.9}, {}, {}), (("immutable inputs", 0.9),))

    assert hashlib.sha256(prepared_case.source_path.read_bytes()).hexdigest() == source_before
    assert tree_hashes(prepared_case.bundle.directory) == bundle_before


def test_vad_failure_stops_the_branch_before_any_transcription(prepared_case):
    vad_session = FakeVadSession(fail=RuntimeError("private VAD failure"))
    transcriber = FakeTranscriber((("must not run", 0.9),))

    with pytest.raises(speech_segments.SpeechSegmentationError) as error:
        speech_segments.extract_prepared_speech_segments(
            prepared_case.paths,
            prepared_case.bundle.manifest_path.relative_to(prepared_case.paths.interim_data),
            loaded_vad(vad_session),
            now=NOW,
        )

    assert error.value.code == "model_failed"
    assert "private" not in str(error.value)
    assert transcriber.calls == []


def test_transcription_failure_aborts_without_returning_partial_evidence(prepared_case):
    segmentation = speech_segments.extract_prepared_speech_segments(
        prepared_case.paths,
        prepared_case.bundle.manifest_path.relative_to(prepared_case.paths.interim_data),
        loaded_vad(FakeVadSession(({0: 0.9, 2: 0.9}, {}, {}))),
        now=NOW,
    )
    transcriber = FakeTranscriber((("first", 0.9), ("second", 0.9)), fail_at=1)

    with pytest.raises(speech_transcription.SpeechOrchestrationError) as error:
        speech_transcription.orchestrate_speech_transcription(
            prepared_case.paths,
            segmentation,
            loaded_transcriber(transcriber),
            now=NOW,
        )

    assert error.value.code == "model_failed"
    assert "private" not in str(error.value)
    assert len(transcriber.calls) == 2


def test_window_tamper_between_stages_is_rejected_before_transcription(prepared_case):
    segmentation = speech_segments.extract_prepared_speech_segments(
        prepared_case.paths,
        prepared_case.bundle.manifest_path.relative_to(prepared_case.paths.interim_data),
        loaded_vad(FakeVadSession(({0: 0.9}, {}, {}))),
        now=NOW,
    )
    target = prepared_case.paths.interim_data / segmentation.windows[0].window.audio_path
    target.write_bytes(target.read_bytes() + b"tampered")
    transcriber = FakeTranscriber((("must not run", 0.9),))

    with pytest.raises(speech_transcription.SpeechOrchestrationError) as error:
        speech_transcription.orchestrate_speech_transcription(
            prepared_case.paths,
            segmentation,
            loaded_transcriber(transcriber),
            now=NOW,
        )

    assert error.value.code == "source_mismatch"
    assert transcriber.calls == []


def test_final_evidence_round_trips_without_waveforms_or_absolute_paths(prepared_case):
    branch = run_branch(
        prepared_case,
        ({0: 0.9}, {}, {}),
        (("portable evidence", 0.9),),
    )
    encoded = branch.result.evidence.model_dump_json()
    summary = json.dumps(branch.result.to_summary(), sort_keys=True)

    assert SpeechEvidenceDocument.model_validate_json(encoded) == branch.result.evidence
    assert "waveform" not in summary
    assert str(prepared_case.paths.root) not in summary
    assert str(prepared_case.source_path) not in summary
