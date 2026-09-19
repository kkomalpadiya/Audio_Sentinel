"""A4.2 tests for verified speech-window scoring and sample-exact extraction."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
import hashlib
import json

import numpy as np
import pytest
from pydantic import ValidationError
import soundfile as sf

from audio_sentinel import speech_segments as segments
from audio_sentinel.config import AudioSettings
from audio_sentinel.contracts import ConsentRecord, ConsentStatus, ProcessingScope
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.pipeline import AudioPreparationService
from audio_sentinel.speech_contracts import SpeechReliabilityPolicy, TranscriptReliability
from audio_sentinel.vad import (
    SILERO_VAD,
    LoadedVadModel,
    VadModelMetadata,
    VadTensorContract,
)


NOW = datetime(2026, 9, 19, 12, tzinfo=UTC)


class FakeVadSession:
    def __init__(self, score_sets=(), *, fail=None, on_run=None):
        self.score_sets = tuple(score_sets)
        self.fail = fail
        self.on_run = on_run
        self.calls = []

    def run(self, output_names, feeds):
        call_index = len(self.calls)
        self.calls.append({name: value.copy() for name, value in feeds.items()})
        if self.on_run:
            self.on_run(call_index)
        if self.fail:
            raise self.fail
        size = len(feeds["input"])
        scores = np.full(size, np.float32(0.1), dtype=np.float32)
        values = self.score_sets[call_index] if call_index < len(self.score_sets) else {}
        for index, score in values.items():
            scores[index] = np.float32(score)
        state = np.zeros((1, 1, SILERO_VAD.state_size), dtype=np.float32)
        return scores, state, state.copy()


def loaded_model(session=None, metadata=None):
    metadata = metadata or VadModelMetadata(
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
    return LoadedVadModel(metadata=metadata, session=session or FakeVadSession())


@pytest.fixture
def prepared(temporary_settings, active_consent):
    temporary_settings.paths.raw_data.mkdir(parents=True)
    path = temporary_settings.paths.raw_data / "speech-source.wav"
    samples = 0.2 * np.sin(2 * np.pi * 220 * np.arange(25_600) / 16_000)
    sf.write(path, samples, 16_000, subtype="PCM_16")
    bundle = AudioPreparationService(temporary_settings, "synthetic-speech").prepare(
        InputAudio("speech-segment-clip", path, active_consent),
        audio_settings=AudioSettings(
            normalize_loudness=False,
            window_seconds=(1.0, 5.0),
            window_overlap_ratio=0.5,
        ),
        now=NOW,
    )
    return temporary_settings.paths, bundle


def run(prepared, session=None, **kwargs):
    paths, bundle = prepared
    model = loaded_model(session or FakeVadSession())
    result = segments.extract_prepared_speech_segments(
        paths,
        bundle.manifest_path.relative_to(paths.interim_data),
        model,
        now=NOW,
        **kwargs,
    )
    return result, model


def selected_windows(bundle, seconds=1.0):
    return tuple(item for item in bundle.manifest.windows if item.window_seconds == seconds)


def test_extracts_absolute_non_overlapping_segment_from_overlapping_windows(prepared):
    session = FakeVadSession(({20: 0.7}, {4: 0.9}, {}))

    result, _ = run(prepared, session)

    segment = result.segments[0]
    source_windows = selected_windows(prepared[1])
    assert result.selected_window_seconds == 1.0
    assert result.input_frame_count == 83
    assert len(result.windows) == 3 and len(session.calls) == 3
    assert (segment.start_sample, segment.end_sample) == (10_048, 10_752)
    assert (segment.start_seconds, segment.end_seconds) == (0.628, 0.672)
    assert segment.vad_score == pytest.approx(0.9)
    assert segment.source_window_ids == (
        source_windows[0].window_id,
        source_windows[1].window_id,
    )
    assert [(item.start_sample, item.end_sample) for item in segment.contributions] == [
        (10_048, 10_560),
        (10_240, 10_752),
    ]


def test_builds_transcript_free_a4_1_evidence_with_complete_provenance(prepared):
    result, model = run(prepared, FakeVadSession(({0: 0.8}, {}, {})))
    evidence = result.evidence
    paths, bundle = prepared

    assert len(evidence.evidence_id) == 64
    assert evidence.created_at == NOW
    assert evidence.source.clip_id == "speech-segment-clip"
    assert evidence.source.consent_id == "test-consent-001"
    assert evidence.source.processing_scope == "acoustic_and_speech"
    assert evidence.source.preparation_manifest_sha256 == hashlib.sha256(
        bundle.manifest_path.read_bytes()
    ).hexdigest()
    assert evidence.source.preparation_manifest_path == bundle.manifest_path.relative_to(
        paths.interim_data
    ).as_posix()
    assert evidence.vad_model == model.metadata.as_speech_descriptor()
    assert evidence.transcription_model is None
    assert evidence.input_window_count == 3
    assert evidence.segment_count == 1 and evidence.transcribed_segment_count == 0
    assert evidence.segments[0].transcript is None
    assert evidence.segments[0].assessment.reliability is TranscriptReliability.NOT_TRANSCRIBED


def test_threshold_is_inclusive_and_below_threshold_is_excluded(prepared):
    equal, _ = run(prepared, FakeVadSession(({0: 0.6}, {}, {})))
    below, _ = run(prepared, FakeVadSession(({0: np.nextafter(np.float32(0.6), 0)}, {}, {})))

    assert len(equal.segments) == 1
    assert equal.segments[0].vad_score == pytest.approx(0.6)
    assert below.segments == () and below.evidence.segments == ()


def test_adjacent_frames_merge_and_separated_frames_do_not(prepared):
    adjacent, _ = run(prepared, FakeVadSession(({0: 0.8, 1: 0.7}, {}, {})))
    separated, _ = run(prepared, FakeVadSession(({0: 0.8, 2: 0.7}, {}, {})))

    assert [(item.start_sample, item.end_sample) for item in adjacent.segments] == [(0, 1_024)]
    assert [(item.start_sample, item.end_sample) for item in separated.segments] == [
        (0, 512),
        (1_024, 1_536),
    ]


@pytest.mark.parametrize("gap,expected", [(511, 2), (512, 1)])
def test_merge_gap_uses_an_exact_inclusive_sample_boundary(prepared, gap, expected):
    settings = segments.SpeechSegmentationSettings(merge_gap_samples=gap)
    result, _ = run(
        prepared,
        FakeVadSession(({0: 0.8, 2: 0.7}, {}, {})),
        settings=settings,
    )
    assert len(result.segments) == expected


def test_minimum_duration_filters_after_frame_union(prepared):
    settings = segments.SpeechSegmentationSettings(minimum_speech_samples=513)
    result, _ = run(prepared, FakeVadSession(({0: 0.8}, {}, {})), settings=settings)
    assert result.segments == ()


def test_segment_limit_fails_instead_of_returning_partial_output(prepared):
    settings = segments.SpeechSegmentationSettings(max_segments=1)
    with pytest.raises(segments.SpeechSegmentationError) as error:
        run(
            prepared,
            FakeVadSession(({0: 0.8, 2: 0.7}, {}, {})),
            settings=settings,
        )
    assert error.value.code == "too_many_segments"


def test_tail_frame_stops_at_real_clip_end_and_never_includes_padding(prepared):
    result, _ = run(prepared, FakeVadSession(({}, {}, {18: 0.85})))
    segment = result.segments[0]
    assert (segment.start_sample, segment.end_sample) == (25_216, 25_600)
    assert segment.end_seconds == 1.6
    assert result.windows[-1].vad.frames[-1].padding_samples == 128


def test_explicit_available_duration_uses_only_that_complete_window_group(prepared):
    settings = segments.SpeechSegmentationSettings(window_seconds=5.0)
    session = FakeVadSession(({0: 0.8},))
    result, _ = run(prepared, session, settings=settings)

    assert result.selected_window_seconds == 5.0
    assert len(result.windows) == len(session.calls) == 1
    assert result.input_frame_count == 50
    assert result.evidence.input_window_count == 1


def test_evidence_identity_is_stable_across_creation_times(prepared):
    first, _ = run(prepared, FakeVadSession(({0: 0.8}, {}, {})))
    paths, bundle = prepared
    second = segments.extract_prepared_speech_segments(
        paths,
        bundle.manifest_path.relative_to(paths.interim_data),
        loaded_model(FakeVadSession(({0: 0.8}, {}, {}))),
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )
    assert first.evidence.evidence_id == second.evidence.evidence_id
    assert first.evidence.created_at != second.evidence.created_at


def test_summary_is_json_ready_and_contains_no_waveform_or_absolute_path(prepared):
    result, _ = run(prepared, FakeVadSession(({0: 0.8}, {}, {})))
    document = json.dumps(result.to_summary(), sort_keys=True)
    assert json.loads(document)["segment_count"] == 1
    assert "waveform" not in document
    assert str(prepared[0].root) not in document


def test_input_samples_and_source_files_are_not_modified(prepared):
    paths, bundle = prepared
    before = {
        item.audio_path: hashlib.sha256((paths.interim_data / item.audio_path).read_bytes()).hexdigest()
        for item in selected_windows(bundle)
    }
    run(prepared, FakeVadSession(({0: 0.8}, {}, {})))
    after = {
        item.audio_path: hashlib.sha256((paths.interim_data / item.audio_path).read_bytes()).hexdigest()
        for item in selected_windows(bundle)
    }
    assert before == after


def test_public_result_types_are_immutable(prepared):
    result, _ = run(prepared, FakeVadSession(({0: 0.8}, {}, {})))
    with pytest.raises(FrozenInstanceError):
        result.input_frame_count = 0
    with pytest.raises(FrozenInstanceError):
        result.segments[0].start_sample = 1
    with pytest.raises(ValidationError):
        result.evidence.segment_count = 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"window_seconds": 0},
        {"window_seconds": float("nan")},
        {"merge_gap_samples": -1},
        {"merge_gap_samples": True},
        {"minimum_speech_samples": 0},
        {"max_manifest_bytes": 0},
        {"max_windows": 0},
        {"max_frames": True},
        {"unexpected": 1},
    ],
)
def test_invalid_segmentation_settings_are_rejected(overrides):
    with pytest.raises(ValidationError):
        segments.SpeechSegmentationSettings(**overrides)


def test_rejects_unavailable_window_duration_before_model_calls(prepared):
    session = FakeVadSession()
    with pytest.raises(segments.SpeechSegmentationError) as error:
        run(
            prepared,
            session,
            settings=segments.SpeechSegmentationSettings(window_seconds=2.0),
        )
    assert error.value.code == "window_duration_unavailable" and session.calls == []


def test_drop_tail_that_does_not_cover_clip_is_rejected(temporary_settings, active_consent):
    temporary_settings.paths.raw_data.mkdir(parents=True)
    path = temporary_settings.paths.raw_data / "short.wav"
    sf.write(path, np.zeros(8_000), 16_000, subtype="PCM_16")
    bundle = AudioPreparationService(temporary_settings, "short").prepare(
        InputAudio("short-speech-clip", path, active_consent),
        audio_settings=AudioSettings(window_seconds=(1.0,), tail_policy="drop"),
        now=NOW,
    )
    session = FakeVadSession()
    with pytest.raises(segments.SpeechSegmentationError) as error:
        segments.extract_prepared_speech_segments(
            temporary_settings.paths,
            bundle.manifest_path.relative_to(temporary_settings.paths.interim_data),
            loaded_model(session),
            now=NOW,
        )
    assert error.value.code == "incomplete_window_coverage" and session.calls == []


@pytest.mark.parametrize(
    "settings,code",
    [
        (segments.SpeechSegmentationSettings(max_windows=2), "too_many_windows"),
        (segments.SpeechSegmentationSettings(max_frames=82), "too_many_frames"),
        (segments.SpeechSegmentationSettings(max_source_bytes=1), "source_too_large"),
        (segments.SpeechSegmentationSettings(max_decoded_bytes=1), "decoded_audio_too_large"),
    ],
)
def test_resource_limits_fail_without_partial_results(prepared, settings, code):
    session = FakeVadSession()
    with pytest.raises(segments.SpeechSegmentationError) as error:
        run(prepared, session, settings=settings)
    assert error.value.code == code and session.calls == []


def test_rejects_wrong_sample_rate_and_non_mono_sources(temporary_settings, active_consent):
    temporary_settings.paths.raw_data.mkdir(parents=True)
    source = temporary_settings.paths.raw_data / "wrong-shape.wav"
    sf.write(source, np.zeros((8_000, 2)), 8_000, subtype="PCM_16")
    clip = InputAudio("wrong-shape-clip", source, active_consent)

    rate_bundle = AudioPreparationService(temporary_settings, "wrong-rate").prepare(
        clip,
        audio_settings=AudioSettings(
            target_sample_rate_hz=8_000,
            convert_to_mono=True,
            window_seconds=(1.0,),
        ),
        now=NOW,
    )
    with pytest.raises(segments.SpeechSegmentationError) as rate_error:
        segments.extract_prepared_speech_segments(
            temporary_settings.paths,
            rate_bundle.manifest_path.relative_to(temporary_settings.paths.interim_data),
            loaded_model(),
            now=NOW,
        )
    assert rate_error.value.code == "sample_rate_mismatch"

    mono_bundle = AudioPreparationService(temporary_settings, "wrong-channels").prepare(
        clip,
        audio_settings=AudioSettings(
            target_sample_rate_hz=16_000,
            convert_to_mono=False,
            window_seconds=(1.0,),
        ),
        now=NOW,
    )
    with pytest.raises(segments.SpeechSegmentationError) as mono_error:
        segments.extract_prepared_speech_segments(
            temporary_settings.paths,
            mono_bundle.manifest_path.relative_to(temporary_settings.paths.interim_data),
            loaded_model(),
            now=NOW,
        )
    assert mono_error.value.code == "mono_required"


def test_rejects_unverified_model_before_reading_sources(prepared):
    paths, bundle = prepared
    changed = replace(loaded_model().metadata, artifact_sha256="0" * 64)
    missing = paths.interim_data / "missing.json"
    with pytest.raises(segments.SpeechSegmentationError) as error:
        segments.extract_prepared_speech_segments(
            paths,
            missing.relative_to(paths.interim_data),
            loaded_model(metadata=changed),
            now=NOW,
        )
    assert error.value.code == "model_mismatch"
    assert bundle.manifest_path.exists()


def test_model_failure_is_wrapped_with_safe_stable_error(prepared):
    session = FakeVadSession(fail=RuntimeError("private runtime detail"))
    with pytest.raises(segments.SpeechSegmentationError, match="could not score") as error:
        run(prepared, session)
    assert error.value.code == "model_failed"
    assert "private" not in str(error.value)


def test_detects_manifest_change_during_segmentation(prepared):
    _, bundle = prepared

    def mutate(call_index):
        if call_index == 0:
            bundle.manifest_path.write_bytes(bundle.manifest_path.read_bytes() + b"\n")

    with pytest.raises(segments.SpeechSegmentationError) as error:
        run(prepared, FakeVadSession(on_run=mutate))
    assert error.value.code == "source_changed"


def test_detects_window_change_during_segmentation(prepared):
    paths, bundle = prepared
    target = paths.interim_data / selected_windows(bundle)[0].audio_path

    def mutate(call_index):
        if call_index == 0:
            target.write_bytes(target.read_bytes() + b"changed")

    with pytest.raises(segments.SpeechSegmentationError) as error:
        run(prepared, FakeVadSession(on_run=mutate))
    assert error.value.code == "source_changed"


def test_rejects_nonzero_prepared_padding(prepared):
    paths, bundle = prepared
    tail = selected_windows(bundle)[-1]
    target = paths.interim_data / tail.audio_path
    samples, rate = sf.read(target, dtype="int16", always_2d=False)
    samples[-1] = 1
    sf.write(target, samples, rate, subtype="PCM_16")
    with pytest.raises(segments.SpeechSegmentationError) as error:
        run(prepared)
    assert error.value.code == "source_mismatch"


def test_naive_creation_time_is_rejected_before_io(prepared):
    paths, _ = prepared
    with pytest.raises(segments.SpeechSegmentationError) as error:
        segments.extract_prepared_speech_segments(
            paths,
            "missing.json",
            loaded_model(),
            now=datetime(2026, 9, 19),
        )
    assert error.value.code == "invalid_time"


def test_acoustic_only_consent_is_rejected_before_model_calls(temporary_settings):
    temporary_settings.paths.raw_data.mkdir(parents=True)
    path = temporary_settings.paths.raw_data / "acoustic-only.wav"
    sf.write(path, np.zeros(16_000), 16_000, subtype="PCM_16")
    consent = ConsentRecord(
        consent_id="acoustic-only-consent",
        status=ConsentStatus.GRANTED,
        processing_scope=ProcessingScope.ACOUSTIC_ONLY,
        device_authorized=True,
        granted_at=NOW,
    )
    bundle = AudioPreparationService(temporary_settings, "acoustic-only").prepare(
        InputAudio("acoustic-only-clip", path, consent),
        audio_settings=AudioSettings(window_seconds=(1.0,)),
        now=NOW,
    )
    session = FakeVadSession()
    with pytest.raises(segments.SpeechSegmentationError) as error:
        segments.extract_prepared_speech_segments(
            temporary_settings.paths,
            bundle.manifest_path.relative_to(temporary_settings.paths.interim_data),
            loaded_model(session),
            now=NOW,
        )
    assert getattr(error.value, "code", None) == "scope_not_allowed"
    assert session.calls == []
