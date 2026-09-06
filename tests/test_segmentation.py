from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_sentinel.audio_arrays import AudioTransformError
from audio_sentinel.audio_loader import AudioLoadError, load_audio
from audio_sentinel.audio_transforms import NormalizationStats, PreparedSignal, prepare_signal
from audio_sentinel.config import AudioSettings
from audio_sentinel.contracts import PreparedClipRecord
from audio_sentinel.preparation import PreparedAudioManifest, SourceAudioMetadata
from audio_sentinel.segmentation import iter_windows


NOW = datetime(2026, 9, 6, tzinfo=UTC)


@pytest.fixture
def make_signal(active_consent):
    def make(frames=25_600, channels=1, **options):
        settings = AudioSettings(window_seconds=(1.0,), convert_to_mono=channels == 1, **options)
        samples = np.arange(frames * channels, dtype=np.float32).reshape(frames, channels) / 100_000
        source = SourceAudioMetadata(
            audio_path="synthetic/input.wav", sha256="a" * 64, format="WAV",
            sample_rate_hz=16_000, channels=channels, num_frames=frames, size_bytes=44 + frames * channels * 2,
        )
        return PreparedSignal(
            clip_id="clip-001", consent=active_consent, source=source, settings=settings,
            sample_rate_hz=16_000, samples=samples,
            normalization=NormalizationStats("disabled", 0, None, None, None, None),
        )
    return make


def spans(windows):
    return [(w.record.start_sample, w.record.end_sample, w.record.padding_samples) for w in windows]


@pytest.mark.parametrize("frames,policy,expected", [
    (1, "pad", [(0, 1, 15_999)]), (1, "drop", []),
    (12_000, "pad", [(0, 12_000, 4_000)]), (12_000, "drop", []),
    (16_000, "pad", [(0, 16_000, 0)]), (16_000, "drop", [(0, 16_000, 0)]),
    (24_000, "pad", [(0, 16_000, 0), (8_000, 24_000, 0)]),
    (25_600, "pad", [(0, 16_000, 0), (8_000, 24_000, 0), (16_000, 25_600, 6_400)]),
    (25_600, "drop", [(0, 16_000, 0), (8_000, 24_000, 0)]),
    (16_001, "pad", [(0, 16_000, 0), (8_000, 16_001, 7_999)]),
])
def test_sample_grid_and_tail_boundaries(make_signal, frames, policy, expected):
    signal = make_signal(frames=frames, tail_policy=policy)
    windows = list(iter_windows(signal, now=NOW))
    assert spans(windows) == expected
    for window in windows:
        start, end = window.record.start_sample, window.record.end_sample
        assert window.samples.shape == (16_000, 1)
        np.testing.assert_array_equal(window.samples[:end - start], signal.samples[start:end])
        assert not window.samples[end - start:].any()


def test_default_three_contexts_are_ordered_and_complete(make_signal):
    signal = make_signal()
    signal = replace(signal, settings=AudioSettings())
    windows = list(iter_windows(signal, now=NOW))
    assert [w.record.window_seconds for w in windows] == [1, 1, 1, 5, 10]
    assert [len(w.samples) for w in windows] == [16_000, 16_000, 16_000, 80_000, 160_000]
    assert [w.record.padding_samples for w in windows[-2:]] == [54_400, 134_400]


@pytest.mark.parametrize("overlap,expected_starts", [(0, [0, 16_000]), (0.25, [0, 12_000]), (0.75, [0, 4_000, 8_000, 12_000])])
def test_configurable_overlap(make_signal, overlap, expected_starts):
    windows = list(iter_windows(make_signal(window_overlap_ratio=overlap), now=NOW))
    assert [w.record.start_sample for w in windows] == expected_starts


def test_rounded_fractional_durations_and_one_sample_hop(make_signal):
    signal = make_signal(frames=7)
    signal = replace(signal, settings=AudioSettings(window_seconds=(0.00015625,), window_overlap_ratio=0.5))
    # 2.5 samples rounds to 2, then hop = round(2 * 0.5) = 1.
    windows = list(iter_windows(signal, now=NOW))
    assert spans(windows) == [(i, i + 2, 0) for i in range(6)]
    assert all(w.samples.shape == (2, 1) for w in windows)


def test_stereo_padding_and_independent_sample_storage(make_signal):
    signal = make_signal(channels=2)
    saved = signal.samples.copy()
    first, second, tail = list(iter_windows(signal, now=NOW))
    assert tail.samples.shape == (16_000, 2)
    np.testing.assert_array_equal(tail.samples[:9_600], signal.samples[16_000:])
    assert not tail.samples[9_600:].any()
    first.samples[:] = -99
    np.testing.assert_array_equal(second.samples, saved[8_000:24_000])
    np.testing.assert_array_equal(signal.samples, saved)


def test_ids_and_paths_are_deterministic_unique_and_portable(make_signal):
    signal = replace(make_signal(), clip_id="A:" + "x" * 126)
    first, second = list(iter_windows(signal, now=NOW)), list(iter_windows(signal, now=NOW))
    assert [w.record for w in first] == [w.record for w in second]
    assert len({w.record.window_id for w in first}) == len(first)
    assert len({w.record.audio_path for w in first}) == len(first)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.samples, b.samples)
        assert len(a.record.window_id) <= 128
        assert ":" not in a.record.audio_path
        assert "\\" not in a.record.audio_path
        assert a.clip_id == signal.clip_id
        assert a.sample_rate_hz == signal.sample_rate_hz


def test_distinct_groups_that_round_to_same_length_have_unique_ids(make_signal):
    signal = replace(make_signal(frames=4), settings=AudioSettings(window_seconds=(0.0001, 0.00011)))
    windows = list(iter_windows(signal, now=NOW))
    assert len({w.record.window_id for w in windows}) == len(windows)
    assert len({w.record.audio_path for w in windows}) == len(windows)


def test_source_or_settings_changes_produce_different_output_namespace(make_signal):
    signal = make_signal()
    reference = next(iter_windows(signal, now=NOW)).record.audio_path
    changed_source = replace(signal, source=signal.source.model_copy(update={"sha256": "b" * 64}))
    changed_settings = replace(signal, settings=AudioSettings(window_seconds=(1.0,), window_overlap_ratio=0))
    for changed in (changed_source, changed_settings, replace(signal, clip_id="clip-002")):
        assert next(iter_windows(changed, now=NOW)).record.audio_path != reference


def test_allocates_one_window_at_a_time(make_signal, monkeypatch):
    from audio_sentinel import segmentation
    signal = make_signal(frames=1)
    signal = replace(signal, settings=AudioSettings())
    calls = []
    original = np.zeros
    def zeros(shape, **kwargs):
        calls.append(shape)
        return original(shape, **kwargs)
    monkeypatch.setattr(segmentation.np, "zeros", zeros)
    windows = iter_windows(signal, now=NOW)
    assert calls == []
    next(windows)
    assert calls == [(16_000, 1)]
    next(windows)
    assert calls == [(16_000, 1), (80_000, 1)]
    windows.close()


def test_oversized_padded_window_is_rejected_before_iteration(make_signal):
    signal = make_signal(frames=1, max_decoded_bytes=100)
    with pytest.raises(AudioTransformError) as caught:
        iter_windows(signal, now=NOW)
    assert caught.value.code == "decoded_audio_too_large"
    # A non-emitted window in drop mode needs no allocation.
    signal = replace(signal, settings=AudioSettings(window_seconds=(1.0,), max_decoded_bytes=100, tail_policy="drop"))
    assert list(iter_windows(signal, now=NOW)) == []


@pytest.mark.parametrize("change,code", [
    ({"sample_rate_hz": 8_000}, "sample_rate_mismatch"),
    ({"samples": np.zeros((20, 1), dtype=np.float32)}, "source_mismatch"),
    ({"samples": np.empty((0, 1), dtype=np.float32)}, "invalid_samples"),
    ({"samples": np.array([[np.nan]], dtype=np.float32)}, "non_finite_audio"),
    ({"clip_id": "../unsafe"}, "invalid_clip_id"),
])
def test_invalid_prepared_signal_is_rejected_eagerly(make_signal, change, code):
    with pytest.raises(AudioTransformError) as caught:
        iter_windows(replace(make_signal(), **change), now=NOW)
    assert caught.value.code == code


def test_consent_expiry_is_rechecked_when_iterator_resumes(make_signal, monkeypatch):
    from audio_sentinel import segmentation
    signal = make_signal()
    signal = replace(signal, consent=signal.consent.model_copy(update={"expires_at": NOW + timedelta(seconds=1)}))
    times = iter([NOW, NOW, NOW + timedelta(seconds=1)])
    class Clock:
        @staticmethod
        def now(tz):
            return next(times)
    monkeypatch.setattr(segmentation, "datetime", Clock)
    windows = iter_windows(signal)
    next(windows)
    with pytest.raises(AudioLoadError) as caught:
        next(windows)
    assert caught.value.code == "consent_expired"


@pytest.mark.parametrize("policy", ["pad", "drop"])
def test_generated_records_are_accepted_by_manifest(make_signal, policy):
    signal = replace(make_signal(tail_policy=policy), settings=AudioSettings(tail_policy=policy))
    windows = list(iter_windows(signal, now=NOW))
    manifest = PreparedAudioManifest(
        source=signal.source, settings=signal.settings, channels=signal.channels, num_frames=signal.num_frames,
        clip=PreparedClipRecord(clip_id=signal.clip_id, source_dataset="synthetic", audio_path="prepared/full.wav",
                                sample_rate_hz=signal.sample_rate_hz, duration_seconds=signal.duration_seconds,
                                consent=signal.consent),
        windows=tuple(w.record for w in windows),
    )
    assert len(manifest.windows) == len(windows)


def test_loader_through_preparation_and_segmentation(temporary_settings, active_consent):
    from audio_sentinel.interfaces import InputAudio
    temporary_settings.ensure_directories()
    path = temporary_settings.paths.raw_data / "synthetic.wav"
    sf.write(path, np.zeros((76_800, 2), dtype=np.float32), 48_000, subtype="PCM_16")
    loaded = load_audio(InputAudio("clip-001", Path("synthetic.wav"), active_consent), temporary_settings, now=NOW)
    prepared = prepare_signal(loaded, temporary_settings.audio, now=NOW)
    windows = list(iter_windows(prepared, now=NOW))
    assert prepared.num_frames == 25_600
    assert [w.record.window_seconds for w in windows] == [1, 1, 1, 5, 10]
    assert windows[2].record.padding_samples == 6_400
    assert list(temporary_settings.paths.interim_data.iterdir()) == []
