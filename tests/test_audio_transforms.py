from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_sentinel.audio_loader import AudioLoadError, load_audio
from audio_sentinel.audio_transforms import (
    AudioTransformError, convert_to_mono, normalize_loudness, prepare_signal, resample_audio,
)
from audio_sentinel.config import AudioSettings
from audio_sentinel.interfaces import InputAudio


NOW = datetime(2026, 9, 5, 12, tzinfo=UTC)
BUDGET = 1_000_000


def resample(samples, source, target, budget=BUDGET):
    return resample_audio(samples, source, target, max_decoded_bytes=budget)


def tone(frequency, rate, seconds=1, amplitude=0.2):
    return (amplitude * np.sin(2 * np.pi * frequency * np.arange(round(rate * seconds)) / rate)).astype(np.float32)[:, None]


def test_mono_averages_all_channels_without_modifying_input():
    original = np.array([[1, 0, -0.5], [-1, 1, 0.75]], dtype=np.float32)
    saved = original.copy()
    result = convert_to_mono(original)
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result[:, 0], [1 / 6, 0.25])
    np.testing.assert_array_equal(original, saved)
    assert not np.shares_memory(result, original)


def test_opposite_phase_channels_cancel_and_large_values_do_not_overflow():
    np.testing.assert_array_equal(convert_to_mono(np.array([[0.5, -0.5]], dtype=np.float32)), [[0]])
    large = np.full((2, 8), np.finfo(np.float32).max, dtype=np.float32)
    np.testing.assert_array_equal(convert_to_mono(large), large[:, :1])


@pytest.mark.parametrize("frames,source,target,expected", [
    (48_000, 48_000, 16_000, 16_000), (44_100, 44_100, 16_000, 16_000),
    (8_000, 8_000, 16_000, 16_000), (7, 16_000, 16_000, 7),
    (4, 44_100, 16_000, 1), (9, 48_000, 8_000, 2), (15, 48_000, 8_000, 2),
    (1, 8_000, 16_000, 2),
])
def test_resampling_frame_count_obeys_contract(frames, source, target, expected):
    samples = np.zeros((frames, 2), dtype=np.float32)
    result = resample(samples, source, target)
    assert result.shape == (expected, 2)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert not np.shares_memory(result, samples)


@pytest.mark.parametrize("source,target", [(48_000, 16_000), (44_100, 16_000), (8_000, 16_000)])
def test_resampling_preserves_tone_pitch_and_interior_amplitude(source, target):
    result = resample(tone(1_000, source), source, target)[:, 0]
    # Measure pitch independently, rather than comparing to another resample_poly call.
    frequency = np.fft.rfftfreq(len(result), 1 / target)[np.argmax(abs(np.fft.rfft(result)))]
    assert frequency == pytest.approx(1_000, abs=1)
    rms = np.sqrt(np.mean(result[100:-100].astype(np.float64) ** 2))
    assert rms == pytest.approx(0.2 / np.sqrt(2), rel=0.015)


def test_downsampling_rejects_out_of_band_tone_instead_of_aliasing():
    result = resample(tone(12_000, 48_000), 48_000, 16_000)[100:-100]
    assert np.sqrt(np.mean(result.astype(np.float64) ** 2)) < 0.002


def test_resampling_preserves_channel_separation():
    samples = np.concatenate([tone(500, 48_000), tone(2_000, 48_000)], axis=1)
    result = resample(samples, 48_000, 16_000)
    frequencies = np.fft.rfftfreq(len(result), 1 / 16_000)
    assert frequencies[np.argmax(abs(np.fft.rfft(result[:, 0])))] == 500
    assert frequencies[np.argmax(abs(np.fft.rfft(result[:, 1])))] == 2_000


def test_same_rate_is_an_exact_independent_copy():
    samples = tone(1_000, 16_000)
    result = resample(samples, 16_000, 16_000)
    np.testing.assert_array_equal(samples, result)
    assert not np.shares_memory(samples, result)


def test_output_memory_limit_is_checked_before_upsampling(monkeypatch):
    from audio_sentinel import audio_transforms
    monkeypatch.setattr(audio_transforms, "resample_poly", lambda *a, **kw: pytest.fail("Must reject before allocation"))
    with pytest.raises(AudioTransformError) as caught:
        resample(np.zeros((100, 1), dtype=np.float32), 8_000, 192_000, budget=1_000)
    assert caught.value.code == "decoded_audio_too_large"


@pytest.mark.parametrize("rate", [0, 7_999, 192_001, 16_000.5, True])
def test_invalid_resampling_rate_is_rejected(rate):
    with pytest.raises(AudioTransformError, match="Sample rates"):
        resample(np.zeros((10, 1), dtype=np.float32), rate, 16_000)


def test_resampling_that_would_produce_zero_frames_is_rejected():
    with pytest.raises(AudioTransformError) as caught:
        resample(np.zeros((1, 1), dtype=np.float32), 192_000, 8_000)
    assert caught.value.code == "audio_too_short"


@pytest.mark.parametrize("samples", [
    np.empty((0, 1), dtype=np.float32), np.zeros((5, 0), dtype=np.float32),
    np.zeros(5, dtype=np.float32), np.zeros((5, 1), dtype=np.int16),
    np.array([[np.nan]], dtype=np.float32), np.array([[np.inf]], dtype=np.float32),
])
def test_transform_boundaries_reject_invalid_samples(samples):
    for transform in (convert_to_mono, lambda x: resample(x, 16_000, 8_000),
                      lambda x: normalize_loudness(x, AudioSettings())):
        with pytest.raises(AudioTransformError):
            transform(samples)


def test_normalization_reaches_rms_target_without_mutating_source():
    samples = tone(1_000, 16_000, amplitude=0.5)
    original = samples.copy()
    result = normalize_loudness(samples, AudioSettings())
    assert result.stats.reason == "target"
    assert result.stats.output_rms_dbfs == pytest.approx(-20, abs=1e-5)
    assert result.stats.gain_db < 0
    np.testing.assert_array_equal(samples, original)
    assert result.samples.dtype == np.float32


def test_normalization_uses_one_gain_preserving_stereo_balance():
    samples = np.tile(np.array([[0.4, 0.1], [-0.4, -0.1]], dtype=np.float32), (100, 1))
    result = normalize_loudness(samples, AudioSettings())
    np.testing.assert_allclose(result.samples[:, 0], result.samples[:, 1] * 4)
    assert result.stats.output_rms_dbfs == pytest.approx(-20, abs=1e-5)


def test_peak_limit_protects_sparse_transients_with_a_uniform_gain():
    samples = np.zeros((16_000, 1), dtype=np.float32)
    samples[8_000] = 2
    samples[8_001] = 0.5
    result = normalize_loudness(samples, AudioSettings())
    assert result.stats.reason == "peak_limit"
    assert result.stats.output_peak_dbfs <= -1
    assert result.stats.output_rms_dbfs < -20
    assert result.samples[8_000, 0] / result.samples[8_001, 0] == pytest.approx(4)


def test_gain_limit_prevents_excessive_amplification():
    result = normalize_loudness(np.full((100, 1), 0.002, dtype=np.float32), AudioSettings())
    assert result.stats.reason == "gain_limit"
    assert result.stats.gain_db == pytest.approx(20)
    np.testing.assert_allclose(result.samples, 0.02, rtol=1e-6)


@pytest.mark.parametrize("level", [0, 1e-6, 0.001])
def test_silence_floor_is_unchanged_and_zero_metrics_are_finite_or_none(level):
    samples = np.full((100, 1), level, dtype=np.float32)
    result = normalize_loudness(samples, AudioSettings())
    assert result.stats.reason == "silence"
    assert result.stats.gain_db == 0
    np.testing.assert_array_equal(samples, result.samples)
    if level == 0:
        assert result.stats.output_rms_dbfs is None
        assert result.stats.output_peak_dbfs is None


def test_disabled_normalization_preserves_values_above_full_scale():
    samples = np.array([[2], [-3], [0]], dtype=np.float32)
    result = normalize_loudness(samples, AudioSettings(normalize_loudness=False))
    np.testing.assert_array_equal(result.samples, samples)
    assert result.stats.reason == "disabled"
    assert not np.shares_memory(result.samples, samples)


def test_large_finite_float_input_does_not_overflow_rms_measurement():
    samples = np.full((100, 2), np.finfo(np.float32).max, dtype=np.float32)
    result = normalize_loudness(samples, AudioSettings())
    assert np.isfinite(result.samples).all()
    assert result.stats.output_rms_dbfs == pytest.approx(-20, abs=1e-5)


@pytest.fixture
def loaded_clip(temporary_settings, active_consent):
    temporary_settings.ensure_directories()
    path = temporary_settings.paths.raw_data / "test.wav"
    samples = np.concatenate([tone(500, 48_000, amplitude=0.5), tone(500, 48_000, amplitude=0.25)], axis=1)
    sf.write(path, samples, 48_000, subtype="PCM_16")
    return load_audio(InputAudio("clip-001", Path("test.wav"), active_consent), temporary_settings, now=NOW)


def test_loader_to_transforms_preserves_provenance_and_matches_manifest_dimensions(loaded_clip, temporary_settings):
    original = loaded_clip.samples.copy()
    result = prepare_signal(loaded_clip, AudioSettings(), now=NOW)
    assert result.samples.shape == (16_000, 1)
    assert result.duration_seconds == 1
    assert result.sample_rate_hz == 16_000
    assert result.source.sample_rate_hz == 48_000
    assert result.source.channels == 2
    assert result.source == loaded_clip.source
    assert result.clip_id == loaded_clip.clip_id
    assert result.consent == loaded_clip.consent
    assert result.normalization.output_rms_dbfs == pytest.approx(-20, abs=1e-5)
    np.testing.assert_array_equal(loaded_clip.samples, original)
    assert list(temporary_settings.paths.interim_data.iterdir()) == []
    repeat = prepare_signal(loaded_clip, AudioSettings(), now=NOW)
    np.testing.assert_array_equal(result.samples, repeat.samples)


def test_optional_mono_and_normalization_can_both_be_disabled(loaded_clip):
    settings = AudioSettings(convert_to_mono=False, normalize_loudness=False, target_sample_rate_hz=48_000)
    result = prepare_signal(loaded_clip, settings, now=NOW)
    np.testing.assert_array_equal(result.samples, loaded_clip.samples)
    assert result.channels == 2


def test_noise_reduction_is_not_silently_ignored(loaded_clip):
    with pytest.raises(AudioTransformError) as caught:
        prepare_signal(loaded_clip, AudioSettings(noise_reduction={"enabled": True}), now=NOW)
    assert caught.value.code == "noise_reduction_unavailable"


def test_mismatched_loaded_shape_is_rejected(loaded_clip):
    with pytest.raises(AudioTransformError) as caught:
        prepare_signal(replace(loaded_clip, samples=loaded_clip.samples[:-1]), AudioSettings(), now=NOW)
    assert caught.value.code == "source_mismatch"


def test_expired_permission_prevents_processing(loaded_clip):
    consent = loaded_clip.consent.model_copy(update={"expires_at": NOW - timedelta(seconds=1)})
    with pytest.raises(AudioLoadError) as caught:
        prepare_signal(replace(loaded_clip, consent=consent), AudioSettings(), now=NOW)
    assert caught.value.code == "consent_expired"
