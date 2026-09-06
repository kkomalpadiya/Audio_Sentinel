from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
import soundfile as sf

from audio_sentinel.audio_arrays import AudioTransformError
from audio_sentinel.audio_loader import load_audio
from audio_sentinel.audio_transforms import convert_to_mono, normalize_loudness, prepare_signal, resample_audio
from audio_sentinel.config import AudioSettings, NoiseReductionSettings
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.noise_reduction import NoiseReducedAudio, NoiseReductionStats, reduce_noise


RATE = 16_000
NOW = datetime(2026, 9, 5, 12, tzinfo=UTC)


def reduce(samples, **overrides):
    return reduce_noise(samples, RATE, NoiseReductionSettings(enabled=True, **overrides), max_decoded_bytes=2_000_000)


@pytest.fixture
def noisy_tone():
    clean = np.zeros(RATE * 2)
    clean[8_000:24_000] = 0.3 * np.sin(2 * np.pi * 1_000 * np.arange(RATE) / RATE)
    noise = np.random.default_rng(22).normal(0, 0.03, clean.size)
    return clean, (clean + noise).astype(np.float32)[:, None]


def test_reduces_stationary_background_while_preserving_intermittent_tone(noisy_tone):
    clean, samples = noisy_tone
    saved = samples.copy()
    result = reduce(samples, reduction_strength=0.8)
    original_error = np.sum((samples[:, 0] - clean) ** 2)
    output_error = np.sum((result.samples[:, 0] - clean) ** 2)
    assert 10 * np.log10(original_error / output_error) > 2
    assert np.sqrt(np.mean(result.samples[1000:7000] ** 2)) < 0.75 * np.sqrt(np.mean(samples[1000:7000] ** 2))
    active = result.samples[10_000:22_000, 0]
    frequency = np.fft.rfftfreq(len(active), 1 / RATE)[np.argmax(abs(np.fft.rfft(active)))]
    assert frequency == pytest.approx(1_000, abs=2)
    assert result.stats.reason == "processed"
    assert result.stats.profile_frames >= 4
    assert 0.2 - 1e-12 <= result.stats.mean_spectral_gain <= 1
    np.testing.assert_array_equal(samples, saved)
    assert result.samples.shape == samples.shape
    assert result.samples.dtype == np.float32


def test_disabled_is_exact_copy_without_any_fft(noisy_tone, monkeypatch):
    from audio_sentinel import noise_reduction
    monkeypatch.setattr(noise_reduction.ShortTimeFFT, "from_window", lambda *a, **k: pytest.fail("Disabled must bypass FFT"))
    samples = noisy_tone[1]
    result = reduce_noise(samples, RATE, NoiseReductionSettings(max_working_bytes=1), max_decoded_bytes=2_000_000)
    np.testing.assert_array_equal(samples, result.samples)
    assert not np.shares_memory(samples, result.samples)
    assert result.stats.reason == "disabled"


@pytest.mark.parametrize("length", [1, 2, 255, 511, 512, 513, 800])
def test_short_audio_is_preserved_without_padding_or_warnings(length):
    samples = np.full((length, 2), 0.2, dtype=np.float32)
    result = reduce(samples)
    np.testing.assert_array_equal(samples, result.samples)
    assert result.stats.reason == "too_short"


def test_silence_is_exact_and_finite():
    samples = np.zeros((2_000, 2), dtype=np.float32)
    result = reduce(samples)
    np.testing.assert_array_equal(samples, result.samples)
    assert result.stats.reason == "silence"
    assert result.stats.mean_spectral_gain == 1


def test_each_channel_matches_independent_processing(noisy_tone):
    samples = noisy_tone[1]
    stereo = np.concatenate([samples, samples * 0.25, np.zeros_like(samples)], axis=1)
    joint = reduce(stereo).samples
    for channel in range(3):
        np.testing.assert_array_equal(joint[:, channel], reduce(stereo[:, channel:channel+1]).samples[:, 0])


def test_repeated_runs_are_deterministic(noisy_tone):
    first, second = reduce(noisy_tone[1]), reduce(noisy_tone[1])
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.stats == second.stats


def test_stronger_setting_attenuates_more_background(noisy_tone):
    weak = reduce(noisy_tone[1], reduction_strength=0.2)
    strong = reduce(noisy_tone[1], reduction_strength=0.9)
    assert np.linalg.norm(strong.samples[1000:7000]) < np.linalg.norm(weak.samples[1000:7000])
    assert strong.stats.mean_spectral_gain < weak.stats.mean_spectral_gain


@pytest.mark.parametrize("index", [0, 16_000, 31_999])
def test_impulse_timing_and_boundaries_are_preserved(index):
    samples = np.zeros((32_000, 1), dtype=np.float32)
    samples[index] = 0.5
    result = reduce(samples)
    assert np.argmax(abs(result.samples[:, 0])) == index
    assert result.samples.shape == samples.shape
    np.testing.assert_allclose(result.samples, samples, atol=1e-7)


@pytest.mark.parametrize("settings", [
    {"fft_size": 63}, {"fft_size": 500}, {"fft_size": 16384},
    {"noise_quantile": 0}, {"noise_quantile": 0.6}, {"noise_quantile": float("nan")},
    {"threshold_multiplier": 0}, {"max_working_bytes": 0}, {"reduction_strength": 0},
])
def test_invalid_configuration_is_rejected(settings):
    with pytest.raises(ValidationError):
        NoiseReductionSettings(**settings)


@pytest.mark.parametrize("samples", [
    np.empty((0, 1), dtype=np.float32), np.zeros(20, dtype=np.float32),
    np.zeros((20, 1), dtype=np.int16), np.array([[np.nan]], dtype=np.float32),
    np.array([[np.inf]], dtype=np.float32),
])
def test_invalid_samples_are_rejected(samples):
    with pytest.raises(AudioTransformError):
        reduce(samples)


def test_working_memory_limit_fails_before_spectral_allocation(noisy_tone, monkeypatch):
    from audio_sentinel import noise_reduction
    monkeypatch.setattr(noise_reduction.ShortTimeFFT, "stft", lambda *a, **k: pytest.fail("Must reject before STFT"))
    with pytest.raises(AudioTransformError) as caught:
        reduce(noisy_tone[1], max_working_bytes=1)
    assert caught.value.code == "noise_working_memory_exceeded"


def test_output_memory_limit_is_respected_even_when_disabled(noisy_tone):
    with pytest.raises(AudioTransformError) as caught:
        reduce_noise(noisy_tone[1], RATE, NoiseReductionSettings(), max_decoded_bytes=1)
    assert caught.value.code == "decoded_audio_too_large"


@pytest.mark.parametrize("rate", [True, 4000, 16_000.5])
def test_invalid_rate_is_rejected(noisy_tone, rate):
    with pytest.raises(AudioTransformError) as caught:
        reduce_noise(noisy_tone[1], rate, NoiseReductionSettings(), max_decoded_bytes=2_000_000)
    assert caught.value.code == "invalid_sample_rate"


def test_nonfinite_reconstruction_is_rejected(noisy_tone, monkeypatch):
    from audio_sentinel import noise_reduction
    monkeypatch.setattr(noise_reduction.ShortTimeFFT, "istft", lambda self, spectrum, **kw: np.full(kw['k1'], np.inf))
    with pytest.raises(AudioTransformError) as caught:
        reduce(noisy_tone[1])
    assert caught.value.code == "non_finite_audio"


@pytest.fixture
def loaded_clip(temporary_settings, active_consent):
    temporary_settings.ensure_directories()
    path = temporary_settings.paths.raw_data / "noise-test.wav"
    samples = np.random.default_rng(30).normal(0, 0.02, (48_000, 2)).astype(np.float32)
    sf.write(path, samples, 48_000, subtype="PCM_16")
    return load_audio(InputAudio("clip-001", Path("noise-test.wav"), active_consent), temporary_settings, now=NOW)


def test_disabled_preparation_matches_a1_2_exactly(loaded_clip):
    settings = AudioSettings()
    converted = resample_audio(convert_to_mono(loaded_clip.samples), 48_000, 16_000, max_decoded_bytes=settings.max_decoded_bytes)
    expected = normalize_loudness(converted, settings)
    result = prepare_signal(loaded_clip, settings, now=NOW)
    np.testing.assert_array_equal(result.samples, expected.samples)
    assert result.normalization == expected.stats
    assert result.noise_reduction.reason == "disabled"


def test_noise_reduction_runs_after_resampling_before_normalization(loaded_clip, monkeypatch):
    from audio_sentinel import audio_transforms
    def silence(samples, rate, settings, **kwargs):
        assert samples.shape == (16_000, 1)
        assert rate == 16_000
        assert settings.enabled
        return NoiseReducedAudio(np.zeros_like(samples), NoiseReductionStats("processed"))
    monkeypatch.setattr(audio_transforms, "reduce_noise", silence)
    result = prepare_signal(loaded_clip, AudioSettings(noise_reduction={"enabled": True}), now=NOW)
    assert result.normalization.reason == "silence"
    assert not result.samples.any()
    assert result.source == loaded_clip.source
    assert result.consent == loaded_clip.consent


@pytest.mark.parametrize("length,reason,frames", [(895, "too_short", 0), (896, "processed", 4)])
def test_minimum_four_complete_profile_frames(length, reason, frames):
    samples = np.random.default_rng(91).normal(0, 0.03, (length, 1)).astype(np.float32)
    result = reduce(samples)
    assert result.stats.reason == reason
    assert result.stats.profile_frames == frames
    assert result.samples.shape == samples.shape
    assert not np.shares_memory(result.samples, samples)
    if reason == "too_short":
        np.testing.assert_array_equal(result.samples, samples)


@pytest.mark.parametrize("fft_size,rate", [(64, 8_000), (512, 44_100), (2048, 48_000)])
def test_nondefault_fft_and_rate_preserve_boundary_impulses(fft_size, rate):
    samples = np.zeros((fft_size * 12 + 7, 1), dtype=np.float32)
    samples[0] = 0.25
    samples[-1] = -0.5
    result = reduce_noise(samples, rate, NoiseReductionSettings(enabled=True, fft_size=fft_size),
                          max_decoded_bytes=samples.nbytes)
    assert result.stats.reason == "processed"
    assert result.samples.dtype == np.float32 and result.samples.shape == samples.shape
    np.testing.assert_allclose(result.samples, samples, atol=1e-7)


def test_amplitude_scaling_does_not_change_the_gate_decisions(noisy_tone):
    samples = noisy_tone[1]
    original = reduce(samples)
    quieter = reduce(samples * np.float32(0.125))
    np.testing.assert_allclose(quieter.samples, original.samples * 0.125, rtol=1e-5, atol=1e-8)
    assert quieter.stats.mean_spectral_gain == pytest.approx(original.stats.mean_spectral_gain, abs=1e-10)


def test_readonly_strided_input_is_supported(noisy_tone):
    backing = np.repeat(noisy_tone[1], 2, axis=0)
    samples = backing[::2]
    samples.flags.writeable = False
    assert not samples.flags.c_contiguous
    result = reduce(samples)
    np.testing.assert_array_equal(result.samples, reduce(noisy_tone[1]).samples)
    np.testing.assert_array_equal(samples, noisy_tone[1])
    assert not np.shares_memory(result.samples, backing)


def test_silent_channel_contributes_unity_gain_to_diagnostics(noisy_tone):
    mono = reduce(noisy_tone[1])
    stereo = reduce(np.column_stack((noisy_tone[1][:, 0], np.zeros(len(noisy_tone[1]), dtype=np.float32))))
    assert stereo.stats.mean_spectral_gain == pytest.approx((mono.stats.mean_spectral_gain + 1) / 2)
    assert not stereo.samples[:, 1].any()


def test_spectral_allocation_failure_is_reported_and_input_preserved(noisy_tone, monkeypatch):
    from audio_sentinel import noise_reduction
    samples = noisy_tone[1]
    before = samples.copy()
    def fail(*args, **kwargs):
        raise MemoryError("simulated spectral allocation failure")
    monkeypatch.setattr(noise_reduction.ShortTimeFFT, "stft", fail)
    with pytest.raises(AudioTransformError) as caught:
        reduce(samples)
    assert caught.value.code == "insufficient_memory"
    np.testing.assert_array_equal(samples, before)
