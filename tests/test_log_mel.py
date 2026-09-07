import json
from pathlib import Path
import subprocess
import sys

import librosa
import numpy as np
import pytest

from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.log_mel import LogMelError, estimate_log_mel_working_bytes, generate_log_mel


def tone(n=16000, frequency=1000, amplitude=0.25):
    return (amplitude * np.sin(2 * np.pi * frequency * np.arange(n) / 16000)).astype(np.float32)[:, None]


def reference(samples, recipe):
    """Independent framing, periodic Hann, FFT, Slaney triangles, and dB formula."""
    y = np.pad(samples[:, 0].astype(float), (0, max(0, recipe.n_fft - len(samples))))
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(recipe.win_length) / recipe.win_length)
    left = (recipe.n_fft - recipe.win_length) // 2
    window = np.pad(hann, (left, recipe.n_fft - recipe.win_length - left))
    starts = range(0, len(y) - recipe.n_fft + 1, recipe.hop_length)
    power = np.stack([np.abs(np.fft.rfft(y[s:s+recipe.n_fft] * window))**2 for s in starts], axis=1)
    def to_mel(hz):
        return hz / (200/3) if hz < 1000 else 15 + np.log(hz / 1000) / (np.log(6.4) / 27)
    mel_edges = np.linspace(to_mel(recipe.fmin_hz), to_mel(recipe.fmax_hz), recipe.n_mels + 2)
    edges = np.array([m * (200/3) if m < 15 else 1000 * np.exp((m-15) * np.log(6.4)/27) for m in mel_edges])
    frequencies = np.arange(recipe.n_fft // 2 + 1) * recipe.sample_rate_hz / recipe.n_fft
    filters = np.stack([np.maximum(0, np.minimum((frequencies-a)/(b-a), (c-frequencies)/(c-b))) * 2/(c-a)
                        for a,b,c in zip(edges[:-2], edges[1:-1], edges[2:])])
    db = 10 * np.log10(np.maximum(filters @ power, recipe.amin)) - 10 * np.log10(recipe.reference_power)
    return np.maximum(db, db.max() - recipe.top_db)


@pytest.mark.parametrize("n,frames", [(1,1), (511,1), (512,1), (671,1), (672,2), (16000,97), (80000,497), (160000,997)])
def test_shapes_padding_silence_and_owned_output(n, frames):
    samples = np.zeros((n,1), dtype=np.float32)
    result = generate_log_mel(samples,16000,LogMelSettings())
    assert result.values.shape == (64,frames)
    np.testing.assert_allclose(result.values,-100,atol=1e-5)
    assert result.values.dtype == np.float32 and result.values.flags.c_contiguous
    assert result.values.flags.owndata and not np.shares_memory(result.values,samples)
    assert result.analysis_padding_samples == max(0,512-n)
    assert result.num_samples == n and result.extractor_version == librosa.__version__


@pytest.mark.parametrize("n,overrides", [(16000,{}), (100,{}), (2000,dict(n_fft=1024,win_length=701,hop_length=200,n_mels=32,fmin_hz=100,fmax_hz=6000,reference_power=0.2,top_db=45))])
def test_matches_independent_fft_filterbank_and_db_reference(n, overrides):
    recipe = LogMelSettings(**overrides)
    samples = np.random.default_rng(731).normal(0,0.1,(n,1)).astype(np.float32)
    np.testing.assert_allclose(generate_log_mel(samples,16000,recipe).values,reference(samples,recipe),atol=5e-5,rtol=1e-6)


def test_amplitude_and_frequency_have_expected_effects():
    recipe = LogMelSettings()
    quiet = generate_log_mel(tone(amplitude=0.1),16000,recipe).values
    loud = generate_log_mel(tone(amplitude=0.2),16000,recipe).values
    peak_band = int(np.argmax(quiet.mean(axis=1)))
    centers = librosa.mel_frequencies(n_mels=66, fmin=0, fmax=8000)[1:-1]
    assert abs(centers[peak_band] - 1000) < 100
    np.testing.assert_allclose(loud[peak_band] - quiet[peak_band], 20*np.log10(2),atol=1e-4)
    high_band = int(np.argmax(generate_log_mel(tone(frequency=3000),16000,recipe).values.mean(axis=1)))
    assert high_band > peak_band


def test_strided_readonly_source_unchanged_and_repeatable():
    backing = tone(32000)
    samples = backing[::2]
    samples.flags.writeable=False
    original=backing.copy()
    a=generate_log_mel(samples,16000,LogMelSettings())
    b=generate_log_mel(samples,16000,LogMelSettings())
    np.testing.assert_array_equal(a.values,b.values)
    np.testing.assert_array_equal(backing,original)


def test_complete_fft_tail_is_dropped():
    samples = tone(671)
    before = generate_log_mel(samples,16000,LogMelSettings()).values
    samples[512:]=10
    np.testing.assert_array_equal(before,generate_log_mel(samples,16000,LogMelSettings()).values)


@pytest.mark.parametrize("value", [np.finfo(np.float32).max, np.finfo(np.float32).tiny])
def test_finite_float32_extremes(value):
    result=generate_log_mel(np.full((512,1),value,dtype=np.float32),16000,LogMelSettings())
    assert np.isfinite(result.values).all()
    assert np.ptp(result.values) <= 80.001


@pytest.mark.parametrize("samples,code", [(np.zeros((1,1)),"invalid_samples"),(np.zeros((0,1),dtype=np.float32),"invalid_samples"),
    (np.zeros(10,dtype=np.float32),"invalid_samples"),(np.zeros((10,2),dtype=np.float32),"mono_required"),
    (np.array([[np.nan]],dtype=np.float32),"non_finite_audio"),(np.array([[np.inf]],dtype=np.float32),"non_finite_audio")])
def test_invalid_audio(samples,code):
    with pytest.raises(LogMelError) as error:
        generate_log_mel(samples,16000,LogMelSettings())
    assert error.value.code==code


@pytest.mark.parametrize("rate,code", [(8000,"sample_rate_mismatch"),(16000.0,"invalid_sample_rate"),(True,"invalid_sample_rate")])
def test_invalid_rate(rate,code):
    with pytest.raises(LogMelError) as error:
        generate_log_mel(tone(),rate,LogMelSettings())
    assert error.value.code==code


@pytest.mark.parametrize("limit", [0,-1,True,1.5])
def test_invalid_workspace_limit(limit):
    with pytest.raises(LogMelError,match="positive integer"):
        generate_log_mel(tone(),16000,LogMelSettings(),max_working_bytes=limit)


@pytest.mark.parametrize("kind", ["payload","workspace"])
def test_memory_preflight_precedes_backend(monkeypatch,kind):
    def unexpected(*args,**kwargs):
        pytest.fail("Backend was invoked despite rejected allocation")
    monkeypatch.setattr(librosa.filters,"mel",unexpected)
    settings=LogMelSettings(max_feature_bytes=1) if kind=="payload" else LogMelSettings()
    limit=estimate_log_mel_working_bytes(16000,settings)-1
    with pytest.raises(LogMelError) as error:
        generate_log_mel(tone(),16000,settings,max_working_bytes=limit)
    assert error.value.code==("feature_too_large" if kind=="payload" else "working_memory_exceeded")


def test_empty_filterbank_is_rejected():
    with pytest.raises(LogMelError) as error:
        generate_log_mel(tone(),16000,LogMelSettings(fmin_hz=1000,fmax_hz=1001))
    assert error.value.code=="invalid_mel_filters"


def test_allocation_failure_has_stable_error(monkeypatch):
    def fail(*args,**kwargs):
        raise MemoryError("simulated")
    monkeypatch.setattr(librosa,"stft",fail)
    with pytest.raises(LogMelError) as error:
        generate_log_mel(tone(),16000,LogMelSettings())
    assert error.value.code=="insufficient_memory"


def test_invalid_backend_output_is_rejected(monkeypatch):
    monkeypatch.setattr(librosa,"power_to_db",lambda *args,**kwargs: np.full((64,97),np.nan))
    with pytest.raises(LogMelError) as error:
        generate_log_mel(tone(),16000,LogMelSettings())
    assert error.value.code=="invalid_features"


def test_smoke_script_from_another_directory(project_root,tmp_path):
    result=subprocess.run([sys.executable,str(project_root/"scripts/smoke_test_log_mel.py")],cwd=tmp_path,
                          capture_output=True,text=True,check=True,timeout=90)
    report=json.loads(result.stdout)
    assert report["status"]=="passed"
    assert report["shapes"]==[[64,97],[64,97],[64,97],[64,497],[64,997]]
    assert report["source_unchanged"] and report["temporary_files_removed"]
