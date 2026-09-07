"""B2.2 numerical properties and reproducibility beyond the generator examples."""

from io import BytesIO
import os
import subprocess
import sys

import numpy as np
import pytest

from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.log_mel import generate_log_mel


RECIPES = [
    dict(sample_rate_hz=8000, fmax_hz=4000, n_fft=256, win_length=255, hop_length=64, n_mels=20),
    dict(sample_rate_hz=16000, fmax_hz=8000, n_fft=512, win_length=400, hop_length=160, n_mels=64),
    dict(sample_rate_hz=44100, fmax_hz=18000, n_fft=2048, win_length=1103, hop_length=441, n_mels=80),
    dict(sample_rate_hz=48000, fmax_hz=20000, n_fft=2048, win_length=1200, hop_length=480, n_mels=96),
]


def signal(rate=16000, length=4800):
    t = np.arange(length) / rate
    # Quiet intervals force some values below the requested clipping threshold.
    values = 0.4 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 2300 * t)
    values[length // 3:2 * length // 3] = 0
    return values.astype(np.float32)[:, None]


@pytest.mark.parametrize("options", RECIPES, ids=["8k", "16k", "44k1", "48k"])
def test_frame_grid_across_rates_recipes_and_seeded_lengths(options):
    recipe = LogMelSettings(**options)
    rng = np.random.default_rng(20260907)
    fft, hop = recipe.n_fft, recipe.hop_length
    lengths = [1, fft - 1, fft, fft + hop - 1, fft + hop, fft + 2 * hop - 1, fft + 2 * hop]
    lengths += rng.integers(1, 4 * fft, size=20).tolist()
    for length in lengths:
        samples = rng.normal(0, 0.1, (length, 1)).astype(np.float32)
        result = generate_log_mel(samples, recipe.sample_rate_hz, recipe)
        # Enumerate complete frame starts independently of the production shape helper.
        starts = list(range(0, max(length, fft) - fft + 1, hop))
        assert result.values.shape == (recipe.n_mels, len(starts)), (options, length)
        assert result.values.dtype == np.float32 and result.values.flags.c_contiguous
        assert np.isfinite(result.values).all()
        assert result.analysis_padding_samples == max(0, fft - length)


@pytest.mark.parametrize("top_db", [0.1, 12.0, 40.0, 80.0])
def test_top_db_is_exact_peak_relative_clipping(top_db):
    samples = signal()
    broad = generate_log_mel(samples, 16000, LogMelSettings(top_db=200)).values
    actual = generate_log_mel(samples, 16000, LogMelSettings(top_db=top_db)).values
    threshold = float(broad.max()) - top_db
    assert (broad < threshold).any(), "Fixture must exercise clipping"
    expected = np.maximum(broad.astype(np.float64), threshold)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
    assert float(np.ptp(actual)) <= top_db + 2e-5
    assert float(actual.max()) == float(broad.max())


@pytest.mark.parametrize("reference", [1e-4, 0.25, 4.0, 100.0])
def test_fixed_reference_shifts_every_band_without_peak_normalization(reference):
    samples = signal()
    baseline = generate_log_mel(samples, 16000, LogMelSettings()).values
    changed = generate_log_mel(samples, 16000, LogMelSettings(reference_power=reference)).values
    np.testing.assert_allclose(changed, baseline.astype(np.float64) - 10 * np.log10(reference),
                               rtol=0, atol=2e-5)


@pytest.mark.parametrize("amin", [1e-300, 1e-40, 1e-10, 1e-2])
def test_configurable_silence_floor_is_finite_and_absolute(amin):
    recipe = LogMelSettings(amin=amin, reference_power=0.5, top_db=12)
    actual = generate_log_mel(np.zeros((1024, 1), dtype=np.float32), 16000, recipe).values
    expected = 10 * np.log10(amin) - 10 * np.log10(0.5)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)
    assert np.isfinite(actual).all() and np.ptp(actual) == 0


def test_increasing_power_floor_only_raises_bins_below_that_floor():
    samples = signal()
    base = generate_log_mel(samples, 16000, LogMelSettings(top_db=200)).values
    actual = generate_log_mel(samples, 16000, LogMelSettings(amin=1e-2, top_db=200)).values
    assert (base < -20).any() and (base > -20).any()
    np.testing.assert_allclose(actual, np.maximum(base, -20), rtol=0, atol=2e-5)


def test_feature_values_can_be_positive():
    samples = signal() * np.float32(2)
    result = generate_log_mel(samples, 16000, LogMelSettings()).values
    assert result.max() > 0, "Power dB must not be normalized to a zero peak"


@pytest.mark.parametrize("options", RECIPES, ids=["8k", "16k", "44k1", "48k"])
def test_polarity_and_input_layout_do_not_change_power_features(options):
    recipe = LogMelSettings(**options)
    samples = signal(recipe.sample_rate_hz, recipe.n_fft * 3)
    backing = np.zeros((len(samples) * 2, 1), dtype=np.float32)
    backing[::2] = samples
    strided = backing[::2]
    strided.flags.writeable = False
    expected = generate_log_mel(samples, recipe.sample_rate_hz, recipe).values
    np.testing.assert_array_equal(generate_log_mel(-samples, recipe.sample_rate_hz, recipe).values, expected)
    np.testing.assert_array_equal(generate_log_mel(strided, recipe.sample_rate_hz, recipe).values, expected)


def test_previous_output_mutation_and_interleaved_recipes_cannot_affect_next_call():
    samples = signal()
    original = samples.copy()
    recipe = LogMelSettings()
    first = generate_log_mel(samples, 16000, recipe)
    expected = first.values.copy()
    first.values[:] = 123
    alternate = generate_log_mel(samples * np.float32(0.5), 16000, LogMelSettings(n_mels=32))
    alternate.values[:] = -999
    restored = LogMelSettings.model_validate_json(recipe.model_dump_json())
    actual = generate_log_mel(samples, 16000, restored)
    np.testing.assert_array_equal(actual.values, expected)
    np.testing.assert_array_equal(samples, original)
    assert not np.shares_memory(actual.values, first.values)


def test_fresh_processes_and_different_call_orders_produce_identical_npy_bytes(project_root, tmp_path):
    samples = signal()
    recipe = LogMelSettings()
    input_path = tmp_path / "input.npy"
    config_path = tmp_path / "recipe.json"
    np.save(input_path, samples, allow_pickle=False)
    config_path.write_text(recipe.model_dump_json(), encoding="utf-8")
    expected = BytesIO()
    np.save(expected, generate_log_mel(samples, 16000, recipe).values, allow_pickle=False)
    # Use the same on-disk samples in two clean interpreters, with one doing an
    # unrelated extraction first. No shell, network, or Python hash-order assumptions.
    program = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.log_mel import generate_log_mel
samples = np.load(sys.argv[2], allow_pickle=False)
recipe = LogMelSettings.model_validate_json(Path(sys.argv[3]).read_text())
if sys.argv[5] == "interleaved":
    generate_log_mel(samples * np.float32(0.5), 16000, LogMelSettings(n_mels=32))
result = generate_log_mel(samples, 16000, recipe)
np.save(sys.argv[4], result.values, allow_pickle=False)
'''
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    before = input_path.read_bytes()
    for order in ("direct", "interleaved"):
        output = tmp_path / f"{order}.npy"
        subprocess.run([sys.executable, "-c", program, str(project_root / "src"), str(input_path),
                        str(config_path), str(output), order], cwd=tmp_path, env=environment,
                       capture_output=True, text=True, check=True, timeout=90)
        assert output.read_bytes() == expected.getvalue(), order
    assert input_path.read_bytes() == before
