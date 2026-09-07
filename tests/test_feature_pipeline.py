from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from audio_sentinel import feature_pipeline as pipeline
from audio_sentinel.audio_loader import AudioLoadError
from audio_sentinel.config import AudioSettings, NoiseReductionSettings, load_settings
from audio_sentinel.contracts import EventAnnotation
from audio_sentinel.feature_persistence import FeaturePersistenceError, FeaturePersistenceSettings, load_log_mel
from audio_sentinel.feature_pipeline import AudioFeatureService, FeaturePipelineError
from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.interfaces import InputAudio


NOW = datetime(2026, 9, 7, tzinfo=UTC)


@pytest.fixture
def source(temporary_settings, active_consent):
    temporary_settings.paths.raw_data.mkdir(parents=True)
    path = temporary_settings.paths.raw_data / "source.wav"
    samples = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(76800) / 48000)
    sf.write(path, np.column_stack((samples, samples * 0.5)), 48000, subtype="PCM_16")
    return InputAudio("source-001", path, active_consent)


@pytest.mark.parametrize("format", ["WAV", "FLAC"])
@pytest.mark.parametrize("rate", [8000, 16000])
@pytest.mark.parametrize("denoise", [False, True])
def test_full_service_inventory_and_encoded_outputs(temporary_settings, source, format, rate, denoise):
    if format == "FLAC":
        samples, source_rate = sf.read(source.audio_path, dtype="int16", always_2d=True)
        path = source.audio_path.with_suffix(".flac")
        sf.write(path, samples, source_rate, subtype="PCM_16")
        source = replace(source, audio_path=path)
    before = source.audio_path.read_bytes()
    audio = AudioSettings(target_sample_rate_hz=rate, noise_reduction=NoiseReductionSettings(enabled=denoise))
    recipe = (LogMelSettings() if rate == 16000 else LogMelSettings(
        sample_rate_hz=8000, fmax_hz=4000, n_fft=256, win_length=200, hop_length=80, n_mels=32))
    annotation = EventAnnotation(label="ambient", risk_level="none", start_seconds=0,
                                 end_seconds=1, confidence=1)
    service = AudioFeatureService(temporary_settings, "synthetic-tone")
    result = service.prepare(source, audio_settings=audio, log_mel_settings=recipe,
                             annotations=[annotation], now=NOW)
    assert not result.reused and len(result.features) == 5
    manifest = result.prepared.manifest
    assert manifest.settings == audio and manifest.clip.annotations == [annotation]
    assert manifest.clip.source_dataset == "synthetic-tone"
    assert manifest.source.sha256 == hashlib.sha256(before).hexdigest()
    for feature, window, frames in zip(result.features, manifest.windows, (97, 97, 97, 497, 997), strict=True):
        assert feature.metadata.source.window == window and feature.metadata.settings == recipe
        assert feature.metadata.shape == (recipe.n_mels, frames)
        loaded = load_log_mel(temporary_settings.paths,
                             feature.metadata_path.relative_to(temporary_settings.paths.processed_data), now=NOW)
        assert np.isfinite(loaded.values).all() and loaded.values.dtype == np.float32
    assert source.audio_path.read_bytes() == before
    assert service.settings.audio == AudioSettings() and service.settings.log_mel == LogMelSettings()
    summary = json.loads(json.dumps(result.to_summary()))
    assert summary["feature_count"] == 5 and summary["clip_id"] == source.clip_id
    assert [item["window_id"] for item in summary["features"]] == [w.window_id for w in manifest.windows]


def test_fresh_service_repeat_run_reuses_all_files(temporary_settings, source):
    first = AudioFeatureService(temporary_settings, "synthetic-tone").prepare(source, now=NOW)
    paths = [first.prepared.manifest_path]
    paths += [path for item in first.features for path in (item.feature_path, item.metadata_path)]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    repeated = AudioFeatureService(temporary_settings, "synthetic-tone").prepare(source, now=NOW + timedelta(seconds=5))
    assert repeated.reused and all(feature.reused for feature in repeated.features)
    assert [f.directory for f in first.features] == [f.directory for f in repeated.features]
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def test_json_configured_defaults_control_the_complete_window_inventory(temporary_settings, source, tmp_path):
    audio = AudioSettings(window_seconds=(0.5,), window_overlap_ratio=0, tail_policy="drop", normalize_loudness=False)
    recipe = LogMelSettings(n_mels=32)
    audio_path, feature_path = tmp_path / "audio.json", tmp_path / "feature.json"
    audio_path.write_text(audio.model_dump_json())
    feature_path.write_text(recipe.model_dump_json())
    configured = load_settings(temporary_settings.paths.root, audio_config_path=audio_path,
                               log_mel_config_path=feature_path)
    result = AudioFeatureService(configured, "configured-tone").prepare(source, now=NOW)
    assert result.prepared.manifest.settings == audio
    assert [f.metadata.source.window.start_sample for f in result.features] == [0, 8000, 16000]
    assert [f.metadata.shape for f in result.features] == [(32, 47)] * 3
    assert all(f.metadata.settings == recipe for f in result.features)


def test_drop_tail_empty_inventory_is_explicit_and_repeatable(temporary_settings, source):
    audio = AudioSettings(window_seconds=(5.0,), tail_policy="drop")
    service = AudioFeatureService(temporary_settings, "synthetic-tone", max_total_feature_bytes=1)
    result = service.prepare(source, audio_settings=audio, now=NOW)
    assert result.features == () and result.prepared.manifest.windows == ()
    assert result.to_summary()["feature_count"] == 0
    assert not temporary_settings.paths.processed_data.exists()
    assert service.prepare(source, audio_settings=audio, now=NOW).reused


@pytest.mark.parametrize("options,code", [
    ({"audio_settings": AudioSettings(convert_to_mono=False)}, "mono_required"),
    ({"log_mel_settings": LogMelSettings(sample_rate_hz=8000, fmax_hz=4000)}, "sample_rate_mismatch"),
])
def test_incompatible_recipes_fail_before_preparation(temporary_settings, source, monkeypatch, options, code):
    def unexpected(*args, **kwargs): pytest.fail("Preparation ran with an incompatible recipe")
    monkeypatch.setattr(pipeline.AudioPreparationService, "prepare", unexpected)
    with pytest.raises(FeaturePipelineError) as error:
        AudioFeatureService(temporary_settings, "synthetic-tone").prepare(source, now=NOW, **options)
    assert error.value.code == code
    assert not temporary_settings.paths.interim_data.exists()


@pytest.mark.parametrize("failure", ["input", "permission"])
def test_preparation_failure_never_runs_features(temporary_settings, source, monkeypatch, failure):
    if failure == "input": source = replace(source, audio_path=Path("missing.wav"))
    else: source = replace(source, consent=source.consent.model_copy(update={"expires_at": NOW}))
    def unexpected(*args, **kwargs): pytest.fail("Features ran after preparation failure")
    monkeypatch.setattr(pipeline, "save_window_log_mel", unexpected)
    with pytest.raises(AudioLoadError):
        AudioFeatureService(temporary_settings, "synthetic-tone").prepare(source, now=NOW)
    assert not temporary_settings.paths.processed_data.exists()


@pytest.mark.parametrize("options,code", [
    ({"max_total_feature_bytes": 1}, "total_output_too_large"),
    ({"feature_policy": FeaturePersistenceSettings(max_working_bytes=1)}, "working_memory_exceeded"),
    ({"feature_policy": FeaturePersistenceSettings(max_decoded_bytes=1)}, "decoded_audio_too_large"),
    ({"feature_policy": FeaturePersistenceSettings(max_output_bytes=1)}, "output_too_large"),
    ({"feature_policy": FeaturePersistenceSettings(max_metadata_bytes=1)}, "metadata_too_large"),
])
def test_inventory_budgets_fail_before_first_feature_write(temporary_settings, source, monkeypatch, options, code):
    def unexpected(*args, **kwargs): pytest.fail("Feature persistence ran despite inventory limits")
    monkeypatch.setattr(pipeline, "save_window_log_mel", unexpected)
    with pytest.raises(FeaturePipelineError) as error:
        AudioFeatureService(temporary_settings, "synthetic-tone", **options).prepare(source, now=NOW)
    assert error.value.code == code
    assert not temporary_settings.paths.processed_data.exists()


def test_partial_failure_preserves_valid_bundles_and_retry_completes(temporary_settings, source, monkeypatch):
    service = AudioFeatureService(temporary_settings, "synthetic-tone")
    original = pipeline.save_window_log_mel
    completed = []
    failure = FeaturePersistenceError("write_failed", "injected second-window failure")
    def fail_second(*args, **kwargs):
        if completed: raise failure
        saved = original(*args, **kwargs)
        completed.append(saved)
        return saved
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "save_window_log_mel", fail_second)
        with pytest.raises(FeaturePersistenceError) as error: service.prepare(source, now=NOW)
        assert error.value is failure
    before = completed[0].metadata_path.read_bytes()
    retried = service.prepare(source, now=NOW)
    assert retried.prepared.reused and len(retried.features) == 5
    assert [f.reused for f in retried.features] == [True, False, False, False, False]
    assert completed[0].metadata_path.read_bytes() == before
    assert not list(temporary_settings.paths.processed_data.rglob(".pending-*"))


def test_final_inventory_verification_detects_earlier_bundle_corruption(temporary_settings, source, monkeypatch):
    original = pipeline.save_window_log_mel
    saved = []
    def corrupt_earlier(*args, **kwargs):
        result = original(*args, **kwargs)
        saved.append(result)
        if len(saved) == 5:
            data = bytearray(saved[0].feature_path.read_bytes())
            data[-1] ^= 1
            saved[0].feature_path.write_bytes(data)
        return result
    monkeypatch.setattr(pipeline, "save_window_log_mel", corrupt_earlier)
    with pytest.raises(FeaturePersistenceError, match="hash"):
        AudioFeatureService(temporary_settings, "synthetic-tone").prepare(source, now=NOW)


def test_manifest_changes_between_windows_do_not_form_a_mixed_inventory(temporary_settings, source, monkeypatch):
    original = pipeline.save_window_log_mel
    calls = []
    def alter_manifest(*args, **kwargs):
        result = original(*args, **kwargs)
        if not calls:
            path = temporary_settings.paths.interim_data / result.metadata.source.preparation_manifest_path
            with path.open("ab") as stream: stream.write(b"\n")
        calls.append(result)
        return result
    monkeypatch.setattr(pipeline, "save_window_log_mel", alter_manifest)
    with pytest.raises(FeaturePipelineError, match="source differs"):
        AudioFeatureService(temporary_settings, "synthetic-tone").prepare(source, now=NOW)


def test_live_clock_expiry_between_windows_stops_processing(temporary_settings, source, monkeypatch):
    base = datetime.now(UTC)
    source = replace(source, consent=source.consent.model_copy(update={"expires_at": base + timedelta(hours=1)}))
    current = [base]
    class Clock:
        @staticmethod
        def now(tz): return current[0]
    monkeypatch.setattr(pipeline, "datetime", Clock)
    original = pipeline.save_window_log_mel
    calls = []
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        current[0] = base + timedelta(hours=2)
        return result
    monkeypatch.setattr(pipeline, "save_window_log_mel", delayed)
    with pytest.raises(AudioLoadError) as error:
        AudioFeatureService(temporary_settings, "synthetic-tone").prepare(source)
    assert error.value.code == "consent_expired" and len(calls) == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_total_budget_rejected_without_io(temporary_settings, limit):
    with pytest.raises(ValueError, match="positive integer"):
        AudioFeatureService(temporary_settings, "synthetic-tone", max_total_feature_bytes=limit)
    assert not temporary_settings.paths.root.exists()


def test_construction_does_not_create_directories(temporary_settings):
    AudioFeatureService(temporary_settings, "synthetic-tone")
    assert not temporary_settings.paths.root.exists()


def test_integrated_smoke_script_runs_standalone(project_root, tmp_path):
    process = subprocess.run([sys.executable, str(project_root / "scripts/smoke_test_feature_pipeline.py")],
                             cwd=tmp_path, capture_output=True, text=True, check=True, timeout=90)
    report = json.loads(process.stdout)
    assert report["status"] == "passed" and report["feature_count"] == 5
    assert report["reused"] and report["source_unchanged"] and report["temporary_files_removed"]
