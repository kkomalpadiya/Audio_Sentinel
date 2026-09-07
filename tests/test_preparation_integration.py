"""A1.5: exercise real preparation stages together, through disk readback."""

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError
import soundfile as sf

from audio_sentinel.config import AudioSettings, load_settings
from audio_sentinel.contracts import EventAnnotation, ProcessingScope
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.persistence import AudioPersistenceError
from audio_sentinel.pipeline import AudioPreparationService
from audio_sentinel.preparation import PreparedAudioManifest


NOW = datetime(2026, 9, 6, tzinfo=UTC)


def recording(root, consent, *, format="WAV", rate=48_000, channels=2, silent=False):
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / ("sample." + format.lower())
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(rate * 5 // 4) / rate)
    samples = np.column_stack([tone / (channel + 1) for channel in range(channels)])
    if silent:
        samples[:] = 0
    sf.write(path, samples, rate, format=format, subtype="PCM_16")
    return InputAudio("integration-001", path, consent)


def configured(root, audio):
    config = root / "audio-settings.json"
    config.write_text(audio.model_dump_json(), encoding="utf-8")
    return load_settings(root, audio_config_path=Path(config.name))


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}


def read_bundle(result, settings):
    manifest = PreparedAudioManifest.model_validate_json(result.manifest_path.read_bytes())
    assert manifest == result.manifest
    full, rate = sf.read(result.audio.audio_path, dtype="int16", always_2d=True)
    assert (rate, len(full), full.shape[1]) == (manifest.clip.sample_rate_hz, manifest.num_frames, manifest.channels)
    expected_files = {"audio.wav", "manifest.json"}
    for window in manifest.windows:
        path = settings.paths.interim_data / window.audio_path
        assert path.resolve().is_relative_to(result.directory.resolve())
        expected_files.add(path.relative_to(result.directory).as_posix())
        actual, window_rate = sf.read(path, dtype="int16", always_2d=True)
        real = full[window.start_sample:window.end_sample]
        np.testing.assert_array_equal(actual[:len(real)], real)
        assert window_rate == rate and len(actual) == len(real) + window.padding_samples
        assert not actual[len(real):].any()
        assert sf.info(path).subtype == "PCM_16"
    assert set(snapshot(result.directory)) == expected_files
    assert not list(result.directory.parent.glob(".pending-*"))
    return manifest, full.astype(np.float64) / 32768


@pytest.mark.parametrize("format", ["WAV", "FLAC"])
@pytest.mark.parametrize("rate,channels,target,mono,tail,denoise", [
    (48_000, 2, 16_000, True, "pad", False),
    (44_100, 1, 8_000, True, "drop", False),
    (16_000, 2, 16_000, False, "pad", True),
    (8_000, 1, 16_000, True, "drop", True),
])
def test_format_and_recipe_matrix(tmp_path, active_consent, format, rate, channels, target, mono, tail, denoise):
    clip = recording(tmp_path, active_consent, format=format, rate=rate, channels=channels)
    original = clip.audio_path.read_bytes()
    audio = AudioSettings(target_sample_rate_hz=target, convert_to_mono=mono, tail_policy=tail,
                          noise_reduction={"enabled": denoise})
    settings = configured(tmp_path, audio)
    annotation = EventAnnotation(label="ambient", risk_level="none", start_seconds=0.1, end_seconds=1.2, confidence=1)
    service = AudioPreparationService(settings, "generated-integration")
    result = service.prepare(clip, annotations=[annotation], now=NOW)
    manifest, full = read_bundle(result, settings)
    assert full.shape == (target * 5 // 4, 1 if mono else channels)
    assert manifest.settings == audio and manifest.source.format == format
    assert manifest.source.sha256 == hashlib.sha256(original).hexdigest()
    assert manifest.clip.annotations == [annotation] and manifest.clip.consent == clip.consent
    assert manifest.clip.duration_seconds == 1.25 and manifest.clip.source_dataset == "generated-integration"
    assert len(manifest.windows) == (4 if tail == "pad" else 1)
    assert 20 * np.log10(np.sqrt(np.mean(full ** 2))) == pytest.approx(-20, abs=0.03)
    assert np.max(np.abs(full)) <= 10 ** (-1 / 20) + 1 / 32768
    frequency = np.fft.rfftfreq(len(full), 1 / target)[np.argmax(abs(np.fft.rfft(full[:, 0])))]
    assert frequency == pytest.approx(440, abs=2)
    assert clip.audio_path.read_bytes() == original
    before = snapshot(result.directory)
    restarted = AudioPreparationService(load_settings(tmp_path, audio_config_path=Path("audio-settings.json")), "generated-integration")
    reused = restarted.prepare(clip, annotations=[annotation], now=NOW)
    assert reused.reused and reused.directory == result.directory
    assert snapshot(result.directory) == before


def test_silence_survives_enabled_pipeline_and_padded_windows(tmp_path, active_consent):
    clip = recording(tmp_path, active_consent, silent=True)
    settings = configured(tmp_path, AudioSettings(noise_reduction={"enabled": True}))
    result = AudioPreparationService(settings, "generated-silence").prepare(clip, now=NOW)
    manifest, full = read_bundle(result, settings)
    assert not full.any() and len(manifest.windows) == 4


def test_late_write_failure_keeps_previous_bundle_and_allows_retry(tmp_path, active_consent, monkeypatch):
    clip = recording(tmp_path, active_consent)
    settings = configured(tmp_path, AudioSettings())
    service = AudioPreparationService(settings, "generated-recovery")
    baseline = service.prepare(replace(clip, clip_id="previous-001"), now=NOW)
    original_files = snapshot(baseline.directory.parent)
    raw = clip.audio_path.read_bytes()
    write = sf.SoundFile.write
    count = 0
    def failing_write(audio, data):
        nonlocal count
        count += 1
        write(audio, data)
        if count == 3:
            raise OSError("simulated disk failure after a window write")
    with monkeypatch.context() as patch:
        patch.setattr(sf.SoundFile, "write", failing_write)
        with pytest.raises(AudioPersistenceError) as caught:
            service.prepare(clip, now=NOW)
        assert caught.value.code == "write_failed"
    assert snapshot(baseline.directory.parent) == original_files
    assert list(baseline.directory.parent.iterdir()) == [baseline.directory]
    assert clip.audio_path.read_bytes() == raw
    recovered = service.prepare(clip, now=NOW)
    assert not recovered.reused and recovered.directory != baseline.directory
    read_bundle(recovered, settings)


def test_changed_annotations_conflict_and_original_request_still_reuses(tmp_path, active_consent):
    clip = recording(tmp_path, active_consent)
    settings = configured(tmp_path, AudioSettings())
    service = AudioPreparationService(settings, "generated-annotations")
    original = service.prepare(clip, now=NOW)
    before = snapshot(original.directory)
    annotation = EventAnnotation(label="ambient", risk_level="none", start_seconds=0, end_seconds=1, confidence=1)
    with pytest.raises(AudioPersistenceError) as caught:
        service.prepare(clip, annotations=[annotation], now=NOW)
    assert caught.value.code == "output_conflict"
    assert snapshot(original.directory) == before
    assert service.prepare(clip, now=NOW).reused


def test_acoustic_permission_does_not_admit_language_annotations(tmp_path, active_consent):
    consent = active_consent.model_copy(update={"processing_scope": ProcessingScope.ACOUSTIC_ONLY})
    clip = recording(tmp_path, consent)
    settings = configured(tmp_path, AudioSettings())
    service = AudioPreparationService(settings, "generated-permission")
    speech = EventAnnotation(label="threatening_speech", risk_level="high", start_seconds=0, end_seconds=1, confidence=1)
    with pytest.raises(ValidationError):
        service.prepare(clip, annotations=[speech], now=NOW)
    assert not settings.paths.interim_data.exists()
    read_bundle(service.prepare(clip, now=NOW), settings)


def test_smoke_script_runs_from_another_directory_without_pythonpath(project_root, tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, str(project_root / "scripts/smoke_test_preparation.py")],
                               cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60, check=True)
    report = json.loads(completed.stdout)
    assert report["status"] == "passed"
    assert report["sample_rate_hz"] == 16_000 and report["window_count"] == 5
    assert report["reused"] and report["source_unchanged"] and report["temporary_files_removed"]
