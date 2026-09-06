from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_sentinel import pipeline
from audio_sentinel.audio_arrays import AudioTransformError
from audio_sentinel.audio_loader import AudioLoadError, load_audio
from audio_sentinel.audio_transforms import prepare_signal
from audio_sentinel.config import AudioSettings, NoiseReductionSettings, PersistenceSettings
from audio_sentinel.contracts import ConsentStatus, EventAnnotation, ProcessingScope
from audio_sentinel.interfaces import AudioPreprocessor, InputAudio
from audio_sentinel.persistence import AudioPersistenceError
from audio_sentinel.pipeline import AudioPreparationService
from audio_sentinel.preparation import PreparedAudioManifest


NOW = datetime(2026, 9, 6, tzinfo=UTC)


@pytest.fixture
def source(temporary_settings, active_consent):
    temporary_settings.paths.raw_data.mkdir(parents=True)
    path = temporary_settings.paths.raw_data / "generated.wav"
    tone = np.sin(2 * np.pi * 440 * np.arange(76_800) / 48_000)
    sf.write(path, np.column_stack((tone * 0.2, tone * 0.1)), 48_000, subtype="PCM_16")
    return InputAudio("generated-001", path, active_consent)


def assert_no_output(settings):
    parent = settings.paths.interim_data / "prepared"
    assert not parent.exists() or not list(parent.iterdir())


@pytest.mark.parametrize("denoise", [False, True])
def test_service_matches_preparation_recipe_and_persists_complete_bundle(temporary_settings, source, denoise):
    audio = AudioSettings(noise_reduction=NoiseReductionSettings(enabled=denoise))
    settings = temporary_settings.model_copy(update={"audio": audio})
    original = source.audio_path.read_bytes()
    annotation = EventAnnotation(label="ambient", risk_level="none", start_seconds=0, end_seconds=1, confidence=1)
    result = AudioPreparationService(settings, "synthetic-tone").prepare(source, annotations=[annotation], now=NOW)
    assert PreparedAudioManifest.model_validate_json(result.manifest_path.read_bytes()) == result.manifest
    manifest = result.manifest
    assert manifest.source.sha256 == hashlib.sha256(original).hexdigest()
    assert manifest.source.sample_rate_hz == 48_000 and manifest.source.channels == 2
    assert manifest.settings == audio
    assert manifest.clip.source_dataset == "synthetic-tone"
    assert manifest.clip.consent == source.consent and manifest.clip.annotations == [annotation]
    assert len(manifest.windows) == 5 and manifest.num_frames == 25_600 and manifest.channels == 1
    assert result.audio.audio_path.is_file()
    actual, rate = sf.read(result.audio.audio_path, dtype="float32", always_2d=True)
    expected = prepare_signal(load_audio(source, settings, now=NOW), audio, now=NOW)
    np.testing.assert_allclose(actual, expected.samples, atol=1 / 65536)
    assert rate == 16_000 and actual.shape == (25_600, 1)
    for window in manifest.windows:
        assert (settings.paths.interim_data / window.audio_path).is_file()
    assert source.audio_path.read_bytes() == original


def test_protocol_adapter_uses_override_for_all_stages_and_preserves_defaults(temporary_settings, source):
    service = AudioPreparationService(temporary_settings, "synthetic-tone")
    assert isinstance(service, AudioPreprocessor)
    override = AudioSettings(target_sample_rate_hz=8_000, convert_to_mono=False,
                             normalize_loudness=False, window_seconds=(5.0,), tail_policy="drop")
    audio = service.preprocess(source, override)
    manifest = PreparedAudioManifest.model_validate_json((audio.audio_path.parent / "manifest.json").read_bytes())
    assert manifest.settings == override and manifest.windows == ()
    assert sf.info(audio.audio_path).channels == 2
    assert audio.sample_rate_hz == 8_000 and sf.info(audio.audio_path).frames == 12_800
    assert service.settings.audio == AudioSettings()
    default = service.prepare(source, now=NOW)
    assert default.directory != audio.audio_path.parent and default.manifest.settings == AudioSettings()
    # Input acceptance must also use the override, not just the transform settings.
    with pytest.raises(AudioLoadError) as error:
        service.preprocess(source, AudioSettings(accepted_formats=("FLAC",)))
    assert error.value.code == "unsupported_format"


def test_relative_raw_path_and_repeat_save(temporary_settings, source):
    relative = InputAudio(source.clip_id, Path("generated.wav"), source.consent)
    service = AudioPreparationService(temporary_settings, "synthetic-tone")
    first = service.prepare(relative, now=NOW)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in first.directory.rglob("*") if p.is_file()}
    second = service.prepare(source, now=NOW)
    assert not first.reused and second.reused and first.directory == second.directory
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in first.directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure", ["denied", "missing", "outside"])
def test_input_failure_stops_before_transforms_and_saving(temporary_settings, source, monkeypatch, failure):
    if failure == "denied":
        source = InputAudio(source.clip_id, source.audio_path, source.consent.model_copy(
            update={"status": ConsentStatus.DENIED, "processing_scope": ProcessingScope.NONE,
                    "device_authorized": False}))
    elif failure == "missing":
        source = InputAudio(source.clip_id, Path("missing.wav"), source.consent)
    else:
        source = InputAudio(source.clip_id, Path("../escaped.wav"), source.consent)
    def unexpected(*args, **kwargs):
        pytest.fail("A later stage must not run after rejected input")
    monkeypatch.setattr(pipeline, "prepare_signal", unexpected)
    monkeypatch.setattr(pipeline, "save_prepared_audio", unexpected)
    with pytest.raises(AudioLoadError) as error:
        AudioPreparationService(temporary_settings, "synthetic-tone").prepare(source, now=NOW)
    assert error.value.code == {"denied": "consent_denied", "missing": "file_not_found", "outside": "invalid_path"}[failure]
    assert_no_output(temporary_settings)


def test_transform_failure_stops_before_saving(temporary_settings, source, monkeypatch):
    audio = AudioSettings(noise_reduction=NoiseReductionSettings(enabled=True, max_working_bytes=1))
    settings = temporary_settings.model_copy(update={"audio": audio})
    def unexpected(*args, **kwargs):
        pytest.fail("Saving must not run after a failed transform")
    monkeypatch.setattr(pipeline, "save_prepared_audio", unexpected)
    with pytest.raises(AudioTransformError) as error:
        AudioPreparationService(settings, "synthetic-tone").prepare(source, now=NOW)
    assert error.value.code == "noise_working_memory_exceeded"
    assert_no_output(settings)


def test_persistence_policy_is_forwarded_and_error_preserved(temporary_settings, source):
    settings = temporary_settings.model_copy(update={"persistence": PersistenceSettings(max_windows=1)})
    with pytest.raises(AudioPersistenceError) as error:
        AudioPreparationService(settings, "synthetic-tone").prepare(source, now=NOW)
    assert error.value.code == "too_many_windows"
    assert_no_output(settings)


def test_live_clock_is_not_frozen_at_service_entry(temporary_settings, source, monkeypatch):
    import audio_sentinel.audio_loader as loader
    import audio_sentinel.audio_transforms as transforms
    clock = [NOW]
    class Clock:
        @staticmethod
        def now(tz):
            return clock[0]
    consent = source.consent.model_copy(update={"expires_at": NOW + timedelta(seconds=1)})
    source = InputAudio(source.clip_id, source.audio_path, consent)
    original_load = pipeline.load_audio
    def delayed_load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        clock[0] += timedelta(seconds=2)
        return result
    monkeypatch.setattr(loader, "datetime", Clock)
    monkeypatch.setattr(transforms, "datetime", Clock)
    monkeypatch.setattr(pipeline, "load_audio", delayed_load)
    with pytest.raises(AudioLoadError) as error:
        AudioPreparationService(temporary_settings, "synthetic-tone").prepare(source)
    assert error.value.code == "consent_expired"
    assert_no_output(temporary_settings)


@pytest.mark.parametrize("dataset", ["", "   ", "x" * 129, None])
def test_invalid_dataset_rejected_without_creating_directories(temporary_settings, dataset):
    with pytest.raises(ValueError, match="source_dataset"):
        AudioPreparationService(temporary_settings, dataset)
    assert not temporary_settings.paths.root.exists()


def test_service_construction_has_no_disk_side_effects(temporary_settings):
    AudioPreparationService(temporary_settings, "synthetic-tone")
    assert not temporary_settings.paths.root.exists()
