import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.config import AudioSettings, load_settings
from audio_sentinel.preparation import PreparedAudioManifest, preparation_schema_documents


@pytest.fixture
def manifest_data(active_consent):
    return {
        "source": {
            "audio_path": "example/clip-001.wav", "sha256": "a" * 64,
            "format": "WAV", "sample_rate_hz": 48_000, "channels": 2,
            "num_frames": 36_000, "size_bytes": 144_044,
        },
        "settings": AudioSettings(window_seconds=(1.0,)).model_dump(mode="json"),
        "clip": {
            "clip_id": "clip-001", "source_dataset": "example",
            "audio_path": "prepared/clip-001/audio.wav", "sample_rate_hz": 16_000,
            "duration_seconds": 0.75, "consent": active_consent.model_dump(mode="json"),
        },
        "channels": 1, "num_frames": 12_000,
        "windows": [{
            "window_id": "clip-001-1s-0000", "audio_path": "prepared/clip-001/windows/1s-0000.wav",
            "window_seconds": 1, "start_sample": 0, "end_sample": 12_000, "padding_samples": 4_000,
        }],
    }


def test_manifest_roundtrip_preserves_existing_clip_contract(manifest_data):
    manifest = PreparedAudioManifest.model_validate(manifest_data)
    assert PreparedAudioManifest.model_validate_json(manifest.model_dump_json()) == manifest
    assert manifest.clip.schema_version == "1.0"
    assert manifest.settings.window_sample_counts(1.0) == (16_000, 8_000)


@pytest.mark.parametrize("override", [
    {"window_seconds": []}, {"window_seconds": [float("inf")]},
    {"window_seconds": [float("nan")]}, {"window_seconds": [0.000001]},
    {"window_overlap_ratio": 1}, {"window_overlap_ratio": -0.1},
    {"window_overlap_ratio": 0.99999999}, {"target_rms_dbfs": float("nan")},
    {"target_rms_dbfs": -0.5}, {"silence_floor_dbfs": -10},
    {"max_input_bytes": 0}, {"max_decoded_bytes": 0}, {"max_duration_seconds": 0}, {"max_input_channels": 0},
    {"accepted_formats": []}, {"accepted_formats": ["WAV", "WAV"]},
    {"accepted_formats": ["MP3"]}, {"noise_reduction": {"reduction_strength": 2}},
    {"output_subtype": "FLOAT"}, {"typo_setting": True},
])
def test_invalid_settings_fail_before_audio_is_loaded(override):
    with pytest.raises(ValidationError):
        AudioSettings(**override)


@pytest.mark.parametrize("path", [
    "../outside.wav", "/outside.wav", "C:/outside.wav", "C:outside.wav",
    "folder\\clip.wav", "folder//clip.wav", "./clip.wav", "folder/", "clip\x00.wav",
])
def test_source_path_is_portable_and_relative(manifest_data, path):
    manifest_data["source"]["audio_path"] = path
    with pytest.raises(ValidationError, match="relative POSIX"):
        PreparedAudioManifest.model_validate(manifest_data)


@pytest.mark.parametrize("section,field,value", [
    ("clip", "sample_rate_hz", 48_000), ("clip", "duration_seconds", 0.8),
    ("clip", "duration_seconds", float("inf")), ("source", "channels", 9),
    ("source", "size_bytes", 300_000_000), ("source", "sha256", "invalid"),
    ("source", "num_frames", 48_000 * 601), ("clip", "audio_path", "prepared/file.flac"),
    ("settings", "max_decoded_bytes", 1),
])
def test_inconsistent_manifest_metadata_is_rejected(manifest_data, section, field, value):
    manifest_data[section][field] = value
    with pytest.raises(ValidationError):
        PreparedAudioManifest.model_validate(manifest_data)


@pytest.mark.parametrize("field,value", [("channels", 2), ("num_frames", 12_001)])
def test_output_must_match_conversion_settings(manifest_data, field, value):
    manifest_data[field] = value
    with pytest.raises(ValidationError):
        PreparedAudioManifest.model_validate(manifest_data)


def test_stereo_can_be_preserved_explicitly(manifest_data):
    manifest_data["settings"]["convert_to_mono"] = False
    manifest_data["channels"] = 2
    assert PreparedAudioManifest.model_validate(manifest_data).channels == 2


@pytest.mark.parametrize("field,value", [
    ("start_sample", 1), ("end_sample", 12_001), ("padding_samples", 0),
    ("window_seconds", 5), ("audio_path", "prepared/clip-001/audio.wav"),
])
def test_incorrect_window_grid_is_rejected(manifest_data, field, value):
    manifest_data["windows"][0][field] = value
    with pytest.raises(ValidationError):
        PreparedAudioManifest.model_validate(manifest_data)


def test_drop_policy_allows_no_windows_for_short_clip(manifest_data):
    manifest_data["settings"]["tail_policy"] = "drop"
    manifest_data["windows"] = []
    assert PreparedAudioManifest.model_validate(manifest_data).windows == ()


def test_missing_or_duplicate_window_is_rejected(manifest_data):
    original = manifest_data["windows"][0]
    for windows in ([], [original, original]):
        manifest_data["windows"] = windows
        with pytest.raises(ValidationError):
            PreparedAudioManifest.model_validate(manifest_data)


@pytest.mark.parametrize("duration,spans", [
    (1.0, [(0, 16_000, 0)]),
    (1.5, [(0, 16_000, 0), (8_000, 24_000, 0)]),
    (1.6, [(0, 16_000, 0), (8_000, 24_000, 0), (16_000, 25_600, 6_400)]),
])
def test_overlapping_windows_stop_at_first_window_reaching_end(manifest_data, duration, spans):
    manifest_data["source"]["num_frames"] = round(duration * 48_000)
    manifest_data["num_frames"] = round(duration * 16_000)
    manifest_data["clip"]["duration_seconds"] = duration
    manifest_data["windows"] = [
        {"window_id": f"window-{i:04d}", "audio_path": f"windows/{i:04d}.wav",
         "window_seconds": 1, "start_sample": start, "end_sample": end, "padding_samples": padding}
        for i, (start, end, padding) in enumerate(spans)
    ]
    assert len(PreparedAudioManifest.model_validate(manifest_data).windows) == len(spans)


def test_manifest_preserves_consent_and_annotation_checks(manifest_data):
    manifest_data["clip"]["consent"]["processing_scope"] = "acoustic_only"
    manifest_data["clip"]["annotations"] = [{
        "label": "distress_speech", "risk_level": "medium", "start_seconds": 0,
        "end_seconds": 0.5, "confidence": 0.8,
    }]
    with pytest.raises(ValidationError, match="speech event labels"):
        PreparedAudioManifest.model_validate(manifest_data)


def test_json_config_is_explicit_and_resolved_from_project_root(tmp_path):
    (tmp_path / "audio.json").write_text('{"window_seconds": [2], "normalize_loudness": false}')
    settings = load_settings(tmp_path, audio_config_path=Path("audio.json"))
    assert settings.audio.window_seconds == (2.0,)
    assert not settings.audio.normalize_loudness
    assert load_settings(tmp_path).audio == AudioSettings()
    (tmp_path / "audio.json").write_text('{"window_overalp_ratio": 0.5}')
    with pytest.raises(ValidationError):
        load_settings(tmp_path, audio_config_path=Path("audio.json"))


def test_documented_examples_and_schemas_are_current(project_root):
    settings_path = project_root / "configs" / "preprocessing.example.json"
    assert AudioSettings.model_validate_json(settings_path.read_text()) == AudioSettings()
    manifest_path = project_root / "docs" / "examples" / "prepared-audio-manifest.json"
    PreparedAudioManifest.model_validate_json(manifest_path.read_text())
    for filename, expected in preparation_schema_documents().items():
        assert json.loads((project_root / "docs" / "schemas" / "v1" / filename).read_text()) == expected
