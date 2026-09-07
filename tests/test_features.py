import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.config import AudioSettings, load_settings
from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.features import LogMelFeatureMetadata, feature_schema_documents, write_feature_schemas
from audio_sentinel.preparation import PreparedAudioManifest


@pytest.fixture
def feature_data(project_root):
    return json.loads((project_root / "docs/examples/log-mel-feature.json").read_text())


@pytest.mark.parametrize("samples,frames", [(1, 1), (511, 1), (512, 1), (671, 1),
                                          (672, 2), (16000, 97), (80000, 497), (160000, 997)])
def test_frame_grid_has_explicit_short_padding_and_drop_tail(samples, frames):
    assert LogMelSettings().expected_shape(samples) == (64, frames)


@pytest.mark.parametrize("samples", [0, -1, 1.5, True, "16000"])
def test_invalid_sample_count(samples):
    with pytest.raises(ValueError):
        LogMelSettings().frame_count(samples)


@pytest.mark.parametrize("overrides", [
    {"n_fft": 500}, {"n_fft": True}, {"n_fft": "512"}, {"win_length": 513},
    {"hop_length": 401}, {"hop_length": 0}, {"n_mels": 0},
    {"n_fft": 32, "win_length": 32, "hop_length": 16, "n_mels": 18},
    {"fmin_hz": -1}, {"fmin_hz": 8000}, {"fmax_hz": 8001},
    {"sample_rate_hz": 8000}, {"fmax_hz": float("nan")}, {"amin": float("inf")},
    {"amin": 0}, {"amin": 2}, {"reference_power": 0}, {"top_db": 0},
    {"center": True}, {"window": "hamming"}, {"power": 1},
    {"mel_scale": "htk"}, {"unexpected": 1}, {"max_feature_bytes": 0},
])
def test_invalid_recipes_rejected(overrides):
    with pytest.raises(ValidationError):
        LogMelSettings(**overrides)


def test_feature_memory_limit_before_allocation():
    assert LogMelSettings(max_feature_bytes=24832).expected_shape(16000) == (64, 97)
    with pytest.raises(ValueError, match="max_feature_bytes"):
        LogMelSettings(max_feature_bytes=24831).expected_shape(16000)


def test_json_roundtrip_and_link_to_preparation(feature_data, project_root):
    feature = LogMelFeatureMetadata.model_validate(feature_data)
    assert LogMelFeatureMetadata.model_validate_json(feature.model_dump_json()) == feature
    manifest = PreparedAudioManifest.model_validate_json(
        (project_root / "docs/examples/prepared-audio-manifest.json").read_text())
    feature.source.validate_against(manifest)
    assert feature.source.num_samples == 16000
    assert feature.source.window.padding_samples == 4000
    assert feature.analysis_padding_samples == 0


@pytest.mark.parametrize("field,value", [
    ("shape", [97, 64]), ("shape", [64, 98]), ("shape", [64, 0]),
    ("shape", [64, 97, 1]), ("shape", [64, "97"]),
    ("dtype", "float64"), ("axes", ["time", "mel"]),
    ("feature_sha256", "not-a-hash"), ("feature_path", "../file.npy"),
    ("feature_path", "C:/file.npy"), ("feature_path", "folder\\file.npy"),
    ("feature_path", "folder/file.png"), ("created_at", "2026-09-07T00:00:00"),
    ("extractor_version", " "), ("analysis_padding_samples", 1),
    ("schema_version", "2.0"),
])
def test_bad_feature_metadata(feature_data, field, value):
    feature_data[field] = value
    with pytest.raises(ValidationError):
        LogMelFeatureMetadata.model_validate(feature_data)


@pytest.mark.parametrize("field,value", [
    ("channels", 2), ("preparation_manifest_path", "../manifest.json"),
    ("preparation_manifest_path", "manifest.wav"), ("raw_audio_sha256", "g" * 64),
    ("window_audio_sha256", "a" * 63), ("preparation_manifest_sha256", "A" * 64),
])
def test_bad_source_metadata(feature_data, field, value):
    feature_data["source"][field] = value
    with pytest.raises(ValidationError):
        LogMelFeatureMetadata.model_validate(feature_data)


def test_rate_mismatch_rejected_without_changing_audio(feature_data):
    feature_data["settings"].update(sample_rate_hz=8000, fmax_hz=4000)
    with pytest.raises(ValidationError, match="sample rate"):
        LogMelFeatureMetadata.model_validate(feature_data)


def test_window_span_and_padding_must_match_duration(feature_data):
    feature_data["source"]["window"]["padding_samples"] -= 1
    with pytest.raises(ValidationError, match="span and padding"):
        LogMelFeatureMetadata.model_validate(feature_data)


def test_analysis_padding_is_separate_from_preparation_padding(feature_data):
    feature_data["source"]["window"].update(window_seconds=0.01, start_sample=0,
                                               end_sample=100, padding_samples=60)
    feature_data.update(shape=[64, 1], analysis_padding_samples=352)
    result = LogMelFeatureMetadata.model_validate(feature_data)
    assert result.source.num_samples == 160
    feature_data["analysis_padding_samples"] = 412
    with pytest.raises(ValidationError, match="analysis padding"):
        LogMelFeatureMetadata.model_validate(feature_data)


@pytest.mark.parametrize("change", ["clip", "raw_hash", "window", "channels"])
def test_semantic_linkage_rejects_another_source(feature_data, project_root, change):
    manifest = PreparedAudioManifest.model_validate_json(
        (project_root / "docs/examples/prepared-audio-manifest.json").read_text())
    if change == "clip":
        feature_data["source"]["clip_id"] = "another-clip"
    elif change == "raw_hash":
        feature_data["source"]["raw_audio_sha256"] = "b" * 64
    elif change == "window":
        feature_data["source"]["window"]["window_id"] = "another-window"
    else:
        data = manifest.model_dump()
        data["settings"]["convert_to_mono"] = False
        data["channels"] = 2
        manifest = PreparedAudioManifest.model_validate(data)
    with pytest.raises(ValueError, match="manifest"):
        LogMelFeatureMetadata.model_validate(feature_data).source.validate_against(manifest)


def test_settings_load_independently_without_breaking_phase_one(tmp_path):
    (tmp_path / "features.json").write_text('{"n_mels": 32}')
    (tmp_path / "audio.json").write_text('{"target_sample_rate_hz": 8000}')
    settings = load_settings(tmp_path, log_mel_config_path=Path("features.json"),
                             audio_config_path=Path("audio.json"))
    assert settings.log_mel.n_mels == 32
    assert settings.audio.target_sample_rate_hz == 8000
    assert load_settings(tmp_path).audio == AudioSettings()
    assert not (tmp_path / "data").exists()
    # Preparation-only settings remain valid. Extraction will reject a rate mismatch.
    with pytest.raises(ValidationError):
        settings.log_mel.n_mels = 16


def test_examples_and_exported_schemas_match_models(project_root, tmp_path, feature_data):
    assert LogMelSettings.model_validate_json(
        (project_root / "configs/log-mel.example.json").read_text()) == LogMelSettings()
    original = copy.deepcopy(feature_data)
    LogMelFeatureMetadata.model_validate(feature_data)
    assert feature_data == original
    write_feature_schemas(tmp_path)
    for name, document in feature_schema_documents().items():
        assert json.loads((project_root / "docs/schemas/v1" / name).read_text()) == document
        assert json.loads((tmp_path / name).read_text()) == document
