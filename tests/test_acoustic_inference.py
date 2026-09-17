"""A3.2 tests for raw YAMNet orchestration over prepared waveform windows."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_sentinel import acoustic_inference as inference
from audio_sentinel.acoustic_loader import (
    AcousticModelMetadata,
    LoadedAcousticModel,
    TensorContract,
)
from audio_sentinel.acoustic_model import YAMNET, load_class_map
from audio_sentinel.audio_loader import AudioLoadError
from audio_sentinel.config import AudioSettings
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.pipeline import AudioPreparationService


NOW = datetime(2026, 9, 7, tzinfo=UTC)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values


class FakeModel:
    def __init__(self, mutate=None):
        self.waveforms = []
        self.mutate = mutate
        self.output_change = None

    def __call__(self, waveform):
        self.waveforms.append(waveform)
        if self.mutate:
            self.mutate(len(self.waveforms))
        patches = inference.expected_yamnet_patches(len(waveform))
        frames = YAMNET.patch_frames + (patches - 1) * 48
        scores = np.tile(np.linspace(0, 1, YAMNET.num_classes, dtype=np.float32), (patches, 1))
        embeddings = np.zeros((patches, 1024), dtype=np.float32)
        spectrogram = np.zeros((frames, YAMNET.mel_bands), dtype=np.float32)
        outputs = [scores, embeddings, spectrogram]
        if self.output_change:
            self.output_change(outputs)
        return tuple(FakeTensor(value) for value in outputs)


def loaded_model(model=None):
    metadata = AcousticModelMetadata(
        model_id=YAMNET.model_id,
        model_version="1",
        model_handle=YAMNET.model_handle,
        model_path="yamnet/1",
        artifact_sha256=YAMNET.artifact_sha256,
        class_map_sha256=YAMNET.class_map_sha256,
        label_mapping_version="1.0",
        runtime_distribution=YAMNET.runtime_distribution,
        runtime_version=YAMNET.runtime_version,
        exported_with_tensorflow="2.3.0",
        exported_with_tensorflow_git="test",
        signature_name="serving_default",
        input=TensorContract("waveform", (None,), "float32"),
        outputs=(
            TensorContract("output_0", (None, 521), "float32"),
            TensorContract("output_1", (None, 1024), "float32"),
            TensorContract("output_2", (None, 64), "float32"),
        ),
        num_classes=521,
    )
    return LoadedAcousticModel(metadata, load_class_map(), model or FakeModel())


@pytest.fixture
def prepared(temporary_settings, active_consent):
    temporary_settings.paths.raw_data.mkdir(parents=True)
    path = temporary_settings.paths.raw_data / "source.wav"
    samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(25_600) / 16_000)
    sf.write(path, samples, 16_000, subtype="PCM_16")
    clip = InputAudio("inference-clip", path, active_consent)
    audio = AudioSettings(
        target_sample_rate_hz=16_000,
        normalize_loudness=False,
        window_seconds=(1.0,),
        window_overlap_ratio=0.5,
    )
    bundle = AudioPreparationService(temporary_settings, "synthetic").prepare(
        clip, audio_settings=audio, now=NOW
    )
    return temporary_settings.paths, bundle


def run(prepared, loaded=None, **kwargs):
    paths, bundle = prepared
    return inference.infer_prepared_audio(
        paths,
        bundle.manifest_path.relative_to(paths.interim_data),
        loaded or loaded_model(),
        now=NOW,
        **kwargs,
    )


def test_runs_each_manifest_window_in_order_and_preserves_raw_scores(prepared):
    loaded = loaded_model()
    result = run(prepared, loaded)
    windows = prepared[1].manifest.windows

    assert [item.window for item in result.windows] == list(windows)
    assert [len(waveform) for waveform in loaded.model.waveforms] == [16_000, 16_000, 9_600]
    assert [item.patch_count for item in result.windows] == [2, 2, 1]
    assert np.array_equal(result.windows[0].scores[0], np.linspace(0, 1, 521, dtype=np.float32))
    assert result.patch_count == 5
    assert not any(waveform.flags.writeable for waveform in loaded.model.waveforms)
    assert not any(item.scores.flags.writeable for item in result.windows)


def test_result_records_file_and_model_provenance_without_absolute_paths(prepared):
    paths, bundle = prepared
    result = run(prepared)
    summary = json.loads(json.dumps(result.to_summary()))

    assert result.preparation_manifest_sha256 == hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest()
    assert result.raw_audio_sha256 == bundle.manifest.source.sha256
    assert result.clip_id == "inference-clip"
    assert result.preparation_manifest_path == bundle.manifest_path.relative_to(paths.interim_data).as_posix()
    assert [item["score_shape"] for item in summary["windows"]] == [[2, 521], [2, 521], [1, 521]]
    assert str(paths.root) not in json.dumps(summary)


def test_internal_and_prepared_padding_are_not_scored_as_real_audio(prepared):
    result = run(prepared)
    assert result.windows[-1].window.padding_samples == 6_400
    assert result.windows[-1].input_num_samples == 9_600
    assert result.windows[-1].patch_spans_samples == ((16_000, 25_600),)
    assert result.windows[0].patch_spans_samples == ((0, 15_600), (7_680, 16_000))


@pytest.mark.parametrize("samples,patches", [(1, 1), (15_600, 1), (15_601, 2), (23_280, 2), (23_281, 3)])
def test_expected_patch_grid_boundaries(samples, patches):
    assert inference.expected_yamnet_patches(samples) == patches


@pytest.mark.parametrize("samples", [0, -1, 1.5, True])
def test_expected_patch_grid_rejects_invalid_lengths(samples):
    with pytest.raises(ValueError, match="positive integer"):
        inference.expected_yamnet_patches(samples)


def test_empty_drop_tail_manifest_returns_empty_result(temporary_settings, active_consent):
    temporary_settings.paths.raw_data.mkdir(parents=True)
    path = temporary_settings.paths.raw_data / "short.wav"
    sf.write(path, np.zeros(8_000), 16_000, subtype="PCM_16")
    clip = InputAudio("short-clip", path, active_consent)
    bundle = AudioPreparationService(temporary_settings, "synthetic").prepare(
        clip,
        audio_settings=AudioSettings(window_seconds=(1.0,), tail_policy="drop"),
        now=NOW,
    )
    model = FakeModel()
    result = inference.infer_prepared_audio(
        temporary_settings.paths,
        bundle.manifest_path.relative_to(temporary_settings.paths.interim_data),
        loaded_model(model),
        now=NOW,
    )
    assert result.windows == () and result.patch_count == 0 and model.waveforms == []


def test_output_budget_is_checked_before_calling_model(prepared):
    model = FakeModel()
    policy = inference.AcousticInferenceSettings(max_model_output_bytes=1)
    with pytest.raises(inference.AcousticInferenceError, match="exceed") as error:
        run(prepared, loaded_model(model), policy=policy)
    assert error.value.code == "output_too_large" and model.waveforms == []


def test_window_count_limit_is_checked_before_model_calls(prepared):
    model = FakeModel()
    policy = inference.AcousticInferenceSettings(max_windows=2)
    with pytest.raises(inference.AcousticInferenceError) as error:
        run(prepared, loaded_model(model), policy=policy)
    assert error.value.code == "too_many_windows" and model.waveforms == []


@pytest.mark.parametrize("field", [
    "model_id", "artifact_sha256", "class_map_sha256", "runtime_version", "num_classes"
])
def test_rejects_unverified_loaded_model_before_reading_sources(prepared, field):
    loaded = loaded_model()
    values = {"model_id": "other", "artifact_sha256": "0" * 64,
              "class_map_sha256": "0" * 64, "runtime_version": "other", "num_classes": 520}
    changed = replace(loaded, metadata=replace(loaded.metadata, **{field: values[field]}))
    with pytest.raises(inference.AcousticInferenceError) as error:
        run(prepared, changed)
    assert error.value.code == "model_mismatch"


@pytest.mark.parametrize("change", [
    lambda outputs: outputs.__setitem__(0, outputs[0][:, :-1]),
    lambda outputs: outputs.__setitem__(1, outputs[1][:-1]),
    lambda outputs: outputs.__setitem__(2, outputs[2][:-1]),
    lambda outputs: outputs.__setitem__(0, outputs[0].astype(np.float64)),
    lambda outputs: outputs[0].__setitem__((0, 0), np.nan),
    lambda outputs: outputs[0].__setitem__((0, 0), 1.1),
])
def test_rejects_invalid_model_outputs(prepared, change):
    model = FakeModel()
    model.output_change = change
    with pytest.raises(inference.AcousticInferenceError) as error:
        run(prepared, loaded_model(model))
    assert error.value.code == "invalid_output"


def test_wraps_model_failure_with_stable_code(prepared):
    class Broken:
        def __call__(self, waveform):
            raise RuntimeError("private runtime detail")

    with pytest.raises(inference.AcousticInferenceError, match="YAMNet failed") as error:
        run(prepared, loaded_model(Broken()))
    assert error.value.code == "model_failed" and "private" not in str(error.value)


def test_detects_manifest_change_during_inference(prepared):
    _, bundle = prepared

    def mutate(call):
        if call == 1:
            bundle.manifest_path.write_bytes(bundle.manifest_path.read_bytes() + b"\n")

    with pytest.raises(inference.AcousticInferenceError) as error:
        run(prepared, loaded_model(FakeModel(mutate)))
    assert error.value.code in {"invalid_manifest", "source_changed"}


def test_detects_window_change_during_inference(prepared):
    paths, bundle = prepared
    first = paths.interim_data / bundle.manifest.windows[0].audio_path
    original = first.read_bytes()

    def mutate(call):
        if call == 1:
            first.write_bytes(original + b"x")

    with pytest.raises(inference.AcousticInferenceError) as error:
        run(prepared, loaded_model(FakeModel(mutate)))
    assert error.value.code == "source_changed"


def test_nonzero_preparation_padding_is_rejected(prepared):
    paths, bundle = prepared
    last = paths.interim_data / bundle.manifest.windows[-1].audio_path
    values, rate = sf.read(last, dtype="float32")
    values[-1] = 0.25
    sf.write(last, values, rate, subtype="PCM_16")
    with pytest.raises(inference.AcousticInferenceError, match="padding") as error:
        run(prepared)
    assert error.value.code == "source_mismatch"


def test_expired_consent_stops_before_model_calls(prepared):
    _, bundle = prepared
    data = json.loads(bundle.manifest_path.read_text())
    data["clip"]["consent"]["expires_at"] = (NOW + timedelta(days=1)).isoformat()
    bundle.manifest_path.write_text(json.dumps(data))
    model = FakeModel()
    with pytest.raises(AudioLoadError) as error:
        inference.infer_prepared_audio(
            prepared[0],
            prepared[1].manifest_path.relative_to(prepared[0].interim_data),
            loaded_model(model),
            now=NOW + timedelta(days=2),
        )
    assert getattr(error.value, "code", None) == "consent_expired" and model.waveforms == []


def test_source_audio_is_never_modified(prepared):
    path = prepared[0].raw_data / "source.wav"
    before = path.read_bytes()
    run(prepared)
    assert path.read_bytes() == before


def test_module_import_does_not_require_tensorflow():
    assert "tensorflow" not in inference.__dict__
