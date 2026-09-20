from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from audio_sentinel import vad
from audio_sentinel.config import Paths


class FakeNode:
    def __init__(self, name: str, shape: tuple[object, ...], dtype: str = "tensor(float)"):
        self.name = name
        self.shape = list(shape)
        self.type = dtype


class FakeLoadSession:
    def __init__(self):
        self.run_calls = 0
        self.providers = ["CPUExecutionProvider"]
        self.inputs = [
            FakeNode("input", ("seq_len", 576)),
            FakeNode("h", (1, 1, 128)),
            FakeNode("c", (1, 1, 128)),
        ]
        self.outputs = [
            FakeNode("speech_probs", ("Reshapespeech_probs_dim_0",)),
            FakeNode("hn", (1, 1, 128)),
            FakeNode("cn", (1, 1, 128)),
        ]

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, *_args, **_kwargs):
        self.run_calls += 1
        raise AssertionError("loading must not run inference")


def model_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Paths, Path, str]:
    root = tmp_path / "project"
    directory = root / "models" / Path(*vad.MODEL_RELATIVE_PATH.parts)
    directory.mkdir(parents=True)
    payload = b"silero-vad-test-model"
    digest = hashlib.sha256(payload).hexdigest()
    spec = replace(
        vad.SILERO_VAD,
        artifact_sha256=digest,
        artifact_size_bytes=len(payload),
        runtime_version="test-runtime",
    )
    monkeypatch.setattr(vad, "SILERO_VAD", spec)
    (directory / vad.MODEL_FILENAME).write_bytes(payload)
    (directory / vad.INSTALL_MARKER).write_text(
        json.dumps(spec.marker(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return Paths.from_root(root), directory, digest


def install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    session: FakeLoadSession,
    *,
    version: str = "test-runtime",
):
    captured = {}

    class Options:
        pass

    def create_session(path, *, providers, sess_options):
        captured.update(path=path, providers=providers, options=sess_options)
        return session

    runtime = SimpleNamespace(
        __version__=version,
        SessionOptions=Options,
        InferenceSession=create_session,
    )
    monkeypatch.setattr(vad, "import_module", lambda name: runtime)
    return captured


def test_loads_pinned_cpu_model_and_returns_portable_metadata(tmp_path, monkeypatch):
    paths, directory, digest = model_tree(tmp_path, monkeypatch)
    session = FakeLoadSession()
    captured = install_fake_runtime(monkeypatch, session)

    loaded = vad.load_silero_vad(paths)

    assert loaded.session is session and session.run_calls == 0
    assert loaded.metadata.artifact_sha256 == digest
    assert loaded.metadata.model_path == "silero-vad/6"
    assert loaded.metadata.providers == ("CPUExecutionProvider",)
    assert loaded.metadata.inputs[0] == vad.VadTensorContract("input", (None, 576), "tensor(float)")
    assert loaded.metadata.outputs[0] == vad.VadTensorContract("speech_probs", (None,), "tensor(float)")
    assert captured["path"] == str(directory / vad.MODEL_FILENAME)
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["options"].inter_op_num_threads == 1
    assert captured["options"].intra_op_num_threads == 1
    assert captured["options"].enable_cpu_mem_arena is False
    assert captured["options"].log_severity_level == 4
    serialized = json.dumps(loaded.metadata.as_dict(), sort_keys=True)
    assert str(tmp_path) not in serialized


def test_loaded_metadata_converts_to_a4_1_speech_descriptor(tmp_path, monkeypatch):
    paths, _, digest = model_tree(tmp_path, monkeypatch)
    install_fake_runtime(monkeypatch, FakeLoadSession())

    descriptor = vad.load_silero_vad(paths).metadata.as_speech_descriptor()

    assert descriptor.model_id == "silero-vad"
    assert descriptor.model_version == "6.0"
    assert descriptor.artifact_sha256 == digest
    assert descriptor.runtime_distribution == "onnxruntime"


@pytest.mark.parametrize("relative", ["missing/6", "../outside", Path("C:/outside")])
def test_loader_rejects_missing_or_uncontained_paths(tmp_path, relative):
    paths = Paths.from_root(tmp_path / "project")
    paths.models.mkdir(parents=True)

    with pytest.raises(vad.VadError) as error:
        vad.load_silero_vad(paths, relative)

    assert error.value.code in {"invalid_path", "model_not_found"}


@pytest.mark.parametrize("change", ["missing_model", "missing_marker", "unexpected"])
def test_model_requires_exact_file_inventory(tmp_path, monkeypatch, change):
    _, directory, _ = model_tree(tmp_path, monkeypatch)
    if change == "missing_model":
        (directory / vad.MODEL_FILENAME).unlink()
    elif change == "missing_marker":
        (directory / vad.INSTALL_MARKER).unlink()
    else:
        (directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(vad.VadError) as error:
        vad.model_artifact_sha256(directory)

    assert error.value.code == "invalid_artifact"


@pytest.mark.parametrize("change", ["bad_json", "wrong_value", "too_large"])
def test_loader_rejects_invalid_install_marker(tmp_path, monkeypatch, change):
    _, directory, _ = model_tree(tmp_path, monkeypatch)
    marker = directory / vad.INSTALL_MARKER
    if change == "bad_json":
        marker.write_text("not json", encoding="utf-8")
    elif change == "wrong_value":
        value = vad.SILERO_VAD.marker()
        value["model_version"] = "other"
        marker.write_text(json.dumps(value), encoding="utf-8")
    else:
        monkeypatch.setattr(vad, "MAX_MARKER_BYTES", 1)

    with pytest.raises(vad.VadError) as error:
        vad.model_artifact_sha256(directory)

    assert error.value.code == "invalid_artifact"


def test_loader_checks_exact_model_size_before_hash(tmp_path, monkeypatch):
    _, directory, _ = model_tree(tmp_path, monkeypatch)
    (directory / vad.MODEL_FILENAME).write_bytes(b"wrong-size")

    with pytest.raises(vad.VadError, match="size differs") as error:
        vad.model_artifact_sha256(directory)

    assert error.value.code == "invalid_artifact"


def test_model_byte_limit_is_enforced_before_hashing(tmp_path, monkeypatch):
    _, directory, _ = model_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(vad, "MAX_MODEL_BYTES", 1)
    monkeypatch.setattr(
        vad.hashlib,
        "sha256",
        lambda: pytest.fail("oversized model must not be hashed"),
    )

    with pytest.raises(vad.VadError) as error:
        vad.model_artifact_sha256(directory)

    assert error.value.code == "invalid_artifact"


@pytest.mark.parametrize("linked_name,expected_code", [("silero-vad", "invalid_path"),
                                                         (vad.MODEL_FILENAME, "invalid_artifact"),
                                                         (vad.INSTALL_MARKER, "invalid_artifact")])
def test_loader_rejects_links_in_model_path_or_payload(
    tmp_path, monkeypatch, linked_name, expected_code
):
    paths, directory, _ = model_tree(tmp_path, monkeypatch)
    original = vad._is_link
    monkeypatch.setattr(vad, "_is_link", lambda path: path.name == linked_name or original(path))

    with pytest.raises(vad.VadError) as error:
        vad.load_silero_vad(paths)

    assert error.value.code == expected_code


def test_loader_detects_model_change_during_hash(tmp_path, monkeypatch):
    _, directory, _ = model_tree(tmp_path, monkeypatch)
    real_fstat = vad.os.fstat
    calls = 0

    def changing_fstat(descriptor):
        nonlocal calls
        calls += 1
        details = real_fstat(descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_size=details.st_size,
                st_mtime_ns=details.st_mtime_ns,
                st_ctime_ns=details.st_ctime_ns + 1,
            )
        return details

    monkeypatch.setattr(vad.os, "fstat", changing_fstat)
    with pytest.raises(vad.VadError) as error:
        vad.model_artifact_sha256(directory)
    assert error.value.code == "artifact_changed"


def test_loader_detects_model_change_while_runtime_loads(tmp_path, monkeypatch):
    paths, _, digest = model_tree(tmp_path, monkeypatch)
    install_fake_runtime(monkeypatch, FakeLoadSession())
    hashes = iter((digest, "0" * 64))
    monkeypatch.setattr(vad, "model_artifact_sha256", lambda _directory: next(hashes))

    with pytest.raises(vad.VadError) as error:
        vad.load_silero_vad(paths)

    assert error.value.code == "artifact_changed"


def test_changed_model_digest_is_rejected_before_runtime_import(tmp_path, monkeypatch):
    paths, directory, _ = model_tree(tmp_path, monkeypatch)
    original = vad.SILERO_VAD
    monkeypatch.setattr(vad, "SILERO_VAD", replace(original, artifact_sha256="0" * 64))
    marker = directory / vad.INSTALL_MARKER
    marker.write_text(json.dumps(vad.SILERO_VAD.marker()), encoding="utf-8")
    monkeypatch.setattr(vad, "import_module", lambda name: pytest.fail("runtime imported too early"))

    with pytest.raises(vad.VadError) as error:
        vad.load_silero_vad(paths)

    assert error.value.code == "artifact_mismatch"


def test_loader_reports_missing_and_wrong_runtime(tmp_path, monkeypatch):
    paths, _, _ = model_tree(tmp_path, monkeypatch)

    def missing(_name):
        raise ImportError("not installed")

    monkeypatch.setattr(vad, "import_module", missing)
    with pytest.raises(vad.VadError, match="setup_speech") as missing_error:
        vad.load_silero_vad(paths)
    assert missing_error.value.code == "runtime_missing"

    monkeypatch.setattr(vad, "import_module", lambda name: SimpleNamespace(__version__="other"))
    with pytest.raises(vad.VadError, match="test-runtime") as version_error:
        vad.load_silero_vad(paths)
    assert version_error.value.code == "runtime_mismatch"


def test_loader_wraps_runtime_creation_failure_without_leaking_details(tmp_path, monkeypatch):
    paths, _, _ = model_tree(tmp_path, monkeypatch)

    class Options:
        pass

    runtime = SimpleNamespace(
        __version__="test-runtime",
        SessionOptions=Options,
        InferenceSession=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private runtime path")
        ),
    )
    monkeypatch.setattr(vad, "import_module", lambda _name: runtime)

    with pytest.raises(vad.VadError) as error:
        vad.load_silero_vad(paths)

    assert error.value.code == "model_load_failed"
    assert "private" not in str(error.value)


def test_loader_reports_uninspectable_session_as_signature_mismatch(tmp_path, monkeypatch):
    paths, _, _ = model_tree(tmp_path, monkeypatch)
    session = FakeLoadSession()
    session.get_inputs = None
    install_fake_runtime(monkeypatch, session)

    with pytest.raises(vad.VadError) as error:
        vad.load_silero_vad(paths)

    assert error.value.code == "signature_mismatch"


@pytest.mark.parametrize(
    "change",
    ["provider", "input_name", "input_shape", "input_dtype", "output_name",
     "output_shape", "output_dtype", "extra_input", "no_run"],
)
def test_loader_rejects_provider_and_signature_drift(tmp_path, monkeypatch, change):
    paths, _, _ = model_tree(tmp_path, monkeypatch)
    session = FakeLoadSession()
    if change == "provider":
        session.providers = ["CUDAExecutionProvider"]
    elif change == "input_name":
        session.inputs[0].name = "waveform"
    elif change == "input_shape":
        session.inputs[0].shape = ["seq_len", 512]
    elif change == "input_dtype":
        session.inputs[0].type = "tensor(double)"
    elif change == "output_name":
        session.outputs[0].name = "output"
    elif change == "output_shape":
        session.outputs[0].shape = ["seq", 1]
    elif change == "output_dtype":
        session.outputs[0].type = "tensor(double)"
    elif change == "extra_input":
        session.inputs.append(FakeNode("extra", (1,)))
    else:
        session.run = None
    install_fake_runtime(monkeypatch, session)

    with pytest.raises(vad.VadError, match="contract|entry point") as error:
        vad.load_silero_vad(paths)

    assert error.value.code == "signature_mismatch"


def expected_metadata() -> vad.VadModelMetadata:
    return vad.VadModelMetadata(
        model_id=vad.SILERO_VAD.model_id,
        model_version=vad.SILERO_VAD.model_version,
        model_source=vad.SILERO_VAD.model_source,
        model_path=vad.MODEL_RELATIVE_PATH.as_posix(),
        artifact_sha256=vad.SILERO_VAD.artifact_sha256,
        source_distribution=vad.SILERO_VAD.source_distribution,
        source_distribution_version=vad.SILERO_VAD.source_distribution_version,
        runtime_distribution=vad.SILERO_VAD.runtime_distribution,
        runtime_version=vad.SILERO_VAD.runtime_version,
        providers=("CPUExecutionProvider",),
        inputs=(
            vad.VadTensorContract("input", (None, 576), "tensor(float)"),
            vad.VadTensorContract("h", (1, 1, 128), "tensor(float)"),
            vad.VadTensorContract("c", (1, 1, 128), "tensor(float)"),
        ),
        outputs=(
            vad.VadTensorContract("speech_probs", (None,), "tensor(float)"),
            vad.VadTensorContract("hn", (1, 1, 128), "tensor(float)"),
            vad.VadTensorContract("cn", (1, 1, 128), "tensor(float)"),
        ),
        sample_rate_hz=16_000,
        frame_samples=512,
        context_samples=64,
    )


class FakeScoringSession:
    def __init__(self):
        self.calls = []
        self.mutate = None
        self.fail = None

    def run(self, output_names, feeds):
        if self.fail:
            raise self.fail
        captured = {name: value.copy() for name, value in feeds.items()}
        self.calls.append((output_names, captured))
        size = len(feeds["input"])
        base = int(feeds["h"][0, 0, 0]) * 2
        outputs = [
            np.arange(base, base + size, dtype=np.float32) / 10,
            feeds["h"] + np.float32(1),
            feeds["c"] + np.float32(2),
        ]
        if self.mutate:
            self.mutate(outputs)
        return outputs


def loaded(session=None, metadata=None) -> vad.LoadedVadModel:
    return vad.LoadedVadModel(metadata or expected_metadata(), session or FakeScoringSession())


@pytest.mark.parametrize("samples,frames", [(1, 1), (511, 1), (512, 1), (513, 2),
                                             (1024, 2), (1025, 3)])
def test_expected_vad_frame_grid(samples, frames):
    assert vad.expected_vad_frames(samples) == frames


@pytest.mark.parametrize("samples", [0, -1, 1.5, True, "512"])
def test_expected_vad_frame_grid_rejects_bad_lengths(samples):
    with pytest.raises(ValueError, match="positive integer"):
        vad.expected_vad_frames(samples)


def test_scores_frames_with_preceding_context_and_exact_tail_metadata():
    session = FakeScoringSession()
    waveform = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)

    result = vad.infer_vad_probabilities(loaded(session), waveform)

    assert result.input_num_samples == 1000
    assert result.padded_num_samples == 1024
    assert result.frame_count == 2
    assert [frame.speech_probability for frame in result.frames] == pytest.approx([0.0, 0.1])
    assert result.frames[0] == vad.VadFrameProbability(0, 0, 512, 0.0, 0.032, 0.0, 0)
    tail = result.frames[1]
    assert (tail.frame_index, tail.start_sample, tail.end_sample) == (1, 512, 1000)
    assert (tail.start_seconds, tail.end_seconds, tail.padding_samples) == (0.032, 0.0625, 24)
    assert tail.speech_probability == pytest.approx(0.1)
    model_input = session.calls[0][1]["input"]
    assert model_input.shape == (2, 576) and model_input.dtype == np.float32
    assert np.array_equal(model_input[0, :64], np.zeros(64, dtype=np.float32))
    assert np.array_equal(model_input[0, 64:], waveform[:512])
    assert np.array_equal(model_input[1, :64], waveform[448:512])
    assert np.array_equal(model_input[1, 64:552], waveform[512:])
    assert np.array_equal(model_input[1, 552:], np.zeros(24, dtype=np.float32))


def test_probability_endpoints_and_waveform_amplitude_endpoints_are_valid():
    session = FakeScoringSession()
    session.mutate = lambda outputs: outputs[0].__setitem__(slice(None), (0.0, 1.0))
    waveform = np.concatenate(
        (np.full(512, -1.0, dtype=np.float32), np.full(512, 1.0, dtype=np.float32))
    )

    result = vad.infer_vad_probabilities(loaded(session), waveform)

    assert [frame.speech_probability for frame in result.frames] == [0.0, 1.0]


def test_batches_model_calls_and_carries_recurrent_state():
    session = FakeScoringSession()
    waveform = np.zeros(5 * 512, dtype=np.float32)

    result = vad.infer_vad_probabilities(
        loaded(session), waveform, settings=vad.VadInferenceSettings(max_frames_per_call=2)
    )

    assert [len(call[1]["input"]) for call in session.calls] == [2, 2, 1]
    assert np.count_nonzero(session.calls[0][1]["h"]) == 0
    assert np.all(session.calls[1][1]["h"] == 1)
    assert np.all(session.calls[2][1]["h"] == 2)
    assert np.all(session.calls[1][1]["c"] == 2)
    assert np.all(session.calls[2][1]["c"] == 4)
    assert [frame.speech_probability for frame in result.frames] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4]
    )


def test_independent_inference_calls_reset_state_and_are_deterministic():
    session = FakeScoringSession()
    model = loaded(session)
    waveform = np.zeros(1024, dtype=np.float32)

    first = vad.infer_vad_probabilities(model, waveform)
    second = vad.infer_vad_probabilities(model, waveform.copy())

    assert first == second
    assert np.count_nonzero(session.calls[0][1]["h"]) == 0
    assert np.count_nonzero(session.calls[1][1]["h"]) == 0


def test_inference_owns_model_input_and_preserves_source_waveform():
    class MutatingSession(FakeScoringSession):
        def run(self, output_names, feeds):
            result = super().run(output_names, feeds)
            feeds["input"][:] = 1.0
            return result

    session = MutatingSession()
    waveform = np.linspace(-0.5, 0.5, 513, dtype=np.float32)
    before = waveform.copy()

    vad.infer_vad_probabilities(loaded(session), waveform)

    assert np.array_equal(waveform, before)


def test_summary_is_json_ready_and_contains_no_waveform():
    result = vad.infer_vad_probabilities(loaded(), np.zeros(513, dtype=np.float32))
    summary = json.loads(json.dumps(result.to_summary()))

    assert summary["frame_count"] == 2
    assert summary["frames"][-1]["padding_samples"] == 511
    assert "waveform" not in json.dumps(summary)


@pytest.mark.parametrize(
    "waveform,code",
    [
        (np.zeros((2, 2), dtype=np.float32), "invalid_samples"),
        (np.zeros(2, dtype=np.float64), "invalid_samples"),
        (np.array([], dtype=np.float32), "invalid_samples"),
        (np.array([np.nan], dtype=np.float32), "invalid_samples"),
        (np.array([np.inf], dtype=np.float32), "invalid_samples"),
        (np.array([1.01], dtype=np.float32), "invalid_samples"),
        (np.array([-1.01], dtype=np.float32), "invalid_samples"),
    ],
)
def test_rejects_invalid_waveforms(waveform, code):
    with pytest.raises(vad.VadError) as error:
        vad.infer_vad_probabilities(loaded(), waveform)
    assert error.value.code == code


def test_input_and_output_budgets_fail_before_model_call():
    session = FakeScoringSession()
    waveform = np.zeros(513, dtype=np.float32)

    with pytest.raises(vad.VadError) as input_error:
        vad.infer_vad_probabilities(
            loaded(session), waveform, settings=vad.VadInferenceSettings(max_input_samples=512)
        )
    assert input_error.value.code == "input_too_large" and session.calls == []

    with pytest.raises(vad.VadError) as output_error:
        vad.infer_vad_probabilities(
            loaded(session), waveform, settings=vad.VadInferenceSettings(max_output_bytes=7)
        )
    assert output_error.value.code == "output_too_large" and session.calls == []


def test_input_output_and_batch_limits_are_inclusive_at_exact_boundaries():
    session = FakeScoringSession()
    result = vad.infer_vad_probabilities(
        loaded(session),
        np.zeros(1024, dtype=np.float32),
        settings=vad.VadInferenceSettings(
            max_input_samples=1024,
            max_frames_per_call=2,
            max_output_bytes=8,
        ),
    )

    assert result.frame_count == 2
    assert [len(call[1]["input"]) for call in session.calls] == [2]


@pytest.mark.parametrize("overrides", [
    {"max_input_samples": 0}, {"max_input_samples": True},
    {"max_frames_per_call": 0}, {"max_frames_per_call": 100_001},
    {"max_output_bytes": 0}, {"unexpected": 1},
])
def test_invalid_inference_settings_are_rejected(overrides):
    with pytest.raises(ValidationError):
        vad.VadInferenceSettings(**overrides)


@pytest.mark.parametrize("field,value", [
    ("model_id", "other"),
    ("model_version", "5.0"),
    ("model_source", "other"),
    ("model_path", "other/6"),
    ("artifact_sha256", "0" * 64),
    ("source_distribution", "other"),
    ("source_distribution_version", "other"),
    ("runtime_distribution", "other"),
    ("runtime_version", "other"),
    ("providers", ("CUDAExecutionProvider",)),
    ("inputs", (vad.VadTensorContract("input", (None, 512), "tensor(float)"),)),
    ("outputs", (vad.VadTensorContract("speech_probs", (None, 1), "tensor(float)"),)),
    ("sample_rate_hz", 8_000),
    ("frame_samples", 256),
    ("context_samples", 32),
])
def test_inference_rejects_unverified_loaded_model(field, value):
    changed = replace(expected_metadata(), **{field: value})
    with pytest.raises(vad.VadError) as error:
        vad.infer_vad_probabilities(loaded(metadata=changed), np.zeros(512, dtype=np.float32))
    assert error.value.code == "model_mismatch"


def test_inference_rejects_loaded_model_without_callable_session():
    with pytest.raises(vad.VadError) as error:
        vad.infer_vad_probabilities(
            vad.LoadedVadModel(expected_metadata(), SimpleNamespace(run=None)),
            np.zeros(512, dtype=np.float32),
        )

    assert error.value.code == "model_mismatch"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda outputs: outputs.__setitem__(0, outputs[0][:-1]),
        lambda outputs: outputs.__setitem__(0, outputs[0].astype(np.float64)),
        lambda outputs: outputs[0].__setitem__(0, np.nan),
        lambda outputs: outputs[0].__setitem__(0, -0.1),
        lambda outputs: outputs[0].__setitem__(0, 1.1),
        lambda outputs: outputs.__setitem__(1, np.zeros((1, 1, 127), dtype=np.float32)),
        lambda outputs: outputs.__setitem__(2, outputs[2].astype(np.float64)),
    ],
)
def test_rejects_invalid_model_outputs(mutate):
    session = FakeScoringSession()
    session.mutate = mutate
    with pytest.raises(vad.VadError) as error:
        vad.infer_vad_probabilities(loaded(session), np.zeros(512, dtype=np.float32))
    assert error.value.code == "invalid_output"


def test_rejects_wrong_output_inventory_and_wraps_runtime_failure():
    class WrongInventory(FakeScoringSession):
        def run(self, output_names, feeds):
            return [np.zeros(len(feeds["input"]), dtype=np.float32)]

    with pytest.raises(vad.VadError) as inventory_error:
        vad.infer_vad_probabilities(loaded(WrongInventory()), np.zeros(512, dtype=np.float32))
    assert inventory_error.value.code == "invalid_output"

    session = FakeScoringSession()
    session.fail = RuntimeError("private runtime detail")
    with pytest.raises(vad.VadError, match="failed while scoring") as runtime_error:
        vad.infer_vad_probabilities(loaded(session), np.zeros(512, dtype=np.float32))
    assert runtime_error.value.code == "model_failed"
    assert "private" not in str(runtime_error.value)


def test_wraps_preparation_memory_failure_with_stable_error(tmp_path, monkeypatch):
    waveform = np.zeros(512, dtype=np.float32)
    monkeypatch.setattr(vad.np, "zeros", lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError()))

    with pytest.raises(vad.VadError) as error:
        vad.infer_vad_probabilities(loaded(), waveform)

    assert error.value.code == "insufficient_memory"


def test_public_results_and_metadata_are_immutable():
    result = vad.infer_vad_probabilities(loaded(), np.zeros(512, dtype=np.float32))

    with pytest.raises(FrozenInstanceError):
        result.input_num_samples = 1
    with pytest.raises(FrozenInstanceError):
        result.frames[0].speech_probability = 1.0
    with pytest.raises(FrozenInstanceError):
        result.model.model_id = "changed"


def test_normal_import_does_not_require_onnxruntime(tmp_path):
    code = "import sys, audio_sentinel.vad; assert 'onnxruntime' not in sys.modules"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
