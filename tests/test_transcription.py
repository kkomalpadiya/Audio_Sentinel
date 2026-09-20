from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from audio_sentinel.config import Paths
from audio_sentinel.speech_contracts import TranscriptConfidenceKind
from audio_sentinel import transcription


class FakeEngine:
    device = "cpu"
    compute_type = "int8_float32"
    is_multilingual = False


class FakeLoadModel:
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.model = FakeEngine()

    def transcribe(self, *_args, **_kwargs):
        raise AssertionError("loading must not run inference")


class FakeInferenceModel:
    def __init__(self, segments=(), info=None, error=None):
        self.segments = segments
        self.info = info or SimpleNamespace(language="en", language_probability=1.0, duration=1.0)
        self.error = error
        self.calls = []

    def transcribe(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        if self.error:
            raise self.error
        return iter(self.segments), self.info


def segment(**changes):
    values = {
        "id": 0,
        "start": 0.0,
        "end": 0.5,
        "text": " hello ",
        "tokens": (1, 2),
        "avg_logprob": math.log(0.8),
        "no_speech_prob": 0.1,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def aggregate_digest(artifacts):
    inventory = {
        item.filename: {"sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in artifacts
    }
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def model_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "project"
    directory = root / "models" / Path(*transcription.MODEL_RELATIVE_PATH.parts)
    directory.mkdir(parents=True)
    payloads = {"config.json": b"{}", "model.bin": b"model", "tokenizer.json": b"tokens", "vocabulary.txt": b"vocab"}
    artifacts = tuple(
        transcription.TranscriptionArtifactSpec(name, len(data), hashlib.sha256(data).hexdigest())
        for name, data in payloads.items()
    )
    spec = replace(
        transcription.WHISPER_TINY_EN,
        artifacts=artifacts,
        artifact_sha256=aggregate_digest(artifacts),
        faster_whisper_version="test-fw",
        ctranslate2_version="test-ct2",
        tokenizers_version="test-tok",
    )
    monkeypatch.setattr(transcription, "WHISPER_TINY_EN", spec)
    for name, data in payloads.items():
        (directory / name).write_bytes(data)
    (directory / transcription.INSTALL_MARKER).write_text(
        json.dumps(spec.marker(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return Paths.from_root(root), directory, spec


def install_fake_loader(monkeypatch):
    created = []

    def make_model(path, **kwargs):
        model = FakeLoadModel(path, **kwargs)
        created.append(model)
        return model

    monkeypatch.setattr(
        transcription,
        "_distribution_version",
        lambda name: {"faster-whisper": "test-fw", "ctranslate2": "test-ct2", "tokenizers": "test-tok"}[name],
    )
    monkeypatch.setattr(
        transcription, "import_module", lambda _name: SimpleNamespace(WhisperModel=make_model)
    )
    return created


def loaded(model=None):
    spec = transcription.WHISPER_TINY_EN
    metadata = transcription.TranscriptionModelMetadata(
        model_id=spec.model_id,
        model_version=spec.model_version,
        model_source=spec.model_source,
        model_path=transcription.MODEL_RELATIVE_PATH.as_posix(),
        source_repository=spec.source_repository,
        source_revision=spec.source_revision,
        artifact_sha256=spec.artifact_sha256,
        artifacts=tuple(
            transcription.TranscriptionArtifactMetadata(item.filename, item.size_bytes, item.sha256)
            for item in spec.artifacts
        ),
        runtime_distribution="faster-whisper",
        runtime_version=spec.faster_whisper_version,
        ctranslate2_version=spec.ctranslate2_version,
        tokenizers_version=spec.tokenizers_version,
        device="cpu",
        compute_type="int8_float32",
        language="en",
        sample_rate_hz=16_000,
    )
    return transcription.LoadedTranscriptionModel(metadata, model or FakeInferenceModel())


def test_loads_verified_local_cpu_model_without_inference(tmp_path, monkeypatch):
    paths, directory, spec = model_tree(tmp_path, monkeypatch)
    created = install_fake_loader(monkeypatch)

    result = transcription.load_transcription_model(paths)

    assert result.model is created[0]
    assert created[0].path == str(directory)
    assert created[0].kwargs == {
        "device": "cpu", "compute_type": "int8", "cpu_threads": 1,
        "num_workers": 1, "local_files_only": True,
    }
    assert result.metadata.artifact_sha256 == spec.artifact_sha256
    assert str(tmp_path) not in json.dumps(result.metadata.as_dict())


def test_metadata_converts_to_speech_descriptor(tmp_path, monkeypatch):
    paths, _, spec = model_tree(tmp_path, monkeypatch)
    install_fake_loader(monkeypatch)

    descriptor = transcription.load_transcription_model(paths).metadata.as_speech_descriptor()

    assert descriptor.model_id == spec.model_id
    assert descriptor.artifact_sha256 == spec.artifact_sha256
    assert descriptor.runtime_distribution == "faster-whisper"


@pytest.mark.parametrize("relative", ["missing/1", "../outside", Path("C:/outside")])
def test_loader_rejects_missing_or_uncontained_paths(tmp_path, relative):
    paths = Paths.from_root(tmp_path / "project")
    paths.models.mkdir(parents=True)

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths, relative)

    assert error.value.code in {"invalid_path", "model_not_found"}


@pytest.mark.parametrize("damage", ["extra", "marker", "payload"])
def test_loader_rejects_inventory_marker_or_payload_damage(tmp_path, monkeypatch, damage):
    paths, directory, _ = model_tree(tmp_path, monkeypatch)
    install_fake_loader(monkeypatch)
    if damage == "extra":
        (directory / "unexpected.txt").write_text("x")
    elif damage == "marker":
        (directory / transcription.INSTALL_MARKER).write_text("{}")
    else:
        (directory / "model.bin").write_bytes(b"wrong")

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths)

    assert error.value.code in {"invalid_artifact", "artifact_mismatch"}


def test_loader_rejects_runtime_version_drift(tmp_path, monkeypatch):
    paths, _, _ = model_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(transcription, "_distribution_version", lambda _name: "wrong")

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths)

    assert error.value.code == "runtime_mismatch"


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("device", "cuda"), ("compute_type", "float32"), ("is_multilingual", True)],
)
def test_loader_rejects_engine_capability_drift(tmp_path, monkeypatch, attribute, value):
    paths, _, _ = model_tree(tmp_path, monkeypatch)
    install_fake_loader(monkeypatch)
    monkeypatch.setattr(FakeEngine, attribute, value)

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths)

    assert error.value.code == "signature_mismatch"


def test_transcribes_with_fixed_offline_decoding_and_owned_input():
    fake = FakeInferenceModel([segment()])
    source = np.zeros(16_000, dtype=np.float32)
    result = transcription.transcribe_segment(loaded(fake), source)

    assert result.candidate.text == "hello"
    assert result.candidate.confidence_kind is TranscriptConfidenceKind.DERIVED_SCORE
    assert result.candidate.confidence_score == pytest.approx(0.8)
    assert result.candidate.language == "en" and result.candidate.language_confidence == 1.0
    passed, kwargs = fake.calls[0]
    assert passed is not source and passed.flags.c_contiguous
    assert kwargs["language"] == "en" and kwargs["task"] == "transcribe"
    assert kwargs["beam_size"] == 1 and kwargs["temperature"] == 0.0
    assert kwargs["condition_on_previous_text"] is False and kwargs["vad_filter"] is False


def test_confidence_is_token_weighted_exponential_mean_log_probability():
    chunks = [
        segment(id=0, text=" first ", tokens=(1,), avg_logprob=math.log(0.5)),
        segment(id=1, start=0.5, end=1.0, text="second", tokens=(2, 3, 4), avg_logprob=math.log(0.9)),
    ]

    result = transcription.transcribe_segment(loaded(FakeInferenceModel(chunks)), np.zeros(16_000, dtype=np.float32))

    assert result.candidate.text == "first second"
    assert result.candidate.confidence_score == pytest.approx(math.exp((math.log(0.5) + 3 * math.log(0.9)) / 4))
    assert [chunk.token_count for chunk in result.chunks] == [1, 3]


def test_empty_or_whitespace_model_output_has_no_candidate():
    result = transcription.transcribe_segment(
        loaded(FakeInferenceModel([segment(text=" \t ", tokens=())])),
        np.zeros(16_000, dtype=np.float32),
    )

    assert result.candidate is None and result.chunks == ()


@pytest.mark.parametrize(
    "waveform",
    [
        np.array([], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(4, dtype=np.float64),
        np.array([np.nan], dtype=np.float32),
        np.array([1.01], dtype=np.float32),
    ],
)
def test_rejects_invalid_waveforms(waveform):
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(loaded(), waveform)

    assert error.value.code == "invalid_samples"


def test_rejects_input_limit_before_model_call():
    fake = FakeInferenceModel()
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(fake), np.zeros(3, dtype=np.float32),
            settings=transcription.TranscriptionSettings(max_input_samples=2),
        )
    assert error.value.code == "input_too_large" and not fake.calls


@pytest.mark.parametrize(
    "bad_segment",
    [
        segment(id=True),
        segment(start=-0.1),
        segment(end=1.1),
        segment(avg_logprob=0.1),
        segment(no_speech_prob=1.1),
        segment(tokens=(True,)),
        segment(text=3),
    ],
)
def test_rejects_invalid_model_segments(bad_segment):
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel([bad_segment])), np.zeros(16_000, dtype=np.float32)
        )
    assert error.value.code == "invalid_output"


@pytest.mark.parametrize(
    ("settings", "segments"),
    [
        (transcription.TranscriptionSettings(max_segments=1), [segment(), segment(id=1)]),
        (transcription.TranscriptionSettings(max_tokens=1), [segment(tokens=(1, 2))]),
        (transcription.TranscriptionSettings(max_output_bytes=1), [segment(text="long")]),
    ],
)
def test_rejects_bounded_model_output(settings, segments):
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel(segments)), np.zeros(16_000, dtype=np.float32), settings=settings
        )
    assert error.value.code == "output_too_large"


@pytest.mark.parametrize(
    "info",
    [
        SimpleNamespace(language="fr", language_probability=1.0, duration=1.0),
        SimpleNamespace(language="en", language_probability=0.9, duration=1.0),
        SimpleNamespace(language="en", language_probability=1.0, duration=2.0),
    ],
)
def test_rejects_invalid_english_only_metadata(info):
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel(info=info)), np.zeros(16_000, dtype=np.float32)
        )
    assert error.value.code == "invalid_output"


def test_wraps_model_failure_in_safe_error():
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel(error=RuntimeError("secret"))),
            np.zeros(16_000, dtype=np.float32),
        )
    assert error.value.code == "model_failed" and "secret" not in str(error.value)


def test_rejects_tampered_loaded_metadata_before_inference():
    item = loaded()
    tampered = replace(item, metadata=replace(item.metadata, device="cuda"))
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(tampered, np.zeros(16_000, dtype=np.float32))
    assert error.value.code == "model_mismatch"


def test_results_are_immutable_and_json_ready():
    result = transcription.transcribe_segment(
        loaded(FakeInferenceModel([segment()])), np.zeros(16_000, dtype=np.float32)
    )
    json.dumps(result.to_summary())
    with pytest.raises(FrozenInstanceError):
        result.input_num_samples = 1


def test_settings_forbid_unknown_and_boolean_limits():
    with pytest.raises(ValidationError):
        transcription.TranscriptionSettings(extra_limit=1)
    with pytest.raises(ValidationError):
        transcription.TranscriptionSettings(max_segments=True)


def test_normal_package_import_does_not_import_faster_whisper(tmp_path):
    code = "import sys, audio_sentinel.transcription; assert 'faster_whisper' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
