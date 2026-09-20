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


@pytest.mark.parametrize("change", ["bad_json", "too_large", "wrong_value"])
def test_loader_rejects_invalid_install_marker(tmp_path, monkeypatch, change):
    _, directory, _ = model_tree(tmp_path, monkeypatch)
    marker = directory / transcription.INSTALL_MARKER
    if change == "bad_json":
        marker.write_text("not json", encoding="utf-8")
    elif change == "too_large":
        monkeypatch.setattr(transcription, "MAX_MARKER_BYTES", 1)
    else:
        value = transcription.WHISPER_TINY_EN.marker()
        value["source_revision"] = "other"
        marker.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.model_artifact_sha256(directory)

    assert error.value.code == "invalid_artifact"


@pytest.mark.parametrize(
    "linked_name,expected_code",
    [
        ("faster-whisper-tiny.en", "invalid_path"),
        ("model.bin", "invalid_artifact"),
        (transcription.INSTALL_MARKER, "invalid_artifact"),
    ],
)
def test_loader_rejects_links_in_model_path_or_payload(
    tmp_path, monkeypatch, linked_name, expected_code
):
    paths, _, _ = model_tree(tmp_path, monkeypatch)
    original = transcription._is_link
    monkeypatch.setattr(
        transcription,
        "_is_link",
        lambda path: path.name == linked_name or original(path),
    )

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths)

    assert error.value.code == expected_code


def test_loader_checks_exact_payload_size_before_hashing(tmp_path, monkeypatch):
    _, directory, _ = model_tree(tmp_path, monkeypatch)
    (directory / "model.bin").write_bytes(b"wrong-size")
    real_hash = transcription._hash_regular_file

    def guarded_hash(path):
        if path.name == "model.bin":
            pytest.fail("wrong-size payload must not be hashed")
        return real_hash(path)

    monkeypatch.setattr(
        transcription,
        "_hash_regular_file",
        guarded_hash,
    )

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.model_artifact_sha256(directory)

    assert error.value.code == "invalid_artifact"


def test_loader_detects_payload_change_during_hash(tmp_path, monkeypatch):
    _, directory, _ = model_tree(tmp_path, monkeypatch)
    real_fstat = transcription.os.fstat
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

    monkeypatch.setattr(transcription.os, "fstat", changing_fstat)

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.model_artifact_sha256(directory)

    assert error.value.code == "artifact_changed"


def test_changed_inventory_digest_is_rejected_before_runtime_check(tmp_path, monkeypatch):
    paths, directory, _ = model_tree(tmp_path, monkeypatch)
    changed = replace(transcription.WHISPER_TINY_EN, artifact_sha256="0" * 64)
    monkeypatch.setattr(transcription, "WHISPER_TINY_EN", changed)
    (directory / transcription.INSTALL_MARKER).write_text(
        json.dumps(transcription.WHISPER_TINY_EN.marker()), encoding="utf-8"
    )
    monkeypatch.setattr(
        transcription,
        "_verify_runtime_versions",
        lambda: pytest.fail("runtime checked before artifact identity"),
    )

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths)

    assert error.value.code == "artifact_mismatch"


def test_loader_rejects_runtime_version_drift(tmp_path, monkeypatch):
    paths, _, _ = model_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(transcription, "_distribution_version", lambda _name: "wrong")

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths)

    assert error.value.code == "runtime_mismatch"


def test_loader_reports_missing_runtime_distribution(tmp_path, monkeypatch):
    paths, _, _ = model_tree(tmp_path, monkeypatch)

    def missing(_name):
        raise transcription.importlib_metadata.PackageNotFoundError("missing")

    monkeypatch.setattr(transcription.importlib_metadata, "version", missing)

    with pytest.raises(transcription.TranscriptionError, match="setup_speech") as error:
        transcription.load_transcription_model(paths)

    assert error.value.code == "runtime_missing"


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


def test_loader_rejects_missing_transcription_entry_point(tmp_path, monkeypatch):
    paths, _, _ = model_tree(tmp_path, monkeypatch)
    install_fake_loader(monkeypatch)
    monkeypatch.setattr(FakeLoadModel, "transcribe", None)

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths)

    assert error.value.code == "signature_mismatch"


def test_loader_wraps_model_creation_failure_without_leaking_details(tmp_path, monkeypatch):
    paths, _, _ = model_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transcription,
        "_distribution_version",
        lambda name: {
            "faster-whisper": "test-fw",
            "ctranslate2": "test-ct2",
            "tokenizers": "test-tok",
        }[name],
    )
    runtime = SimpleNamespace(
        WhisperModel=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private model path")
        )
    )
    monkeypatch.setattr(transcription, "import_module", lambda _name: runtime)

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths)

    assert error.value.code == "model_load_failed"
    assert "private" not in str(error.value)


def test_loader_detects_artifact_change_while_model_loads(tmp_path, monkeypatch):
    paths, _, spec = model_tree(tmp_path, monkeypatch)
    install_fake_loader(monkeypatch)
    hashes = iter((spec.artifact_sha256, "0" * 64))
    monkeypatch.setattr(
        transcription,
        "model_artifact_sha256",
        lambda _directory: next(hashes),
    )

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.load_transcription_model(paths)

    assert error.value.code == "artifact_changed"


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
    assert kwargs == {
        "language": "en",
        "task": "transcribe",
        "beam_size": 1,
        "best_of": 1,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "vad_filter": False,
        "word_timestamps": False,
        "without_timestamps": False,
        "initial_prompt": None,
        "hotwords": None,
        "suppress_blank": True,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }


def test_transcription_owns_model_input_and_preserves_source_waveform():
    class MutatingModel(FakeInferenceModel):
        def transcribe(self, samples, **kwargs):
            result = super().transcribe(samples, **kwargs)
            samples[:] = 1.0
            return result

    source = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    before = source.copy()

    transcription.transcribe_segment(loaded(MutatingModel([segment()])), source)

    assert np.array_equal(source, before)


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


def test_zero_token_chunk_uses_one_weight_and_preserves_valid_confidence_endpoints():
    result = transcription.transcribe_segment(
        loaded(FakeInferenceModel([segment(text="certain", tokens=(), avg_logprob=0.0)])),
        np.zeros(16_000, dtype=np.float32),
    )

    assert result.candidate.confidence_score == 1.0
    assert result.chunks[0].token_count == 0


@pytest.mark.parametrize(
    "waveform",
    [
        [0.0],
        np.array([], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(4, dtype=np.float64),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([1.01], dtype=np.float32),
        np.array([-1.01], dtype=np.float32),
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


def test_input_and_output_limits_are_inclusive_at_exact_boundaries():
    fake = FakeInferenceModel([segment(text="hello", tokens=(1, 2))])
    result = transcription.transcribe_segment(
        loaded(fake),
        np.zeros(16_000, dtype=np.float32),
        settings=transcription.TranscriptionSettings(
            max_input_samples=16_000,
            max_segments=1,
            max_tokens=2,
            max_output_bytes=5,
        ),
    )

    assert result.candidate.text == "hello"


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
    "bad_segment",
    [
        segment(id=-1),
        segment(id="0"),
        segment(start=True),
        segment(start=float("nan")),
        segment(end=0.4, start=0.5),
        segment(avg_logprob=True),
        segment(avg_logprob=float("-inf")),
        segment(no_speech_prob=-0.1),
        segment(no_speech_prob=True),
        segment(tokens="12"),
        segment(tokens=(-1,)),
        segment(tokens=(1.5,)),
        segment(text="hello\x00world"),
    ],
)
def test_rejects_additional_malformed_segment_fields(bad_segment):
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel([bad_segment])), np.zeros(16_000, dtype=np.float32)
        )

    assert error.value.code == "invalid_output"


def test_rejects_decreasing_segment_start_times():
    chunks = [
        segment(id=0, start=0.5, end=0.75),
        segment(id=1, start=0.25, end=0.5),
    ]

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel(chunks)), np.zeros(16_000, dtype=np.float32)
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


def test_output_byte_limit_counts_utf8_bytes_not_characters():
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel([segment(text="é", tokens=(1,))])),
            np.zeros(16_000, dtype=np.float32),
            settings=transcription.TranscriptionSettings(max_output_bytes=1),
        )

    assert error.value.code == "output_too_large"


def test_whitespace_segments_still_count_toward_resource_limits():
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel([segment(text="   ", tokens=(1, 2))])),
            np.zeros(16_000, dtype=np.float32),
            settings=transcription.TranscriptionSettings(max_tokens=1),
        )

    assert error.value.code == "output_too_large"


def test_wraps_lazy_segment_iteration_failure_in_safe_error():
    def broken_segments():
        yield segment()
        raise RuntimeError("private iterator detail")

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel(broken_segments())),
            np.zeros(16_000, dtype=np.float32),
        )

    assert error.value.code == "invalid_output"
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    "info",
    [
        SimpleNamespace(language="fr", language_probability=1.0, duration=1.0),
        SimpleNamespace(language="en", language_probability=0.9, duration=1.0),
        SimpleNamespace(language="en", language_probability=1.0, duration=2.0),
        SimpleNamespace(language="en", language_probability=True, duration=1.0),
        SimpleNamespace(language="en", language_probability=1.0, duration=float("nan")),
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


def test_reports_model_memory_failure_separately():
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(
            loaded(FakeInferenceModel(error=MemoryError())),
            np.zeros(16_000, dtype=np.float32),
        )

    assert error.value.code == "insufficient_memory"


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "other"),
        ("model_version", "other"),
        ("model_source", "other"),
        ("model_path", "other/1"),
        ("source_repository", "other"),
        ("source_revision", "other"),
        ("artifact_sha256", "0" * 64),
        ("artifacts", ()),
        ("runtime_distribution", "other"),
        ("runtime_version", "other"),
        ("ctranslate2_version", "other"),
        ("tokenizers_version", "other"),
        ("device", "cuda"),
        ("compute_type", "float32"),
        ("language", "fr"),
        ("sample_rate_hz", 8_000),
    ],
)
def test_rejects_tampered_loaded_metadata_before_inference(field, value):
    item = loaded()
    tampered = replace(item, metadata=replace(item.metadata, **{field: value}))
    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(tampered, np.zeros(16_000, dtype=np.float32))
    assert error.value.code == "model_mismatch"


def test_rejects_loaded_model_without_callable_transcriber():
    item = loaded(SimpleNamespace(transcribe=None))

    with pytest.raises(transcription.TranscriptionError) as error:
        transcription.transcribe_segment(item, np.zeros(16_000, dtype=np.float32))

    assert error.value.code == "model_mismatch"


def test_results_are_immutable_and_json_ready():
    result = transcription.transcribe_segment(
        loaded(FakeInferenceModel([segment()])), np.zeros(16_000, dtype=np.float32)
    )
    json.dumps(result.to_summary())
    with pytest.raises(FrozenInstanceError):
        result.input_num_samples = 1
    with pytest.raises(FrozenInstanceError):
        result.chunks[0].text = "changed"
    with pytest.raises(FrozenInstanceError):
        result.model.model_id = "changed"
    summary = json.dumps(result.to_summary())
    assert "waveform" not in summary


def test_settings_forbid_unknown_and_boolean_limits():
    with pytest.raises(ValidationError):
        transcription.TranscriptionSettings(extra_limit=1)
    with pytest.raises(ValidationError):
        transcription.TranscriptionSettings(max_segments=True)


def test_normal_package_import_does_not_import_faster_whisper(tmp_path):
    code = "import sys, audio_sentinel.transcription; assert 'faster_whisper' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
