"""B3.1 tests for local artifact, runtime, and signature verification."""

from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_sentinel import acoustic_loader as loader
from audio_sentinel.acoustic_model import load_class_map
from audio_sentinel.config import Paths


class FakeShape:
    def __init__(self, *values):
        self.values = values

    def as_list(self):
        return list(self.values)


class FakeSpec:
    def __init__(self, shape, dtype="float32"):
        self.shape = FakeShape(*shape)
        self.dtype = SimpleNamespace(name=dtype)


class FakeSignature:
    def __init__(self, *, classes=521):
        self.structured_input_signature = ((), {"waveform": FakeSpec((None,))})
        self.structured_outputs = {
            "output_0": FakeSpec((None, classes)),
            "output_1": FakeSpec((None, 1024)),
            "output_2": FakeSpec((None, 64)),
        }


class FakeAssetPath:
    def __init__(self, path: Path):
        self.path = path

    def numpy(self):
        return os.fsencode(self.path)


class FakeModel:
    tensorflow_version = "2.3.0"
    tensorflow_git_version = "v2.3.0-test"

    def __init__(self, directory: Path, *, classes=521):
        self.directory = directory
        self.signatures = {"serving_default": FakeSignature(classes=classes)}

    def __call__(self, waveform):
        raise AssertionError("B3.1 must not run inference")

    def class_map_path(self):
        return FakeAssetPath(self.directory / "assets/yamnet_class_map.csv")


def model_tree(tmp_path: Path) -> tuple[Paths, Path]:
    root = tmp_path / "project"
    directory = root / "models/yamnet/1"
    (directory / "assets").mkdir(parents=True)
    (directory / "variables").mkdir()
    # acoustic_loader deliberately does not own resource lookup. Use the already
    # verified public reader and reproduce its exact CSV bytes from the package.
    from importlib.resources import files
    (directory / "assets/yamnet_class_map.csv").write_bytes(
        files("audio_sentinel").joinpath("resources/yamnet_class_map.csv").read_bytes()
    )
    (directory / "saved_model.pb").write_bytes(b"saved-model-test")
    (directory / "variables/variables.data-00000-of-00001").write_bytes(b"weights-test")
    (directory / "variables/variables.index").write_bytes(b"index-test")
    return Paths.from_root(root), directory


def install_fake_runtime(monkeypatch: pytest.MonkeyPatch, directory: Path, *, classes=521,
                         runtime_version="test-runtime") -> FakeModel:
    model = FakeModel(directory, classes=classes)
    runtime = SimpleNamespace(
        __version__=runtime_version,
        saved_model=SimpleNamespace(load=lambda path: model),
    )
    monkeypatch.setattr(loader, "import_module", lambda name: runtime)
    return model


def pin_fixture(monkeypatch: pytest.MonkeyPatch, directory: Path, *, runtime_version="test-runtime") -> str:
    digest = loader.model_artifact_sha256(directory)
    monkeypatch.setattr(loader, "YAMNET", replace(
        loader.YAMNET, artifact_sha256=digest, runtime_version=runtime_version
    ))
    return digest


def test_loads_local_model_and_returns_json_ready_version_metadata(tmp_path, monkeypatch):
    paths, directory = model_tree(tmp_path)
    digest = pin_fixture(monkeypatch, directory)
    model = install_fake_runtime(monkeypatch, directory)

    loaded = loader.load_yamnet(paths)

    assert loaded.model is model
    assert len(loaded.classes) == 521 and loaded.classes == load_class_map()
    assert loaded.metadata.artifact_sha256 == digest
    assert loaded.metadata.model_path == "yamnet/1"
    assert loaded.metadata.runtime_version == "test-runtime"
    assert loaded.metadata.exported_with_tensorflow == "2.3.0"
    assert loaded.metadata.input == loader.TensorContract("waveform", (None,), "float32")
    assert [spec.shape for spec in loaded.metadata.outputs] == [(None, 521), (None, 1024), (None, 64)]
    document = json.dumps(loaded.metadata.as_dict(), sort_keys=True)
    assert str(tmp_path) not in document and "label_mapping_version" in document


def test_rejects_changed_artifact_before_importing_tensorflow(tmp_path, monkeypatch):
    paths, directory = model_tree(tmp_path)
    pin_fixture(monkeypatch, directory)
    (directory / "saved_model.pb").write_bytes(b"changed")
    monkeypatch.setattr(loader, "import_module", lambda name: pytest.fail("runtime imported too early"))

    with pytest.raises(loader.AcousticModelLoadError, match="digest") as error:
        loader.load_yamnet(paths)
    assert error.value.code == "artifact_mismatch"


@pytest.mark.parametrize("relative", ["missing/1", "../outside", Path("C:/outside")])
def test_rejects_missing_or_uncontained_model_paths(tmp_path, relative):
    paths, _ = model_tree(tmp_path)
    with pytest.raises(loader.AcousticModelLoadError) as error:
        loader.load_yamnet(paths, relative)
    assert error.value.code in {"invalid_path", "model_not_found"}


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_requires_exact_saved_model_file_inventory(tmp_path, change):
    paths, directory = model_tree(tmp_path)
    if change == "missing":
        (directory / "variables/variables.index").unlink()
    else:
        (directory / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(loader.AcousticModelLoadError, match="missing or unexpected") as error:
        loader.model_artifact_sha256(directory)
    assert error.value.code == "invalid_artifact"


def test_zero_byte_download_marker_is_allowed_but_excluded_from_artifact_identity(tmp_path):
    _, directory = model_tree(tmp_path)
    baseline = loader.model_artifact_sha256(directory)
    marker = directory / loader.DOWNLOAD_MARKER
    marker.parent.mkdir(parents=True)
    marker.touch()
    assert loader.model_artifact_sha256(directory) == baseline


@pytest.mark.parametrize("change", ["nonempty", "linked"])
def test_rejects_invalid_download_marker(tmp_path, monkeypatch, change):
    _, directory = model_tree(tmp_path)
    marker = directory / loader.DOWNLOAD_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"invalid" if change == "nonempty" else b"")
    if change == "linked":
        original = loader._is_link
        monkeypatch.setattr(loader, "_is_link", lambda path: path == marker or original(path))
    with pytest.raises(loader.AcousticModelLoadError) as error:
        loader.model_artifact_sha256(directory)
    assert error.value.code == "invalid_artifact"


def test_model_artifact_byte_limit_is_enforced_before_hashing(tmp_path, monkeypatch):
    _, directory = model_tree(tmp_path)
    monkeypatch.setattr(loader, "MAX_MODEL_BYTES", 1)
    with pytest.raises(loader.AcousticModelLoadError) as error:
        loader.model_artifact_sha256(directory)
    assert error.value.code == "model_too_large"


@pytest.mark.parametrize("linked_name,expected_code", [
    ("yamnet", "invalid_path"),
    ("saved_model.pb", "invalid_artifact"),
])
def test_rejects_links_in_model_directory_or_payload(tmp_path, monkeypatch, linked_name, expected_code):
    paths, directory = model_tree(tmp_path)
    original = loader._is_link
    monkeypatch.setattr(
        loader, "_is_link", lambda path: path.name == linked_name or original(path)
    )
    with pytest.raises(loader.AcousticModelLoadError) as error:
        loader.load_yamnet(paths)
    assert error.value.code == expected_code


def test_detects_payload_change_during_direct_hashing(tmp_path, monkeypatch):
    _, directory = model_tree(tmp_path)
    real_fstat = loader.os.fstat
    calls = 0

    def changing_fstat(file_descriptor):
        nonlocal calls
        calls += 1
        details = real_fstat(file_descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_size=details.st_size,
                st_mtime_ns=details.st_mtime_ns,
                st_ctime_ns=details.st_ctime_ns + 1,
            )
        return details

    monkeypatch.setattr(loader.os, "fstat", changing_fstat)
    with pytest.raises(loader.AcousticModelLoadError) as error:
        loader.model_artifact_sha256(directory)
    assert error.value.code == "artifact_changed"


def test_reports_missing_runtime_after_artifact_verification(tmp_path, monkeypatch):
    paths, directory = model_tree(tmp_path)
    pin_fixture(monkeypatch, directory)

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(loader, "import_module", missing)
    with pytest.raises(loader.AcousticModelLoadError, match="setup_yamnet") as error:
        loader.load_yamnet(paths)
    assert error.value.code == "runtime_missing"


def test_rejects_unapproved_runtime_version(tmp_path, monkeypatch):
    paths, directory = model_tree(tmp_path)
    pin_fixture(monkeypatch, directory, runtime_version="2.21.0")
    install_fake_runtime(monkeypatch, directory, runtime_version="2.20.0")
    with pytest.raises(loader.AcousticModelLoadError, match="2.21.0") as error:
        loader.load_yamnet(paths)
    assert error.value.code == "runtime_mismatch"


def test_rejects_signature_drift_without_running_inference(tmp_path, monkeypatch):
    paths, directory = model_tree(tmp_path)
    pin_fixture(monkeypatch, directory)
    install_fake_runtime(monkeypatch, directory, classes=520)
    with pytest.raises(loader.AcousticModelLoadError, match="shape or dtype") as error:
        loader.load_yamnet(paths)
    assert error.value.code == "signature_mismatch"


@pytest.mark.parametrize("change", [
    "not_mapping", "extra_signature", "not_callable", "positional_input",
    "input_name", "output_name", "input_dtype", "output_dtype",
])
def test_rejects_complete_signature_inventory_and_tensor_drift(tmp_path, change):
    model = FakeModel(tmp_path)
    signature = model.signatures["serving_default"]
    if change == "not_mapping":
        model.signatures = []
    elif change == "extra_signature":
        model.signatures["other"] = FakeSignature()
    elif change == "not_callable":
        model = SimpleNamespace(signatures={"serving_default": signature})
    elif change == "positional_input":
        signature.structured_input_signature = ((FakeSpec((None,)),), {"waveform": FakeSpec((None,))})
    elif change == "input_name":
        signature.structured_input_signature = ((), {"audio": FakeSpec((None,))})
    elif change == "output_name":
        signature.structured_outputs["scores"] = signature.structured_outputs.pop("output_0")
    elif change == "input_dtype":
        signature.structured_input_signature = ((), {"waveform": FakeSpec((None,), "float64")})
    else:
        signature.structured_outputs["output_2"] = FakeSpec((None, 64), "float64")
    with pytest.raises(loader.AcousticModelLoadError) as error:
        loader._verify_signature(model)
    assert error.value.code == "signature_mismatch"


def test_accepts_string_class_map_path_and_unknown_exporter_metadata(tmp_path, monkeypatch):
    paths, directory = model_tree(tmp_path)
    pin_fixture(monkeypatch, directory)
    model = install_fake_runtime(monkeypatch, directory)
    model.class_map_path = lambda: str(directory / "assets/yamnet_class_map.csv")
    monkeypatch.delattr(FakeModel, "tensorflow_version")
    monkeypatch.delattr(FakeModel, "tensorflow_git_version")
    loaded = loader.load_yamnet(paths)
    assert loaded.metadata.exported_with_tensorflow == "unknown"
    assert loaded.metadata.exported_with_tensorflow_git == "unknown"


def test_rejects_loaded_model_class_map_redirection(tmp_path, monkeypatch):
    paths, directory = model_tree(tmp_path)
    pin_fixture(monkeypatch, directory)
    model = install_fake_runtime(monkeypatch, directory)
    outside = tmp_path / "outside.csv"
    outside.write_text("not the vocabulary", encoding="utf-8")
    model.class_map_path = lambda: FakeAssetPath(outside)
    with pytest.raises(loader.AcousticModelLoadError) as error:
        loader.load_yamnet(paths)
    assert error.value.code == "class_map_mismatch"


@pytest.mark.parametrize("failure", [AttributeError("missing"), ValueError("invalid")])
def test_wraps_tensorflow_load_or_inspection_failures(tmp_path, monkeypatch, failure):
    paths, directory = model_tree(tmp_path)
    pin_fixture(monkeypatch, directory)
    runtime = SimpleNamespace(
        __version__="test-runtime",
        saved_model=SimpleNamespace(load=lambda path: (_ for _ in ()).throw(failure)),
    )
    monkeypatch.setattr(loader, "import_module", lambda name: runtime)
    with pytest.raises(loader.AcousticModelLoadError) as error:
        loader.load_yamnet(paths)
    assert error.value.code == "model_load_failed"


def test_rejects_artifact_change_during_tensorflow_load(tmp_path, monkeypatch):
    paths, directory = model_tree(tmp_path)
    digest = pin_fixture(monkeypatch, directory)
    install_fake_runtime(monkeypatch, directory)
    checks = iter((digest, "0" * 64))
    monkeypatch.setattr(loader, "model_artifact_sha256", lambda path: next(checks))
    with pytest.raises(loader.AcousticModelLoadError, match="while TensorFlow loaded") as error:
        loader.load_yamnet(paths)
    assert error.value.code == "artifact_changed"


def test_module_import_does_not_require_tensorflow():
    assert "tensorflow" not in loader.__dict__


def test_loader_contract_objects_are_immutable(tmp_path, monkeypatch):
    paths, directory = model_tree(tmp_path)
    pin_fixture(monkeypatch, directory)
    install_fake_runtime(monkeypatch, directory)
    loaded = loader.load_yamnet(paths)
    with pytest.raises(FrozenInstanceError):
        loaded.metadata.model_version = "changed"
    with pytest.raises(FrozenInstanceError):
        loaded.metadata.input.dtype = "float64"
