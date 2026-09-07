"""Check mapping integrity and prevent unsupported semantic/risk inferences."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from audio_sentinel import acoustic_model as model
from audio_sentinel.contracts import EventLabel, SPEECH_EVENT_LABELS


def test_official_vocabulary_and_complete_project_coverage() -> None:
    model.validate_label_mapping()
    assert len(model.load_class_map()) == 521
    assert len(model.LABEL_MAPPING) == 6
    assert len(model.DEFERRED_LABELS) == 7
    assert SPEECH_EVENT_LABELS <= model.DEFERRED_LABELS.keys()


@pytest.mark.parametrize("index", [6, 9, 11, 19, 64, 65, 304, 382, 389, 425, 426, 435, 436])
def test_context_and_lookalike_classes_do_not_directly_assert_target_events(index: int) -> None:
    assert index not in {c.index for rows in model.LABEL_MAPPING.values() for c in rows}


def test_changed_reference_bytes_are_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resource = tmp_path / "resources"
    resource.mkdir()
    (resource / "yamnet_class_map.csv").write_bytes(b"index,mid,display_name\n0,/m/09x0r,Changed\n")
    monkeypatch.setattr(model, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        model.load_class_map()


@pytest.mark.parametrize("change", ["index", "mid", "display_name"])
def test_mapping_drift_is_rejected(monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    mapping = dict(model.LABEL_MAPPING)
    target = mapping[EventLabel.GUNSHOT][0]
    values = {"index": 520, "mid": "/wrong", "display_name": "Wrong class"}
    mapping[EventLabel.GUNSHOT] = (replace(target, **{change: values[change]}),)
    monkeypatch.setattr(model, "LABEL_MAPPING", mapping)
    with pytest.raises(ValueError, match="differs from official vocabulary"):
        model.validate_label_mapping()


def test_duplicate_mapping_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = dict(model.LABEL_MAPPING)
    mapping[EventLabel.EXPLOSION] = mapping[EventLabel.GUNSHOT]
    monkeypatch.setattr(model, "LABEL_MAPPING", mapping)
    with pytest.raises(ValueError, match="mapped more than once"):
        model.validate_label_mapping()


def test_missing_label_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    deferred = dict(model.DEFERRED_LABELS)
    deferred.pop(EventLabel.CROWD_PANIC)
    monkeypatch.setattr(model, "DEFERRED_LABELS", deferred)
    with pytest.raises(ValueError, match="Every project label"):
        model.validate_label_mapping()


def test_specification_and_mapping_are_immutable() -> None:
    with pytest.raises(AttributeError):
        model.YAMNET.sample_rate_hz = 8000
    with pytest.raises(TypeError):
        model.LABEL_MAPPING[EventLabel.AMBIENT] = ()


def test_contract_import_and_validation_work_without_network_or_ml_runtime() -> None:
    code = """
import sys
class NoModelImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'tensorflow', 'tensorflow_hub', 'torch', 'onnxruntime'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
def reject_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError('Unexpected network access')
sys.addaudithook(reject_network)
from audio_sentinel.acoustic_model import validate_label_mapping
validate_label_mapping()
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True, timeout=30)
