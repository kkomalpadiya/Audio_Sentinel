"""A6.3 tests for the future trainable risk-model interface."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from audio_sentinel.contracts import EventLabel, RiskLevel
from audio_sentinel.language_contracts import LanguageCategory
from audio_sentinel import risk_model as model
from audio_sentinel.risk_contracts import RiskInputSet, RiskInputStatus


@pytest.fixture
def risk_inputs(project_root: Path) -> RiskInputSet:
    document = json.loads(
        (project_root / "docs" / "examples" / "risk-assessment.json").read_text(
            encoding="utf-8"
        )
    )
    return RiskInputSet.model_validate(document["inputs"])


@pytest.fixture
def descriptor() -> model.RiskModelDescriptor:
    return model.RiskModelDescriptor(
        model_id="future-risk-model",
        model_version="1.0.0",
        artifact_sha256="a" * 64,
        runtime_distribution="example-runtime",
        runtime_version="1.0",
        training_dataset_sha256="b" * 64,
        training_example_count=100,
        validation_example_count=20,
    )


def target(score: float = 82) -> model.RiskTrainingTarget:
    return model.RiskTrainingTarget(
        score=score,
        severity=model.severity_for_score(score),
        label_source="adjudicated",
    )


def test_extracts_complete_canonical_features(risk_inputs: RiskInputSet) -> None:
    features = model.extract_risk_model_features(risk_inputs)

    assert features.feature_schema_version == "1.0"
    assert tuple(item.label for item in features.acoustic_by_label) == tuple(
        EventLabel
    )
    assert tuple(item.category for item in features.language_by_category) == tuple(
        LanguageCategory
    )
    explosion = next(
        item for item in features.acoustic_by_label if item.label is EventLabel.EXPLOSION
    )
    distress = next(
        item
        for item in features.language_by_category
        if item.category is LanguageCategory.DISTRESS
    )
    assert (explosion.event_count, explosion.max_peak_score) == (1, 0.93)
    assert distress.finding_count == 1
    assert features.speech_segment_count == 2
    assert features.accepted_transcript_count == 1
    assert features.review_required_transcript_count == 1


def test_feature_digest_is_deterministic_and_content_sensitive(
    risk_inputs: RiskInputSet,
) -> None:
    features = model.extract_risk_model_features(risk_inputs)
    same = model.extract_risk_model_features(risk_inputs)
    changed = features.model_copy(update={"speech_segment_count": 3})

    assert model.risk_model_feature_sha256(features) == (
        model.risk_model_feature_sha256(same)
    )
    assert model.risk_model_feature_sha256(features) != (
        model.risk_model_feature_sha256(changed)
    )


def test_features_exclude_source_identity_and_private_content(
    risk_inputs: RiskInputSet,
) -> None:
    features = model.extract_risk_model_features(risk_inputs)
    serialized = features.model_dump_json()

    assert risk_inputs.source.clip_id not in serialized
    assert risk_inputs.source.consent_id not in serialized
    assert set(features.model_dump()).isdisjoint(
        {"clip_id", "consent_id", "transcript_text", "speaker_id", "evidence_id"}
    )


def test_nonpresent_branches_produce_zeroed_canonical_features(
    risk_inputs: RiskInputSet,
) -> None:
    payload = risk_inputs.model_dump(mode="python")
    payload["source"]["processing_scope"] = "acoustic_only"
    payload["acoustic"] = {
        "status": "missing",
        "evidence": None,
        "event_count": 0,
        "max_peak_score": None,
        "signals": [],
    }
    payload["speech"] = {
        "status": "not_permitted",
        "evidence": None,
        "segment_count": 0,
        "accepted_transcript_count": 0,
        "review_required_transcript_count": 0,
        "max_vad_score": None,
    }
    payload["language"] = {
        "status": "not_permitted",
        "evidence": None,
        "finding_count": 0,
        "signals": [],
    }

    features = model.extract_risk_model_features(RiskInputSet.model_validate(payload))

    assert features.acoustic_status is RiskInputStatus.MISSING
    assert features.speech_status is RiskInputStatus.NOT_PERMITTED
    assert features.language_status is RiskInputStatus.NOT_PERMITTED
    assert all(item.event_count == 0 for item in features.acoustic_by_label)
    assert all(item.finding_count == 0 for item in features.language_by_category)


@pytest.mark.parametrize("invalid", [{}, {"source": "wrong"}, None])
def test_feature_extraction_rejects_untyped_inputs(invalid: object) -> None:
    with pytest.raises(model.RiskModelInterfaceError) as error:
        model.extract_risk_model_features(invalid)

    assert error.value.code == "invalid_inputs"


def test_feature_extraction_rejects_tampered_typed_inputs(
    risk_inputs: RiskInputSet,
) -> None:
    changed = risk_inputs.model_copy(update={"speech": {"status": "present"}})

    with pytest.warns(UserWarning, match="Expected `SpeechRiskInputs`"):
        with pytest.raises(model.RiskModelInterfaceError) as error:
            model.extract_risk_model_features(changed)

    assert error.value.code == "invalid_inputs"


@pytest.mark.parametrize(
    "change",
    ["missing_label", "reordered_labels", "wrong_acoustic_count", "wrong_language_count"],
)
def test_feature_contract_rejects_inventory_drift(
    risk_inputs: RiskInputSet,
    change: str,
) -> None:
    payload = model.extract_risk_model_features(risk_inputs).model_dump(mode="python")
    payload["acoustic_by_label"] = list(payload["acoustic_by_label"])
    if change == "missing_label":
        payload["acoustic_by_label"].pop()
    elif change == "reordered_labels":
        payload["acoustic_by_label"][0], payload["acoustic_by_label"][1] = (
            payload["acoustic_by_label"][1],
            payload["acoustic_by_label"][0],
        )
    elif change == "wrong_acoustic_count":
        payload["acoustic_event_count"] += 1
    else:
        payload["language_finding_count"] += 1

    with pytest.raises(ValidationError):
        model.RiskModelFeatureVector.model_validate(payload)


@pytest.mark.parametrize(
    ("score", "severity"),
    [
        (0, RiskLevel.NONE),
        (1, RiskLevel.LOW),
        (25, RiskLevel.MEDIUM),
        (50, RiskLevel.HIGH),
        (75, RiskLevel.CRITICAL),
        (100, RiskLevel.CRITICAL),
    ],
)
def test_training_target_uses_existing_score_bands(
    score: int,
    severity: RiskLevel,
) -> None:
    training_target = model.RiskTrainingTarget(
        score=score,
        severity=severity,
        label_source="human_reviewed",
    )

    assert training_target.severity is severity


def test_training_target_rejects_mismatched_severity() -> None:
    with pytest.raises(ValidationError, match="must match"):
        model.RiskTrainingTarget(
            score=10,
            severity=RiskLevel.CRITICAL,
            label_source="adjudicated",
        )


def test_training_example_pins_source_and_feature_content(
    risk_inputs: RiskInputSet,
) -> None:
    example = model.build_risk_training_example(
        "training-example-001",
        risk_inputs,
        target(),
    )

    assert len(example.source_risk_inputs_sha256) == 64
    assert example.feature_sha256 == model.risk_model_feature_sha256(example.features)
    assert example.target.score == 82
    assert risk_inputs.source.clip_id not in example.model_dump_json()


def test_training_example_rejects_changed_feature_digest(
    risk_inputs: RiskInputSet,
) -> None:
    example = model.build_risk_training_example(
        "training-example-001",
        risk_inputs,
        target(),
    )
    payload = example.model_dump(mode="python")
    payload["feature_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="differs"):
        model.RiskTrainingExample.model_validate(payload)


def test_training_example_requires_typed_reviewed_target(
    risk_inputs: RiskInputSet,
) -> None:
    with pytest.raises(model.RiskModelInterfaceError) as error:
        model.build_risk_training_example(
            "training-example-001",
            risk_inputs,
            {"score": 82},
        )

    assert error.value.code == "invalid_target"


def test_model_descriptor_pins_training_and_runtime_provenance(
    descriptor: model.RiskModelDescriptor,
) -> None:
    assert descriptor.interface_version == "1.0"
    assert descriptor.feature_schema_version == "1.0"
    assert descriptor.output_semantics == "risk_score_0_100"
    assert descriptor.training_example_count == 100


@pytest.mark.parametrize(
    "change",
    [
        {"artifact_sha256": "bad"},
        {"training_dataset_sha256": "0"},
        {"training_example_count": 0},
        {"validation_example_count": -1},
        {"unexpected": "field"},
    ],
)
def test_model_descriptor_rejects_invalid_provenance(change: dict[str, object]) -> None:
    payload = {
        "model_id": "future-risk-model",
        "model_version": "1.0.0",
        "artifact_sha256": "a" * 64,
        "runtime_distribution": "example-runtime",
        "runtime_version": "1.0",
        "training_dataset_sha256": "b" * 64,
        "training_example_count": 1,
        "validation_example_count": 0,
        **change,
    }

    with pytest.raises(ValidationError):
        model.RiskModelDescriptor.model_validate(payload)


def test_prediction_uses_model_provenance_and_existing_score_bands(
    descriptor: model.RiskModelDescriptor,
    risk_inputs: RiskInputSet,
) -> None:
    features = model.extract_risk_model_features(risk_inputs)
    prediction = model.RiskModelPrediction(
        model=descriptor,
        feature_sha256=model.risk_model_feature_sha256(features),
        score=82,
        severity=RiskLevel.CRITICAL,
        confidence_score=0.91,
    )

    assert prediction.model is descriptor
    assert prediction.confidence_score == 0.91


def test_prediction_rejects_invalid_severity_or_confidence(
    descriptor: model.RiskModelDescriptor,
) -> None:
    with pytest.raises(ValidationError, match="must match"):
        model.RiskModelPrediction(
            model=descriptor,
            feature_sha256="c" * 64,
            score=10,
            severity=RiskLevel.CRITICAL,
        )
    with pytest.raises(ValidationError):
        model.RiskModelPrediction(
            model=descriptor,
            feature_sha256="c" * 64,
            score=10,
            severity=RiskLevel.LOW,
            confidence_score=1.1,
        )


class StubPredictor:
    def __init__(self, descriptor: model.RiskModelDescriptor) -> None:
        self._descriptor = descriptor

    @property
    def descriptor(self) -> model.RiskModelDescriptor:
        return self._descriptor

    def predict(
        self, features: model.RiskModelFeatureVector
    ) -> model.RiskModelPrediction:
        return model.RiskModelPrediction(
            model=self.descriptor,
            feature_sha256=model.risk_model_feature_sha256(features),
            score=0,
            severity=RiskLevel.NONE,
        )


class StubTrainer:
    def __init__(self, descriptor: model.RiskModelDescriptor) -> None:
        self.descriptor = descriptor

    def train(
        self,
        training_examples: Sequence[model.RiskTrainingExample],
        validation_examples: Sequence[model.RiskTrainingExample] = (),
    ) -> model.RiskModelPredictor:
        del training_examples, validation_examples
        return StubPredictor(self.descriptor)


def test_stub_implementations_satisfy_and_execute_protocols(
    descriptor: model.RiskModelDescriptor,
    risk_inputs: RiskInputSet,
) -> None:
    example = model.build_risk_training_example(
        "training-example-001", risk_inputs, target()
    )
    trainer = StubTrainer(descriptor)
    predictor = trainer.train([example])

    assert isinstance(trainer, model.RiskModelTrainer)
    assert isinstance(predictor, model.RiskModelPredictor)
    assert predictor.predict(example.features).score == 0


def test_records_are_immutable(
    descriptor: model.RiskModelDescriptor,
    risk_inputs: RiskInputSet,
) -> None:
    features = model.extract_risk_model_features(risk_inputs)

    with pytest.raises(ValidationError):
        features.speech_segment_count = 0
    with pytest.raises(ValidationError):
        descriptor.model_version = "changed"


def test_schema_export_is_portable(tmp_path: Path) -> None:
    exported = model.write_risk_model_schemas(tmp_path)

    assert set(exported) == {
        "risk-model-features.schema.json",
        "risk-training-example.schema.json",
        "risk-model-descriptor.schema.json",
        "risk-model-prediction.schema.json",
    }
    for filename, path in exported.items():
        assert json.loads(path.read_text(encoding="utf-8")) == (
            model.risk_model_schema_documents()[filename]
        )


def test_interface_loads_without_network_or_ml_runtime() -> None:
    code = """
import sys
class NoModelImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'onnxruntime'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
def reject_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError('Unexpected network access')
sys.addaudithook(reject_network)
from audio_sentinel.risk_model import RiskModelFeatureVector
assert RiskModelFeatureVector.model_fields['feature_schema_version'].default == '1.0'
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        capture_output=True,
        timeout=30,
    )
