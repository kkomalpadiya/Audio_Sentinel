"""A7.3 tests for the future trainable verification-model interface."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from audio_sentinel import verification_model as model
from audio_sentinel.consensus_contracts import (
    BranchAgreement,
    ConsensusOutcome,
)
from audio_sentinel.consensus_rules import (
    AgreementReasonCode,
    ConsensusAgreementEvaluation,
    evaluate_evidence_agreement,
    load_agreement_rule_set_bytes,
    load_builtin_agreement_rule_set,
)
from audio_sentinel.risk_contracts import (
    RiskAssessmentDocument,
    RiskEvidenceKind,
    RiskInputSet,
    RiskInputStatus,
    RiskReasonCode,
)
from audio_sentinel.risk_scoring import score_risk


NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


@pytest.fixture
def assessment(project_root: Path) -> RiskAssessmentDocument:
    return RiskAssessmentDocument.model_validate_json(
        (project_root / "docs" / "examples" / "risk-assessment.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def agreement(project_root: Path) -> ConsensusAgreementEvaluation:
    return ConsensusAgreementEvaluation.model_validate_json(
        (project_root / "docs" / "examples" / "consensus-agreement.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def descriptor() -> model.VerificationModelDescriptor:
    return model.VerificationModelDescriptor(
        model_id="future-verification-model",
        model_version="1.0.0",
        artifact_sha256="a" * 64,
        runtime_distribution="example-runtime",
        runtime_version="1.0",
        training_dataset_sha256="b" * 64,
        training_example_count=100,
        validation_example_count=20,
    )


def target(
    outcome: ConsensusOutcome = ConsensusOutcome.REVIEW,
) -> model.VerificationTrainingTarget:
    return model.VerificationTrainingTarget(
        outcome=outcome,
        label_source="adjudicated",
    )


def _outcome_scores(
    no_action: float,
    log: float,
    review: float,
    alert: float,
) -> tuple[model.VerificationOutcomeScore, ...]:
    return tuple(
        model.VerificationOutcomeScore(outcome=outcome, probability=probability)
        for outcome, probability in zip(
            ConsensusOutcome,
            (no_action, log, review, alert),
            strict=True,
        )
    )


def test_extracts_complete_canonical_features(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    features = model.extract_verification_model_features(assessment, agreement)

    assert features.feature_schema_version == "1.0"
    assert features.risk_score == 82
    assert features.risk_severity.value == "critical"
    assert features.risk_human_review_required is True
    assert tuple(item.reason for item in features.risk_reason_features) == tuple(
        RiskReasonCode
    )
    assert tuple(branch.kind for branch in features.branches) == tuple(
        RiskEvidenceKind
    )
    assert all(
        tuple(item.reason for item in branch.reason_features)
        == tuple(AgreementReasonCode)
        for branch in features.branches
    )
    assert features.supporting_branch_count == 2
    assert features.eligible_supporting_branch_count == 2
    assert features.conflicting_branch_count == 0
    assert features.has_conflict is False


def test_feature_reason_flags_preserve_phase_six_and_b71_semantics(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    features = model.extract_verification_model_features(assessment, agreement)
    risk_flags = {item.reason: item.present for item in features.risk_reason_features}
    acoustic_flags = {
        item.reason: item.present for item in features.branches[0].reason_features
    }
    language_flags = {
        item.reason: item.present for item in features.branches[2].reason_features
    }

    assert risk_flags[RiskReasonCode.TRANSCRIPT_REVIEW_REQUIRED] is True
    assert risk_flags[RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED] is False
    assert acoustic_flags[AgreementReasonCode.HIGH_CONFIDENCE_RISK_SIGNAL] is True
    assert language_flags[AgreementReasonCode.ACTIVE_LANGUAGE_RISK] is True
    assert features.branches[1].agreement is BranchAgreement.NEUTRAL


def test_feature_digest_is_deterministic_and_content_sensitive(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    features = model.extract_verification_model_features(assessment, agreement)
    same = model.extract_verification_model_features(assessment, agreement)
    changed = features.model_copy(update={"risk_score": 83})

    assert model.verification_model_feature_sha256(features) == (
        model.verification_model_feature_sha256(same)
    )
    assert model.verification_model_feature_sha256(features) != (
        model.verification_model_feature_sha256(changed)
    )


def test_features_exclude_source_identity_decision_labels_and_private_content(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    features = model.extract_verification_model_features(assessment, agreement)
    serialized = features.model_dump_json()

    for forbidden_value in (
        assessment.assessment_id,
        assessment.inputs.source.clip_id,
        assessment.inputs.source.consent_id,
        agreement.evaluation_id,
    ):
        assert forbidden_value not in serialized
    assert set(features.model_dump()).isdisjoint(
        {
            "assessment_id",
            "evaluation_id",
            "clip_id",
            "consent_id",
            "evidence_id",
            "transcript_text",
            "speaker_id",
            "decision_id",
            "outcome",
            "alert_candidate",
            "created_at",
        }
    )


def test_nonpresent_branches_produce_explicit_zero_identity_free_features(
    assessment: RiskAssessmentDocument,
) -> None:
    payload = assessment.inputs.model_dump(mode="python")
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
    missing_assessment = score_risk(RiskInputSet.model_validate(payload), now=NOW)
    missing_agreement = evaluate_evidence_agreement(missing_assessment, now=NOW)

    features = model.extract_verification_model_features(
        missing_assessment, missing_agreement
    )

    assert features.missing_branch_count == 1
    assert features.not_permitted_branch_count == 2
    assert features.supporting_branch_count == 0
    assert features.branches[0].agreement is BranchAgreement.UNAVAILABLE
    assert features.branches[1].agreement is BranchAgreement.NOT_PERMITTED


@pytest.mark.parametrize("assessment_value,agreement_value", [(None, None), ({}, {})])
def test_feature_extraction_rejects_untyped_sources(
    assessment_value: object,
    agreement_value: object,
) -> None:
    with pytest.raises(model.VerificationModelInterfaceError) as captured:
        model.extract_verification_model_features(
            assessment_value, agreement_value  # type: ignore[arg-type]
        )

    assert captured.value.code == "invalid_sources"


def test_feature_extraction_rejects_mismatched_sources(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    changed_assessment = assessment.model_copy(
        update={"created_at": assessment.created_at + timedelta(minutes=1)}
    )

    with pytest.raises(model.VerificationModelInterfaceError) as captured:
        model.extract_verification_model_features(changed_assessment, agreement)

    assert captured.value.code == "invalid_sources"


def test_custom_agreement_requires_the_exact_trusted_rule_artifact(
    assessment: RiskAssessmentDocument,
) -> None:
    loaded = load_builtin_agreement_rule_set()
    data = loaded.rule_set.model_dump(mode="json")
    data["rule_set_id"] = "future-verification-feature-rules"
    data["rule_set_version"] = "test-1"
    data["acoustic_support"]["minimum_peak_score"] = 0.95
    custom = load_agreement_rule_set_bytes(
        (json.dumps(data, indent=2) + "\n").encode("utf-8")
    )
    custom_agreement = evaluate_evidence_agreement(
        assessment, rule_set=custom, now=NOW
    )

    with pytest.raises(model.VerificationModelInterfaceError):
        model.extract_verification_model_features(assessment, custom_agreement)

    features = model.extract_verification_model_features(
        assessment,
        custom_agreement,
        trusted_rule_set=custom,
    )
    assert features.eligible_supporting_branch_count == 1


@pytest.mark.parametrize(
    "change",
    [
        "missing_risk_reason",
        "reordered_branches",
        "missing_agreement_reason",
        "empty_agreement_reasons",
        "wrong_support_count",
        "wrong_conflict_flag",
    ],
)
def test_feature_contract_rejects_inventory_drift(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
    change: str,
) -> None:
    payload = model.extract_verification_model_features(
        assessment, agreement
    ).model_dump(mode="python")
    payload["risk_reason_features"] = list(payload["risk_reason_features"])
    payload["branches"] = list(payload["branches"])
    if change == "missing_risk_reason":
        payload["risk_reason_features"].pop()
    elif change == "reordered_branches":
        payload["branches"][0], payload["branches"][1] = (
            payload["branches"][1],
            payload["branches"][0],
        )
    elif change == "missing_agreement_reason":
        payload["branches"][0]["reason_features"] = list(
            payload["branches"][0]["reason_features"]
        )[:-1]
    elif change == "empty_agreement_reasons":
        for reason_feature in payload["branches"][0]["reason_features"]:
            reason_feature["present"] = False
    elif change == "wrong_support_count":
        payload["supporting_branch_count"] = 1
    else:
        payload["has_conflict"] = True

    with pytest.raises(ValidationError):
        model.VerificationModelFeatureVector.model_validate(payload)


@pytest.mark.parametrize("outcome", list(ConsensusOutcome))
def test_training_target_accepts_every_reviewed_consensus_outcome(
    outcome: ConsensusOutcome,
) -> None:
    training_target = model.VerificationTrainingTarget(
        outcome=outcome,
        label_source="human_reviewed",
    )

    assert training_target.outcome is outcome


def test_training_target_rejects_unreviewed_labels() -> None:
    with pytest.raises(ValidationError):
        model.VerificationTrainingTarget(
            outcome=ConsensusOutcome.REVIEW,
            label_source="generated",  # type: ignore[arg-type]
        )


def test_training_example_pins_both_sources_and_feature_content(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    example = model.build_verification_training_example(
        "verification-example-001",
        assessment,
        agreement,
        target(),
    )

    assert len(example.source_assessment_sha256) == 64
    assert len(example.source_agreement_sha256) == 64
    assert example.feature_sha256 == model.verification_model_feature_sha256(
        example.features
    )
    assert assessment.assessment_id not in example.model_dump_json()
    assert agreement.evaluation_id not in example.model_dump_json()


def test_training_example_rejects_changed_feature_digest(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    example = model.build_verification_training_example(
        "verification-example-001", assessment, agreement, target()
    )
    payload = example.model_dump(mode="python")
    payload["feature_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="differs"):
        model.VerificationTrainingExample.model_validate(payload)


def test_training_example_requires_typed_reviewed_target(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    with pytest.raises(model.VerificationModelInterfaceError) as captured:
        model.build_verification_training_example(
            "verification-example-001",
            assessment,
            agreement,
            {"outcome": "review"},  # type: ignore[arg-type]
        )

    assert captured.value.code == "invalid_target"


def test_model_descriptor_pins_interface_policy_training_and_runtime_provenance(
    descriptor: model.VerificationModelDescriptor,
) -> None:
    assert descriptor.interface_version == "1.0"
    assert descriptor.feature_schema_version == "1.0"
    assert descriptor.output_semantics == "consensus_outcome_probabilities_v1"
    assert descriptor.risk_assessment_schema_version == "1.0"
    assert descriptor.agreement_schema_version == "1.0"
    assert descriptor.consensus_policy_version == "1.0"
    assert descriptor.training_example_count == 100


@pytest.mark.parametrize(
    "change",
    [
        {"artifact_sha256": "bad"},
        {"training_dataset_sha256": "0"},
        {"training_example_count": 0},
        {"validation_example_count": -1},
        {"consensus_policy_version": "2.0"},
        {"unexpected": "field"},
    ],
)
def test_model_descriptor_rejects_invalid_provenance(
    change: dict[str, object],
) -> None:
    payload = {
        "model_id": "future-verification-model",
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
        model.VerificationModelDescriptor.model_validate(payload)


def test_prediction_carries_probabilities_provenance_and_safety_receipts(
    descriptor: model.VerificationModelDescriptor,
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    features = model.extract_verification_model_features(assessment, agreement)
    prediction = model.VerificationModelPrediction(
        model=descriptor,
        feature_sha256=model.verification_model_feature_sha256(features),
        outcome_scores=_outcome_scores(0.05, 0.1, 0.8, 0.05),
        recommended_outcome=ConsensusOutcome.REVIEW,
        confidence_score=0.8,
    )

    assert prediction.model is descriptor
    assert prediction.requires_deterministic_verification is True
    assert prediction.alert_is_local_candidate_only is True
    assert prediction.recommended_outcome is ConsensusOutcome.REVIEW


@pytest.mark.parametrize(
    "change,match",
    [
        ({"outcome_scores": _outcome_scores(0.1, 0.2, 0.7, 0.0)[:-1]}, "order"),
        ({"outcome_scores": _outcome_scores(0.1, 0.2, 0.6, 0.2)}, "sum"),
        ({"recommended_outcome": ConsensusOutcome.LOG}, "maximum"),
        ({"confidence_score": 0.7}, "confidence"),
        ({"requires_deterministic_verification": False}, "literal"),
    ],
)
def test_prediction_rejects_invalid_probabilities_or_safety_state(
    descriptor: model.VerificationModelDescriptor,
    change: dict[str, object],
    match: str,
) -> None:
    payload = {
        "model": descriptor,
        "feature_sha256": "c" * 64,
        "outcome_scores": _outcome_scores(0.05, 0.1, 0.8, 0.05),
        "recommended_outcome": ConsensusOutcome.REVIEW,
        "confidence_score": 0.8,
        **change,
    }

    with pytest.raises(ValidationError, match=match):
        model.VerificationModelPrediction.model_validate(payload)


class StubPredictor:
    def __init__(self, descriptor: model.VerificationModelDescriptor) -> None:
        self._descriptor = descriptor

    @property
    def descriptor(self) -> model.VerificationModelDescriptor:
        return self._descriptor

    def predict(
        self, features: model.VerificationModelFeatureVector
    ) -> model.VerificationModelPrediction:
        return model.VerificationModelPrediction(
            model=self.descriptor,
            feature_sha256=model.verification_model_feature_sha256(features),
            outcome_scores=_outcome_scores(0.8, 0.1, 0.1, 0.0),
            recommended_outcome=ConsensusOutcome.NO_ACTION,
            confidence_score=0.8,
        )


class StubTrainer:
    def __init__(self, descriptor: model.VerificationModelDescriptor) -> None:
        self.descriptor = descriptor

    def train(
        self,
        training_examples: Sequence[model.VerificationTrainingExample],
        validation_examples: Sequence[model.VerificationTrainingExample] = (),
    ) -> model.VerificationModelPredictor:
        del training_examples, validation_examples
        return StubPredictor(self.descriptor)


def test_stub_implementations_satisfy_and_execute_protocols(
    descriptor: model.VerificationModelDescriptor,
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    example = model.build_verification_training_example(
        "verification-example-001", assessment, agreement, target()
    )
    trainer = StubTrainer(descriptor)
    predictor = trainer.train([example])

    assert isinstance(trainer, model.VerificationModelTrainer)
    assert isinstance(predictor, model.VerificationModelPredictor)
    assert predictor.predict(example.features).recommended_outcome is (
        ConsensusOutcome.NO_ACTION
    )


def test_records_are_immutable(
    descriptor: model.VerificationModelDescriptor,
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> None:
    features = model.extract_verification_model_features(assessment, agreement)

    with pytest.raises(ValidationError):
        features.risk_score = 0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        descriptor.model_version = "changed"  # type: ignore[misc]


def test_schema_export_is_portable(tmp_path: Path) -> None:
    exported = model.write_verification_model_schemas(tmp_path)

    assert set(exported) == {
        "verification-model-features.schema.json",
        "verification-training-example.schema.json",
        "verification-model-descriptor.schema.json",
        "verification-model-prediction.schema.json",
    }
    for filename, path in exported.items():
        assert json.loads(path.read_text(encoding="utf-8")) == (
            model.verification_model_schema_documents()[filename]
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
from audio_sentinel.verification_model import VerificationModelFeatureVector
assert VerificationModelFeatureVector.model_fields['feature_schema_version'].default == '1.0'
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        capture_output=True,
        timeout=30,
    )
