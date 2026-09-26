"""A7.3 contracts for a future trainable verification implementation."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.consensus_contracts import (
    BranchAgreement,
    CONSENSUS_POLICY_VERSION,
    ConsensusOutcome,
)
from audio_sentinel.consensus_rules import (
    AGREEMENT_EVALUATION_SCHEMA_VERSION,
    AgreementReasonCode,
    BranchAgreementEvaluation,
    ConsensusAgreementEvaluation,
    LoadedAgreementRuleSet,
    risk_assessment_sha256,
)
from audio_sentinel.consensus_service import (
    ConsensusVerificationError,
    FinalVerificationService,
)
from audio_sentinel.contracts import RiskLevel
from audio_sentinel.preparation import Identifier
from audio_sentinel.risk_contracts import (
    RISK_ASSESSMENT_SCHEMA_VERSION,
    RiskAssessmentDocument,
    RiskEvidenceKind,
    RiskInputStatus,
    RiskReasonCode,
    severity_for_score,
)


VERIFICATION_MODEL_INTERFACE_VERSION = "1.0"
VERIFICATION_MODEL_FEATURE_SCHEMA_VERSION = "1.0"
VERIFICATION_MODEL_OUTPUT_SEMANTICS = "consensus_outcome_probabilities_v1"


class VerificationModelInterfaceError(RuntimeError):
    """Stable interface failure code plus a safe, non-sensitive explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class VerificationModelRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class RiskReasonFeature(VerificationModelRecord):
    """One fixed-position Phase 6 reason indicator."""

    reason: RiskReasonCode
    present: bool


class AgreementReasonFeature(VerificationModelRecord):
    """One fixed-position B7.1 branch-reason indicator."""

    reason: AgreementReasonCode
    present: bool


class VerificationBranchFeature(VerificationModelRecord):
    """Identity-free agreement features for one evidence branch."""

    kind: RiskEvidenceKind
    input_status: RiskInputStatus
    agreement: BranchAgreement
    reason_features: tuple[AgreementReasonFeature, ...]

    @model_validator(mode="after")
    def validate_reasons(self) -> "VerificationBranchFeature":
        if tuple(item.reason for item in self.reason_features) != tuple(
            AgreementReasonCode
        ):
            raise ValueError(
                "agreement reason features must use complete canonical order"
            )
        BranchAgreementEvaluation(
            kind=self.kind,
            input_status=self.input_status,
            agreement=self.agreement,
            reason_codes=tuple(
                item.reason for item in self.reason_features if item.present
            ),
        )
        return self


class VerificationModelFeatureVector(VerificationModelRecord):
    """Fixed-shape, identity-free features for a future verification model."""

    feature_schema_version: Literal["1.0"] = (
        VERIFICATION_MODEL_FEATURE_SCHEMA_VERSION
    )
    risk_score: float = Field(ge=0, le=100)
    risk_severity: RiskLevel
    risk_human_review_required: bool
    risk_reason_features: tuple[RiskReasonFeature, ...]
    branches: tuple[VerificationBranchFeature, ...]
    supporting_branch_count: int = Field(ge=0, le=3, strict=True)
    eligible_supporting_branch_count: int = Field(ge=0, le=2, strict=True)
    conflicting_branch_count: int = Field(ge=0, le=3, strict=True)
    missing_branch_count: int = Field(ge=0, le=3, strict=True)
    not_permitted_branch_count: int = Field(ge=0, le=3, strict=True)
    has_conflict: bool

    @model_validator(mode="after")
    def validate_inventory(self) -> "VerificationModelFeatureVector":
        if self.risk_severity is not severity_for_score(self.risk_score):
            raise ValueError("feature severity must match the risk score")
        if tuple(item.reason for item in self.risk_reason_features) != tuple(
            RiskReasonCode
        ):
            raise ValueError("risk reason features must use complete canonical order")
        if tuple(branch.kind for branch in self.branches) != tuple(RiskEvidenceKind):
            raise ValueError("branch features must use complete canonical order")

        supporting = tuple(
            branch
            for branch in self.branches
            if branch.agreement is BranchAgreement.SUPPORTS_RISK
        )
        eligible_support = tuple(
            branch
            for branch in supporting
            if branch.kind
            in {RiskEvidenceKind.ACOUSTIC, RiskEvidenceKind.LANGUAGE}
        )
        conflicts = tuple(
            branch
            for branch in self.branches
            if branch.agreement is BranchAgreement.CONFLICTS_RISK
        )
        missing = tuple(
            branch
            for branch in self.branches
            if branch.input_status is RiskInputStatus.MISSING
        )
        not_permitted = tuple(
            branch
            for branch in self.branches
            if branch.input_status is RiskInputStatus.NOT_PERMITTED
        )
        expected = (
            len(supporting),
            len(eligible_support),
            len(conflicts),
            len(missing),
            len(not_permitted),
        )
        actual = (
            self.supporting_branch_count,
            self.eligible_supporting_branch_count,
            self.conflicting_branch_count,
            self.missing_branch_count,
            self.not_permitted_branch_count,
        )
        if actual != expected:
            raise ValueError("verification feature counts differ from branch inventory")
        if self.has_conflict is not bool(conflicts):
            raise ValueError("has_conflict must match the branch inventory")
        return self


class VerificationTrainingTarget(VerificationModelRecord):
    """Human-reviewed or adjudicated consensus outcome used as training truth."""

    outcome: ConsensusOutcome
    label_source: Literal["human_reviewed", "adjudicated"]


class VerificationTrainingExample(VerificationModelRecord):
    """Hash-pinned verification features paired with reviewed training truth."""

    example_id: Identifier
    source_assessment_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_agreement_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    features: VerificationModelFeatureVector
    target: VerificationTrainingTarget

    @model_validator(mode="after")
    def validate_feature_hash(self) -> "VerificationTrainingExample":
        if self.feature_sha256 != verification_model_feature_sha256(self.features):
            raise ValueError("training feature hash differs from feature content")
        return self


class VerificationModelDescriptor(VerificationModelRecord):
    """Reproducibility metadata for a separately trained model artifact."""

    interface_version: Literal["1.0"] = VERIFICATION_MODEL_INTERFACE_VERSION
    feature_schema_version: Literal["1.0"] = (
        VERIFICATION_MODEL_FEATURE_SCHEMA_VERSION
    )
    output_semantics: Literal["consensus_outcome_probabilities_v1"] = (
        VERIFICATION_MODEL_OUTPUT_SEMANTICS
    )
    risk_assessment_schema_version: Literal["1.0"] = (
        RISK_ASSESSMENT_SCHEMA_VERSION
    )
    agreement_schema_version: Literal["1.0"] = (
        AGREEMENT_EVALUATION_SCHEMA_VERSION
    )
    consensus_policy_version: Literal["1.0"] = CONSENSUS_POLICY_VERSION
    model_id: Identifier
    model_version: str = Field(min_length=1, max_length=128)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    runtime_distribution: str = Field(min_length=1, max_length=128)
    runtime_version: str = Field(min_length=1, max_length=128)
    training_dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_example_count: int = Field(gt=0, strict=True)
    validation_example_count: int = Field(ge=0, strict=True)


class VerificationOutcomeScore(VerificationModelRecord):
    """Candidate probability for one canonical A7.1 outcome."""

    outcome: ConsensusOutcome
    probability: float = Field(ge=0, le=1)


class VerificationModelPrediction(VerificationModelRecord):
    """Candidate recommendation that still requires deterministic verification."""

    model: VerificationModelDescriptor
    feature_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    outcome_scores: tuple[VerificationOutcomeScore, ...]
    recommended_outcome: ConsensusOutcome
    confidence_score: float = Field(ge=0, le=1)
    requires_deterministic_verification: Literal[True] = True
    alert_is_local_candidate_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_prediction(self) -> "VerificationModelPrediction":
        if tuple(item.outcome for item in self.outcome_scores) != tuple(
            ConsensusOutcome
        ):
            raise ValueError("outcome scores must use complete canonical order")
        total = sum(item.probability for item in self.outcome_scores)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("outcome probabilities must sum to one")
        selected = next(
            item
            for item in self.outcome_scores
            if item.outcome is self.recommended_outcome
        )
        maximum = max(item.probability for item in self.outcome_scores)
        if not math.isclose(
            selected.probability, maximum, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("recommended outcome must have maximum probability")
        if not math.isclose(
            self.confidence_score,
            selected.probability,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("confidence must equal the recommended probability")
        return self


@runtime_checkable
class VerificationModelPredictor(Protocol):
    @property
    def descriptor(self) -> VerificationModelDescriptor: ...

    def predict(
        self, features: VerificationModelFeatureVector
    ) -> VerificationModelPrediction: ...


@runtime_checkable
class VerificationModelTrainer(Protocol):
    def train(
        self,
        training_examples: Sequence[VerificationTrainingExample],
        validation_examples: Sequence[VerificationTrainingExample] = (),
    ) -> VerificationModelPredictor: ...


def extract_verification_model_features(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
    *,
    trusted_rule_set: LoadedAgreementRuleSet | None = None,
) -> VerificationModelFeatureVector:
    """Build canonical identity-free features from trusted Phase 7 inputs."""

    validated_assessment, validated_agreement = _validated_sources(
        assessment,
        agreement,
        trusted_rule_set=trusted_rule_set,
    )
    return _features_from_validated(validated_assessment, validated_agreement)


def _features_from_validated(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
) -> VerificationModelFeatureVector:
    risk_reasons = set(assessment.reason_codes)
    branches = tuple(
        VerificationBranchFeature(
            kind=branch.kind,
            input_status=branch.input_status,
            agreement=branch.agreement,
            reason_features=tuple(
                AgreementReasonFeature(
                    reason=reason,
                    present=reason in set(branch.reason_codes),
                )
                for reason in AgreementReasonCode
            ),
        )
        for branch in agreement.branches
    )
    supporting = tuple(
        branch
        for branch in branches
        if branch.agreement is BranchAgreement.SUPPORTS_RISK
    )
    conflicts = tuple(
        branch
        for branch in branches
        if branch.agreement is BranchAgreement.CONFLICTS_RISK
    )
    return VerificationModelFeatureVector(
        risk_score=assessment.score,
        risk_severity=assessment.severity,
        risk_human_review_required=assessment.human_review_required,
        risk_reason_features=tuple(
            RiskReasonFeature(reason=reason, present=reason in risk_reasons)
            for reason in RiskReasonCode
        ),
        branches=branches,
        supporting_branch_count=len(supporting),
        eligible_supporting_branch_count=sum(
            branch.kind in {RiskEvidenceKind.ACOUSTIC, RiskEvidenceKind.LANGUAGE}
            for branch in supporting
        ),
        conflicting_branch_count=len(conflicts),
        missing_branch_count=sum(
            branch.input_status is RiskInputStatus.MISSING for branch in branches
        ),
        not_permitted_branch_count=sum(
            branch.input_status is RiskInputStatus.NOT_PERMITTED
            for branch in branches
        ),
        has_conflict=bool(conflicts),
    )


def verification_model_feature_sha256(
    features: VerificationModelFeatureVector,
) -> str:
    """Return the deterministic digest for one validated feature vector."""

    if not isinstance(features, VerificationModelFeatureVector):
        raise TypeError("verification model features have an invalid type")
    validated = VerificationModelFeatureVector.model_validate(
        features.model_dump(mode="python")
    )
    if validated != features:
        raise ValueError("verification model feature integrity validation failed")
    return _canonical_hash(validated.model_dump(mode="json"))


def build_verification_training_example(
    example_id: str,
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
    target: VerificationTrainingTarget,
    *,
    trusted_rule_set: LoadedAgreementRuleSet | None = None,
) -> VerificationTrainingExample:
    """Create one provenance-pinned supervised example without source identity."""

    if not isinstance(target, VerificationTrainingTarget):
        raise VerificationModelInterfaceError(
            "invalid_target", "A reviewed verification training target is required."
        )
    try:
        validated_target = VerificationTrainingTarget.model_validate(
            target.model_dump(mode="python")
        )
    except Exception as error:
        raise VerificationModelInterfaceError(
            "invalid_target", "Verification training target failed validation."
        ) from error
    if validated_target != target:
        raise VerificationModelInterfaceError(
            "invalid_target", "Verification training target failed validation."
        )

    validated_assessment, validated_agreement = _validated_sources(
        assessment,
        agreement,
        trusted_rule_set=trusted_rule_set,
    )
    features = _features_from_validated(validated_assessment, validated_agreement)
    return VerificationTrainingExample(
        example_id=example_id,
        source_assessment_sha256=risk_assessment_sha256(validated_assessment),
        source_agreement_sha256=_canonical_hash(
            validated_agreement.model_dump(mode="json")
        ),
        feature_sha256=verification_model_feature_sha256(features),
        features=features,
        target=validated_target,
    )


def verification_model_schema_documents() -> dict[str, dict[str, object]]:
    """Return portable schemas for future verification-model exchange records."""

    return {
        "verification-model-features.schema.json": (
            VerificationModelFeatureVector.model_json_schema()
        ),
        "verification-training-example.schema.json": (
            VerificationTrainingExample.model_json_schema()
        ),
        "verification-model-descriptor.schema.json": (
            VerificationModelDescriptor.model_json_schema()
        ),
        "verification-model-prediction.schema.json": (
            VerificationModelPrediction.model_json_schema()
        ),
    }


def write_verification_model_schemas(
    output_directory: Path,
) -> dict[str, Path]:
    """Export the trainable-model exchange schemas for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in verification_model_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported


def _validated_sources(
    assessment: RiskAssessmentDocument,
    agreement: ConsensusAgreementEvaluation,
    *,
    trusted_rule_set: LoadedAgreementRuleSet | None,
) -> tuple[RiskAssessmentDocument, ConsensusAgreementEvaluation]:
    if not isinstance(assessment, RiskAssessmentDocument) or not isinstance(
        agreement, ConsensusAgreementEvaluation
    ):
        raise VerificationModelInterfaceError(
            "invalid_sources",
            "Validated Phase 6 and B7.1 source documents are required.",
        )
    try:
        validated_assessment = RiskAssessmentDocument.model_validate(
            assessment.model_dump(mode="python")
        )
        validated_agreement = ConsensusAgreementEvaluation.model_validate(
            agreement.model_dump(mode="python")
        )
        FinalVerificationService(
            trusted_rule_set=trusted_rule_set
        ).decide(
            validated_assessment,
            validated_agreement,
            now=validated_agreement.created_at,
        )
    except ConsensusVerificationError as error:
        raise VerificationModelInterfaceError(
            "invalid_sources",
            "Verification model sources failed trusted alignment checks.",
        ) from error
    except Exception as error:
        raise VerificationModelInterfaceError(
            "invalid_sources", "Verification model sources failed validation."
        ) from error
    if validated_assessment != assessment or validated_agreement != agreement:
        raise VerificationModelInterfaceError(
            "invalid_sources", "Verification model source integrity check failed."
        )
    return validated_assessment, validated_agreement


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()
