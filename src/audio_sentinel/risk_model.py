"""A6.3 contracts for a future trainable risk-scoring implementation."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.contracts import EventLabel, RiskLevel
from audio_sentinel.language_contracts import LanguageCategory
from audio_sentinel.preparation import Identifier
from audio_sentinel.risk_contracts import (
    RiskInputSet,
    RiskInputStatus,
    severity_for_score,
)


RISK_MODEL_INTERFACE_VERSION = "1.0"
RISK_MODEL_FEATURE_SCHEMA_VERSION = "1.0"
RISK_MODEL_OUTPUT_SEMANTICS = "risk_score_0_100"


class RiskModelInterfaceError(RuntimeError):
    """Stable interface failure code plus a safe, non-sensitive explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RiskModelRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class AcousticModelFeature(RiskModelRecord):
    """Aggregate count and maximum confidence for one canonical event label."""

    label: EventLabel
    event_count: int = Field(ge=0, strict=True)
    max_peak_score: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> "AcousticModelFeature":
        if self.event_count == 0 and self.max_peak_score is not None:
            raise ValueError("empty acoustic feature requires null max_peak_score")
        if self.event_count > 0 and self.max_peak_score is None:
            raise ValueError("populated acoustic feature requires max_peak_score")
        return self


class LanguageModelFeature(RiskModelRecord):
    """Finding count for one canonical language category."""

    category: LanguageCategory
    finding_count: int = Field(ge=0, strict=True)


class RiskModelFeatureVector(RiskModelRecord):
    """Identity-free Phase 6 features available to a future risk model."""

    feature_schema_version: Literal["1.0"] = RISK_MODEL_FEATURE_SCHEMA_VERSION
    acoustic_status: RiskInputStatus
    acoustic_event_count: int = Field(ge=0, strict=True)
    acoustic_max_peak_score: float | None = Field(default=None, ge=0, le=1)
    acoustic_by_label: tuple[AcousticModelFeature, ...]
    speech_status: RiskInputStatus
    speech_segment_count: int = Field(ge=0, strict=True)
    accepted_transcript_count: int = Field(ge=0, strict=True)
    review_required_transcript_count: int = Field(ge=0, strict=True)
    max_vad_score: float | None = Field(default=None, ge=0, le=1)
    language_status: RiskInputStatus
    language_finding_count: int = Field(ge=0, strict=True)
    language_by_category: tuple[LanguageModelFeature, ...]

    @model_validator(mode="after")
    def validate_feature_inventory(self) -> "RiskModelFeatureVector":
        if tuple(item.label for item in self.acoustic_by_label) != tuple(EventLabel):
            raise ValueError("acoustic model features must use canonical label order")
        if tuple(item.category for item in self.language_by_category) != tuple(
            LanguageCategory
        ):
            raise ValueError("language model features must use canonical category order")
        if self.acoustic_event_count != sum(
            item.event_count for item in self.acoustic_by_label
        ):
            raise ValueError("acoustic feature count differs from label inventory")
        expected_acoustic_peak = max(
            (
                item.max_peak_score
                for item in self.acoustic_by_label
                if item.max_peak_score is not None
            ),
            default=None,
        )
        if expected_acoustic_peak is None:
            if self.acoustic_max_peak_score is not None:
                raise ValueError("empty acoustic inventory requires null maximum")
        elif self.acoustic_max_peak_score is None or not math.isclose(
            self.acoustic_max_peak_score,
            expected_acoustic_peak,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("acoustic maximum differs from label inventory")
        if self.language_finding_count != sum(
            item.finding_count for item in self.language_by_category
        ):
            raise ValueError("language feature count differs from category inventory")
        if self.accepted_transcript_count > self.speech_segment_count:
            raise ValueError("accepted transcript count exceeds speech segments")
        if self.review_required_transcript_count > self.speech_segment_count:
            raise ValueError("review-required transcript count exceeds speech segments")
        if self.speech_segment_count == 0 and self.max_vad_score is not None:
            raise ValueError("zero speech segments require null max_vad_score")
        if self.speech_segment_count > 0 and self.max_vad_score is None:
            raise ValueError("speech segments require max_vad_score")
        self._validate_branch_statuses()
        return self

    def _validate_branch_statuses(self) -> None:
        if self.acoustic_status is not RiskInputStatus.PRESENT and (
            self.acoustic_event_count or self.acoustic_max_peak_score is not None
        ):
            raise ValueError("non-present acoustic status cannot carry model features")
        if self.speech_status is not RiskInputStatus.PRESENT and (
            self.speech_segment_count
            or self.accepted_transcript_count
            or self.review_required_transcript_count
            or self.max_vad_score is not None
        ):
            raise ValueError("non-present speech status cannot carry model features")
        if (
            self.language_status is not RiskInputStatus.PRESENT
            and self.language_finding_count
        ):
            raise ValueError("non-present language status cannot carry model features")


class RiskTrainingTarget(RiskModelRecord):
    """Human-reviewed or adjudicated score used as supervised training truth."""

    score: float = Field(ge=0, le=100)
    severity: RiskLevel
    label_source: Literal["human_reviewed", "adjudicated"]

    @model_validator(mode="after")
    def validate_severity(self) -> "RiskTrainingTarget":
        if self.severity is not severity_for_score(self.score):
            raise ValueError("training severity must match the target score")
        return self


class RiskTrainingExample(RiskModelRecord):
    """One hash-pinned feature vector paired with reviewed training truth."""

    example_id: Identifier
    source_risk_inputs_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    features: RiskModelFeatureVector
    target: RiskTrainingTarget

    @model_validator(mode="after")
    def validate_feature_hash(self) -> "RiskTrainingExample":
        if self.feature_sha256 != risk_model_feature_sha256(self.features):
            raise ValueError("training feature hash differs from feature content")
        return self


class RiskModelDescriptor(RiskModelRecord):
    """Reproducibility metadata for a separately trained model artifact."""

    interface_version: Literal["1.0"] = RISK_MODEL_INTERFACE_VERSION
    feature_schema_version: Literal["1.0"] = RISK_MODEL_FEATURE_SCHEMA_VERSION
    output_semantics: Literal["risk_score_0_100"] = RISK_MODEL_OUTPUT_SEMANTICS
    model_id: Identifier
    model_version: str = Field(min_length=1, max_length=128)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    runtime_distribution: str = Field(min_length=1, max_length=128)
    runtime_version: str = Field(min_length=1, max_length=128)
    training_dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_example_count: int = Field(gt=0, strict=True)
    validation_example_count: int = Field(ge=0, strict=True)


class RiskModelPrediction(RiskModelRecord):
    """Candidate 0-100 model output, not a consensus decision or alert."""

    model: RiskModelDescriptor
    feature_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    score: float = Field(ge=0, le=100)
    severity: RiskLevel
    confidence_score: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_severity(self) -> "RiskModelPrediction":
        if self.severity is not severity_for_score(self.score):
            raise ValueError("prediction severity must match the predicted score")
        return self


@runtime_checkable
class RiskModelPredictor(Protocol):
    @property
    def descriptor(self) -> RiskModelDescriptor: ...

    def predict(self, features: RiskModelFeatureVector) -> RiskModelPrediction: ...


@runtime_checkable
class RiskModelTrainer(Protocol):
    def train(
        self,
        training_examples: Sequence[RiskTrainingExample],
        validation_examples: Sequence[RiskTrainingExample] = (),
    ) -> RiskModelPredictor: ...


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _validated_inputs(inputs: RiskInputSet) -> RiskInputSet:
    if not isinstance(inputs, RiskInputSet):
        raise RiskModelInterfaceError(
            "invalid_inputs", "A6.1 risk inputs are required for model features."
        )
    try:
        validated = RiskInputSet.model_validate(inputs.model_dump(mode="python"))
    except Exception as error:
        raise RiskModelInterfaceError(
            "invalid_inputs", "Risk inputs failed model-feature validation."
        ) from error
    if validated != inputs:
        raise RiskModelInterfaceError(
            "invalid_inputs", "Risk input integrity validation failed."
        )
    return validated


def extract_risk_model_features(inputs: RiskInputSet) -> RiskModelFeatureVector:
    """Build the canonical identity-free feature vector from A6.1 inputs."""

    validated = _validated_inputs(inputs)
    acoustic_by_label = tuple(
        AcousticModelFeature(
            label=label,
            event_count=sum(
                signal.label is label for signal in validated.acoustic.signals
            ),
            max_peak_score=max(
                (
                    signal.peak_score
                    for signal in validated.acoustic.signals
                    if signal.label is label
                ),
                default=None,
            ),
        )
        for label in EventLabel
    )
    language_by_category = tuple(
        LanguageModelFeature(
            category=category,
            finding_count=sum(
                signal.category is category for signal in validated.language.signals
            ),
        )
        for category in LanguageCategory
    )
    return RiskModelFeatureVector(
        acoustic_status=validated.acoustic.status,
        acoustic_event_count=validated.acoustic.event_count,
        acoustic_max_peak_score=validated.acoustic.max_peak_score,
        acoustic_by_label=acoustic_by_label,
        speech_status=validated.speech.status,
        speech_segment_count=validated.speech.segment_count,
        accepted_transcript_count=validated.speech.accepted_transcript_count,
        review_required_transcript_count=(
            validated.speech.review_required_transcript_count
        ),
        max_vad_score=validated.speech.max_vad_score,
        language_status=validated.language.status,
        language_finding_count=validated.language.finding_count,
        language_by_category=language_by_category,
    )


def risk_model_feature_sha256(features: RiskModelFeatureVector) -> str:
    """Return the deterministic digest for one validated feature vector."""

    if not isinstance(features, RiskModelFeatureVector):
        raise TypeError("risk model features have an invalid type")
    validated = RiskModelFeatureVector.model_validate(
        features.model_dump(mode="python")
    )
    if validated != features:
        raise ValueError("risk model feature integrity validation failed")
    return _canonical_hash(validated.model_dump(mode="json"))


def build_risk_training_example(
    example_id: str,
    inputs: RiskInputSet,
    target: RiskTrainingTarget,
) -> RiskTrainingExample:
    """Create one provenance-pinned supervised example without source identity."""

    validated = _validated_inputs(inputs)
    if not isinstance(target, RiskTrainingTarget):
        raise RiskModelInterfaceError(
            "invalid_target", "A reviewed risk training target is required."
        )
    features = extract_risk_model_features(validated)
    return RiskTrainingExample(
        example_id=example_id,
        source_risk_inputs_sha256=_canonical_hash(
            validated.model_dump(mode="json")
        ),
        feature_sha256=risk_model_feature_sha256(features),
        features=features,
        target=target,
    )


def risk_model_schema_documents() -> dict[str, dict[str, object]]:
    """Return portable schemas for trainable-model exchange records."""

    return {
        "risk-model-features.schema.json": RiskModelFeatureVector.model_json_schema(),
        "risk-training-example.schema.json": RiskTrainingExample.model_json_schema(),
        "risk-model-descriptor.schema.json": RiskModelDescriptor.model_json_schema(),
        "risk-model-prediction.schema.json": RiskModelPrediction.model_json_schema(),
    }


def write_risk_model_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the trainable-model exchange schemas for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in risk_model_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
