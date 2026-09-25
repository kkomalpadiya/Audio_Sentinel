"""B6.1 deterministic, configurable risk-scoring rules engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.contracts import EventLabel
from audio_sentinel.language_contracts import LanguageCategory
from audio_sentinel.risk_contracts import (
    RISK_RULE_FORMAT_VERSION,
    AcousticRiskInputs,
    LanguageRiskInputs,
    RiskAssessmentDocument,
    RiskEvidenceKind,
    RiskInputSet,
    RiskInputStatus,
    RiskMissingDataAssessment,
    RiskReasonCode,
    RiskRuleSetDescriptor,
    RiskScoringPolicy,
    SpeechRiskInputs,
    severity_for_score,
)


RISK_RULE_SET_SCHEMA_VERSION = "1.0"
BUILTIN_RISK_RULE_SET_ID = "audio-sentinel-risk-v1"
BUILTIN_RISK_RULE_SET_VERSION = "1.0.0"
BUILTIN_RISK_RULE_RESOURCE = "risk-scoring-rules-v1.json"
BUILTIN_RISK_RULE_SET_SHA256 = (
    "a34c1972f778d4965b9f71fbf372ab3180a94655c1d1c5cd8b54802d557bfc11"
)
MAX_RISK_RULE_SET_BYTES = 65_536


_SAFE_ACOUSTIC_LABELS = frozenset(
    {
        EventLabel.AMBIENT,
        EventLabel.NO_SPEECH,
        EventLabel.SPEECH_PRESENT,
        EventLabel.NON_THREATENING_SPEECH,
    }
)
_ZERO_LANGUAGE_CATEGORIES = frozenset(
    {
        LanguageCategory.NO_CONCERNING_MATCH,
        LanguageCategory.CONTEXT_SUPPRESSED,
    }
)
_LANGUAGE_REASON = {
    LanguageCategory.DISTRESS: RiskReasonCode.LANGUAGE_DISTRESS,
    LanguageCategory.THREAT: RiskReasonCode.LANGUAGE_THREAT,
    LanguageCategory.WEAPON_REFERENCE: RiskReasonCode.LANGUAGE_WEAPON_REFERENCE,
    LanguageCategory.AMBIGUOUS: RiskReasonCode.LANGUAGE_AMBIGUOUS,
    LanguageCategory.CONTEXT_SUPPRESSED: RiskReasonCode.LANGUAGE_CONTEXT_SUPPRESSED,
}
_MISSING_REASON = {
    RiskEvidenceKind.ACOUSTIC: RiskReasonCode.MISSING_ACOUSTIC_EVIDENCE,
    RiskEvidenceKind.SPEECH: RiskReasonCode.MISSING_SPEECH_EVIDENCE,
    RiskEvidenceKind.LANGUAGE: RiskReasonCode.MISSING_LANGUAGE_EVIDENCE,
}
_REASON_ORDER = {reason: index for index, reason in enumerate(RiskReasonCode)}
_KIND_ORDER = {kind: index for index, kind in enumerate(RiskEvidenceKind)}


class RiskScoringError(RuntimeError):
    """Stable scoring failure code plus a safe, non-sensitive explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RiskRuleRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class AcousticLabelScore(RiskRuleRecord):
    label: EventLabel
    points: float = Field(ge=0, le=100)


class LanguageCategoryScore(RiskRuleRecord):
    category: LanguageCategory
    points: float = Field(ge=0, le=100)


class AcousticScoringRules(RiskRuleRecord):
    aggregation: Literal["maximum_label"] = "maximum_label"
    label_scores: tuple[AcousticLabelScore, ...] = Field(min_length=1)
    high_confidence_threshold: float = Field(ge=0, le=1)
    high_confidence_bonus: float = Field(ge=0, le=100)
    max_points: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_rules(self) -> "AcousticScoringRules":
        labels = tuple(item.label for item in self.label_scores)
        if labels != tuple(EventLabel):
            raise ValueError("acoustic label scores must cover every label in canonical order")
        for item in self.label_scores:
            if item.label in _SAFE_ACOUSTIC_LABELS and item.points != 0:
                raise ValueError("non-risk acoustic labels must have zero points")
        return self


class SpeechScoringRules(RiskRuleRecord):
    segment_presence_points: float = Field(ge=0, le=100)
    review_required_points_per_transcript: float = Field(ge=0, le=100)
    max_review_required_points: float = Field(ge=0, le=100)
    max_points: float = Field(ge=0, le=100)


class LanguageScoringRules(RiskRuleRecord):
    aggregation: Literal["sum_findings"] = "sum_findings"
    category_scores: tuple[LanguageCategoryScore, ...] = Field(min_length=1)
    max_points: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_rules(self) -> "LanguageScoringRules":
        categories = tuple(item.category for item in self.category_scores)
        if categories != tuple(LanguageCategory):
            raise ValueError(
                "language category scores must cover every category in canonical order"
            )
        for item in self.category_scores:
            if item.category in _ZERO_LANGUAGE_CATEGORIES and item.points != 0:
                raise ValueError("non-risk language categories must have zero points")
        return self


class HumanReviewRules(RiskRuleRecord):
    minimum_score: float = Field(gt=0, le=100)
    transcript_review_required: bool = True
    ambiguous_language: bool = True


class RiskRuleSet(RiskRuleRecord):
    """Complete score weights, caps, and human-review triggers for one policy."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/risk-rule-set.schema.json"
        },
    )

    schema_version: Literal["1.0"] = RISK_RULE_SET_SCHEMA_VERSION
    rule_format_version: Literal["1.0"] = RISK_RULE_FORMAT_VERSION
    rule_set_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    rule_set_version: str = Field(min_length=1, max_length=128)
    acoustic: AcousticScoringRules
    speech: SpeechScoringRules
    language: LanguageScoringRules
    human_review: HumanReviewRules
    total_max_score: Literal[100] = 100


@dataclass(frozen=True)
class LoadedRiskRuleSet:
    """Validated rule data plus its exact artifact digest."""

    rule_set: RiskRuleSet
    artifact_sha256: str
    artifact_size_bytes: int

    def as_descriptor(self) -> RiskRuleSetDescriptor:
        return RiskRuleSetDescriptor(
            rule_set_id=self.rule_set.rule_set_id,
            rule_set_version=self.rule_set.rule_set_version,
            rule_format_version=self.rule_set.rule_format_version,
            artifact_sha256=self.artifact_sha256,
        )


def load_risk_rule_set_bytes(
    raw: bytes,
    *,
    expected_sha256: str | None = None,
) -> LoadedRiskRuleSet:
    """Validate bounded JSON rule bytes and optionally pin their exact digest."""

    if not isinstance(raw, bytes):
        raise TypeError("risk rule artifact must be bytes")
    if not raw:
        raise ValueError("risk rule artifact must not be empty")
    if len(raw) > MAX_RISK_RULE_SET_BYTES:
        raise ValueError("risk rule artifact exceeds the byte limit")
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and artifact_sha256 != expected_sha256:
        raise ValueError("risk rule artifact checksum mismatch")
    try:
        rule_set = RiskRuleSet.model_validate_json(raw)
    except Exception as error:
        raise ValueError("risk rule artifact failed contract validation") from error
    return LoadedRiskRuleSet(
        rule_set=rule_set,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=len(raw),
    )


def load_builtin_risk_rule_set() -> LoadedRiskRuleSet:
    """Load the pinned local v1 scoring rules without network access."""

    raw = (
        files("audio_sentinel")
        .joinpath(f"resources/{BUILTIN_RISK_RULE_RESOURCE}")
        .read_bytes()
    )
    loaded = load_risk_rule_set_bytes(
        raw,
        expected_sha256=BUILTIN_RISK_RULE_SET_SHA256,
    )
    if (
        loaded.rule_set.rule_set_id != BUILTIN_RISK_RULE_SET_ID
        or loaded.rule_set.rule_set_version != BUILTIN_RISK_RULE_SET_VERSION
    ):
        raise ValueError("bundled risk rule identity differs from the pinned specification")
    return loaded


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise RiskScoringError(
            "invalid_time", "Risk assessment time must be timezone-aware."
        )
    return value.astimezone(UTC)


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _validate_inputs(inputs: RiskInputSet) -> RiskInputSet:
    if not isinstance(inputs, RiskInputSet):
        raise RiskScoringError("invalid_inputs", "A6.1 risk inputs are required.")
    try:
        validated = RiskInputSet.model_validate(inputs.model_dump(mode="python"))
    except Exception as error:
        raise RiskScoringError(
            "invalid_inputs", "Risk inputs failed contract validation."
        ) from error
    if validated != inputs:
        raise RiskScoringError("invalid_inputs", "Risk input integrity validation failed.")
    return validated


def _validate_loaded_rules(loaded: LoadedRiskRuleSet) -> LoadedRiskRuleSet:
    if not isinstance(loaded, LoadedRiskRuleSet):
        raise ValueError("loaded risk rule set has an invalid type")
    validated = RiskRuleSet.model_validate(loaded.rule_set.model_dump(mode="python"))
    if (
        validated != loaded.rule_set
        or not re.fullmatch(r"[a-f0-9]{64}", loaded.artifact_sha256)
        or not isinstance(loaded.artifact_size_bytes, int)
        or isinstance(loaded.artifact_size_bytes, bool)
        or loaded.artifact_size_bytes <= 0
        or loaded.artifact_size_bytes > MAX_RISK_RULE_SET_BYTES
    ):
        raise ValueError("loaded risk rule set failed integrity validation")
    return LoadedRiskRuleSet(
        rule_set=validated,
        artifact_sha256=loaded.artifact_sha256,
        artifact_size_bytes=loaded.artifact_size_bytes,
    )


def _acoustic_score(
    inputs: AcousticRiskInputs,
    rules: AcousticScoringRules,
) -> tuple[float, set[RiskReasonCode]]:
    if inputs.status is not RiskInputStatus.PRESENT:
        return 0.0, set()
    weights = {item.label: item.points for item in rules.label_scores}
    weighted = [(signal, weights[signal.label]) for signal in inputs.signals]
    positive = [(signal, points) for signal, points in weighted if points > 0]
    if not positive:
        return 0.0, set()
    reasons = {RiskReasonCode.ACOUSTIC_EVENT_CANDIDATE}
    points = max(value for _, value in positive)
    if any(
        signal.peak_score >= rules.high_confidence_threshold
        for signal, _ in positive
    ):
        points += rules.high_confidence_bonus
        reasons.add(RiskReasonCode.ACOUSTIC_HIGH_CONFIDENCE)
    return min(points, rules.max_points), reasons


def _speech_score(
    inputs: SpeechRiskInputs,
    rules: SpeechScoringRules,
) -> tuple[float, set[RiskReasonCode]]:
    if inputs.status is not RiskInputStatus.PRESENT:
        return 0.0, set()
    points = 0.0
    reasons: set[RiskReasonCode] = set()
    if inputs.segment_count > 0:
        points += rules.segment_presence_points
        reasons.add(RiskReasonCode.SPEECH_PRESENT)
    if inputs.review_required_transcript_count > 0:
        if rules.review_required_points_per_transcript == 0:
            review_points = 0.0
        else:
            weight_numerator, weight_denominator = (
                rules.review_required_points_per_transcript.as_integer_ratio()
            )
            cap_numerator, cap_denominator = (
                rules.max_review_required_points.as_integer_ratio()
            )
            weighted_numerator = (
                inputs.review_required_transcript_count * weight_numerator
            )
            review_points = (
                rules.max_review_required_points
                if weighted_numerator * cap_denominator
                >= cap_numerator * weight_denominator
                else weighted_numerator / weight_denominator
            )
        points += review_points
        reasons.add(RiskReasonCode.TRANSCRIPT_REVIEW_REQUIRED)
    return min(points, rules.max_points), reasons


def _language_score(
    inputs: LanguageRiskInputs,
    rules: LanguageScoringRules,
) -> tuple[float, set[RiskReasonCode]]:
    if inputs.status is not RiskInputStatus.PRESENT:
        return 0.0, set()
    weights = {item.category: item.points for item in rules.category_scores}
    points = sum(weights[signal.category] for signal in inputs.signals)
    reasons = {
        _LANGUAGE_REASON[signal.category]
        for signal in inputs.signals
        if signal.category in _LANGUAGE_REASON
    }
    return min(points, rules.max_points), reasons


def _branch_statuses(
    inputs: RiskInputSet,
) -> tuple[tuple[RiskEvidenceKind, RiskInputStatus], ...]:
    return (
        (RiskEvidenceKind.ACOUSTIC, inputs.acoustic.status),
        (RiskEvidenceKind.SPEECH, inputs.speech.status),
        (RiskEvidenceKind.LANGUAGE, inputs.language.status),
    )


def _missing_data(
    inputs: RiskInputSet,
) -> tuple[RiskMissingDataAssessment, set[RiskReasonCode]]:
    statuses = _branch_statuses(inputs)
    missing = tuple(kind for kind, status in statuses if status is RiskInputStatus.MISSING)
    not_permitted = tuple(
        kind for kind, status in statuses if status is RiskInputStatus.NOT_PERMITTED
    )
    reasons = {_MISSING_REASON[kind] for kind in missing}
    if inputs.speech.status is RiskInputStatus.NOT_PERMITTED:
        reasons.add(RiskReasonCode.SPEECH_NOT_PERMITTED)
    if inputs.language.status in {
        RiskInputStatus.NOT_PERMITTED,
        RiskInputStatus.NOT_APPLICABLE,
        RiskInputStatus.NO_ACCEPTED_TEXT,
    }:
        reasons.add(RiskReasonCode.LANGUAGE_NOT_APPLICABLE)
    missing_reasons = set(_MISSING_REASON[kind] for kind in missing)
    review_required = bool(missing)
    if review_required:
        missing_reasons.add(RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED)
        reasons.add(RiskReasonCode.MISSING_DATA_REVIEW_REQUIRED)
    ordered_missing_reasons = tuple(
        sorted(missing_reasons, key=lambda reason: _REASON_ORDER[reason])
    )
    assessment = RiskMissingDataAssessment(
        missing_branches=tuple(sorted(missing, key=lambda kind: _KIND_ORDER[kind])),
        not_permitted_branches=tuple(
            sorted(not_permitted, key=lambda kind: _KIND_ORDER[kind])
        ),
        review_required=review_required,
        reason_codes=ordered_missing_reasons,
    )
    return assessment, reasons


def score_risk(
    inputs: RiskInputSet,
    *,
    rule_set: LoadedRiskRuleSet | None = None,
    now: datetime | None = None,
) -> RiskAssessmentDocument:
    """Score validated A6.1 inputs with deterministic, bounded local rules."""

    created_at = _clock(now)
    validated_inputs = _validate_inputs(inputs)
    try:
        loaded = _validate_loaded_rules(rule_set or load_builtin_risk_rule_set())
    except Exception as error:
        raise RiskScoringError(
            "invalid_rule_set", "Risk-scoring rule data failed integrity validation."
        ) from error
    rules = loaded.rule_set

    acoustic_points, acoustic_reasons = _acoustic_score(
        validated_inputs.acoustic, rules.acoustic
    )
    speech_points, speech_reasons = _speech_score(
        validated_inputs.speech, rules.speech
    )
    language_points, language_reasons = _language_score(
        validated_inputs.language, rules.language
    )
    missing_data, status_reasons = _missing_data(validated_inputs)
    score = min(
        rules.total_max_score,
        acoustic_points + speech_points + language_points,
    )
    if not math.isfinite(score):
        raise RiskScoringError("invalid_score", "Risk rules produced a non-finite score.")
    score = round(score, 6)

    reasons = acoustic_reasons | speech_reasons | language_reasons | status_reasons
    if score == 0:
        reasons.add(RiskReasonCode.NO_RISK_EVIDENCE)
    ordered_reasons = tuple(sorted(reasons, key=lambda reason: _REASON_ORDER[reason]))
    human_review_required = (
        missing_data.review_required
        or score >= rules.human_review.minimum_score
        or (
            rules.human_review.transcript_review_required
            and validated_inputs.speech.review_required_transcript_count > 0
        )
        or (
            rules.human_review.ambiguous_language
            and any(
                signal.category is LanguageCategory.AMBIGUOUS
                for signal in validated_inputs.language.signals
            )
        )
    )
    scoring_policy = RiskScoringPolicy(rule_set=loaded.as_descriptor())
    identity_payload = {
        "scoring_policy": scoring_policy.model_dump(mode="json"),
        "inputs": validated_inputs.model_dump(mode="json"),
        "score": score,
        "severity": severity_for_score(score).value,
        "reason_codes": [reason.value for reason in ordered_reasons],
        "missing_data": missing_data.model_dump(mode="json"),
        "human_review_required": human_review_required,
    }
    return RiskAssessmentDocument(
        assessment_id=_canonical_hash(identity_payload),
        created_at=created_at,
        scoring_policy=scoring_policy,
        inputs=validated_inputs,
        score=score,
        severity=severity_for_score(score),
        reason_codes=ordered_reasons,
        missing_data=missing_data,
        human_review_required=human_review_required,
    )


def risk_rule_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public JSON Schema for configurable v1 scoring rules."""

    return {"risk-rule-set.schema.json": RiskRuleSet.model_json_schema()}


def write_risk_rule_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the portable risk-rule schema for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in risk_rule_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
