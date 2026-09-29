"""B9.1 repeatable collection evaluation and deterministic metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.consensus_contracts import ConsensusOutcome
from audio_sentinel.contracts import RiskLevel
from audio_sentinel.evaluation_service import (
    EvaluationRequest,
    EvaluationResponse,
    run_evaluation,
)
from audio_sentinel.preparation import Identifier


EVALUATION_MANIFEST_SCHEMA_VERSION = "1.0"
EVALUATION_RUN_SCHEMA_VERSION = "1.0"
MAX_EVALUATION_MANIFEST_BYTES = 2_097_152
MAX_EVALUATION_RUN_BYTES = 16_777_216
MAX_EVALUATION_CASES = 10_000
_CANONICAL_OUTCOMES = tuple(ConsensusOutcome)
_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,64}$")


class EvaluationManifestError(RuntimeError):
    """Stable collection-evaluation failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class EvaluationManifestCase(EvaluationRecord):
    """One authorized input and its human-supplied evaluation truth."""

    case_id: Identifier
    request: EvaluationRequest
    expected_positive: StrictBool
    expected_outcome: ConsensusOutcome | None = None
    categories: tuple[Identifier, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_case(self) -> "EvaluationManifestCase":
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("evaluation categories must be unique within a case")
        return self


class EvaluationManifestDocument(EvaluationRecord):
    """Content-addressed list of authorized clips and expected results."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/evaluation-manifest.schema.json"
        },
    )

    schema_version: Literal["1.0"] = EVALUATION_MANIFEST_SCHEMA_VERSION
    document_type: Literal["evaluation_manifest"] = "evaluation_manifest"
    manifest_id: Identifier
    created_at: datetime
    name: str = Field(min_length=1, max_length=128)
    positive_outcomes: tuple[ConsensusOutcome, ...] = (
        ConsensusOutcome.REVIEW,
        ConsensusOutcome.ALERT,
    )
    cases: tuple[EvaluationManifestCase, ...] = Field(
        min_length=1,
        max_length=MAX_EVALUATION_CASES,
    )

    @model_validator(mode="after")
    def validate_manifest(self) -> "EvaluationManifestDocument":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if not self.positive_outcomes:
            raise ValueError("positive_outcomes must not be empty")
        expected_order = tuple(
            outcome
            for outcome in _CANONICAL_OUTCOMES
            if outcome in self.positive_outcomes
        )
        if self.positive_outcomes != expected_order:
            raise ValueError(
                "positive_outcomes must be unique and use canonical outcome order"
            )
        case_ids = tuple(case.case_id for case in self.cases)
        clip_ids = tuple(case.request.clip_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("evaluation case IDs must be unique")
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("evaluation clip IDs must be unique")
        for case in self.cases:
            if case.expected_outcome is not None and (
                case.expected_positive
                is not (case.expected_outcome in self.positive_outcomes)
            ):
                raise ValueError(
                    "expected_positive must match expected_outcome and positive_outcomes"
                )
        if self.manifest_id != _manifest_id(self):
            raise ValueError("manifest_id must match canonical manifest content")
        return self


@dataclass(frozen=True)
class LoadedEvaluationManifest:
    manifest: EvaluationManifestDocument
    artifact_sha256: str
    artifact_size_bytes: int


class EvaluationCaseResult(EvaluationRecord):
    """Privacy-minimized case result; it deliberately omits source paths."""

    case_id: Identifier
    clip_id: Identifier
    categories: tuple[Identifier, ...]
    expected_positive: bool
    expected_outcome: ConsensusOutcome | None = None
    status: Literal["completed", "failed"]
    observed_positive: bool | None = None
    observed_outcome: ConsensusOutcome | None = None
    risk_score: float | None = Field(default=None, ge=0, le=100)
    risk_severity: RiskLevel | None = None
    report_id: Identifier | None = None
    audit_id: Identifier | None = None
    alert_id: Identifier | None = None
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]{1,64}$",
    )

    @model_validator(mode="after")
    def validate_result(self) -> "EvaluationCaseResult":
        result_fields = (
            self.observed_positive,
            self.observed_outcome,
            self.risk_score,
            self.risk_severity,
            self.report_id,
            self.audit_id,
        )
        if self.status == "completed":
            if any(value is None for value in result_fields) or self.error_code is not None:
                raise ValueError("completed cases require result fields and no error")
            if (self.alert_id is not None) is not (
                self.observed_outcome is ConsensusOutcome.ALERT
            ):
                raise ValueError("alert identity must match the observed alert outcome")
        elif any(value is not None for value in (*result_fields, self.alert_id)):
            raise ValueError("failed cases cannot contain observed result fields")
        elif self.error_code is None:
            raise ValueError("failed cases require a stable error code")
        return self


class BinaryEvaluationMetrics(EvaluationRecord):
    evaluated_count: int = Field(ge=0, strict=True)
    positive_count: int = Field(ge=0, strict=True)
    negative_count: int = Field(ge=0, strict=True)
    true_positive: int = Field(ge=0, strict=True)
    false_positive: int = Field(ge=0, strict=True)
    false_negative: int = Field(ge=0, strict=True)
    true_negative: int = Field(ge=0, strict=True)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    specificity: float | None = Field(default=None, ge=0, le=1)
    f1: float | None = Field(default=None, ge=0, le=1)
    balanced_accuracy: float | None = Field(default=None, ge=0, le=1)
    false_positive_rate: float | None = Field(default=None, ge=0, le=1)
    false_negative_rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self) -> "BinaryEvaluationMetrics":
        if self.positive_count != self.true_positive + self.false_negative:
            raise ValueError("positive_count must equal true positives plus false negatives")
        if self.negative_count != self.false_positive + self.true_negative:
            raise ValueError("negative_count must equal false positives plus true negatives")
        if self.evaluated_count != self.positive_count + self.negative_count:
            raise ValueError("evaluated_count must equal positive plus negative counts")
        return self


class EvaluationMetrics(EvaluationRecord):
    total_case_count: int = Field(ge=1, strict=True)
    completed_case_count: int = Field(ge=0, strict=True)
    failed_case_count: int = Field(ge=0, strict=True)
    completion_rate: float = Field(ge=0, le=1)
    binary: BinaryEvaluationMetrics
    outcome_labeled_count: int = Field(ge=0, strict=True)
    outcome_match_count: int = Field(ge=0, strict=True)
    outcome_accuracy: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_totals(self) -> "EvaluationMetrics":
        if self.total_case_count != self.completed_case_count + self.failed_case_count:
            raise ValueError("total cases must equal completed plus failed cases")
        if self.binary.evaluated_count != self.completed_case_count:
            raise ValueError("binary metrics must cover every completed case")
        if self.outcome_match_count > self.outcome_labeled_count:
            raise ValueError("outcome matches cannot exceed labeled outcomes")
        if self.outcome_labeled_count > self.completed_case_count:
            raise ValueError("outcome labels cannot exceed completed cases")
        return self


class EvaluationRunDocument(EvaluationRecord):
    """Deterministic metric report with explicit operational coverage."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/evaluation-run.schema.json"
        },
    )

    schema_version: Literal["1.0"] = EVALUATION_RUN_SCHEMA_VERSION
    document_type: Literal["evaluation_run"] = "evaluation_run"
    run_id: Identifier
    created_at: datetime
    manifest_id: Identifier
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    manifest_size_bytes: int = Field(gt=0, strict=True)
    positive_outcomes: tuple[ConsensusOutcome, ...]
    decision_status: Literal["complete", "incomplete_due_to_failures"]
    metrics: EvaluationMetrics
    cases: tuple[EvaluationCaseResult, ...] = Field(
        min_length=1,
        max_length=MAX_EVALUATION_CASES,
    )
    notification_delivery: Literal["not_sent"] = "not_sent"
    alert_delivery_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_run(self) -> "EvaluationRunDocument":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        expected_order = tuple(
            outcome
            for outcome in _CANONICAL_OUTCOMES
            if outcome in self.positive_outcomes
        )
        if not self.positive_outcomes or self.positive_outcomes != expected_order:
            raise ValueError(
                "positive_outcomes must be unique and use canonical outcome order"
            )
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("evaluation run case IDs must be unique")
        if any(
            case.status == "completed"
            and case.observed_positive is not (
                case.observed_outcome in self.positive_outcomes
            )
            for case in self.cases
        ):
            raise ValueError("observed_positive must follow positive_outcomes")
        if self.metrics != calculate_evaluation_metrics(self.cases):
            raise ValueError("evaluation metrics must match case results")
        expected_status = (
            "complete"
            if self.metrics.failed_case_count == 0
            else "incomplete_due_to_failures"
        )
        if self.decision_status != expected_status:
            raise ValueError("decision_status must reflect failed evaluation cases")
        if self.run_id != _run_id(self):
            raise ValueError("run_id must match canonical evaluation content")
        return self


EvaluationRunner = Callable[
    [AudioSentinelSettings, EvaluationRequest],
    EvaluationResponse,
]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_id(manifest: EvaluationManifestDocument) -> str:
    payload = manifest.model_dump(
        mode="json",
        exclude={"manifest_id", "created_at"},
    )
    return f"evaluation-manifest-{_sha256(_canonical_bytes(payload))[:24]}"


def _run_id(report: EvaluationRunDocument) -> str:
    payload = report.model_dump(mode="json", exclude={"run_id", "created_at"})
    return f"evaluation-run-{_sha256(_canonical_bytes(payload))[:24]}"


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise EvaluationManifestError(
            "invalid_time",
            "Evaluation timestamps must include a UTC offset.",
        )
    return value.astimezone(UTC)


def build_evaluation_manifest(
    name: str,
    cases: Sequence[EvaluationManifestCase],
    *,
    positive_outcomes: Sequence[ConsensusOutcome] = (
        ConsensusOutcome.REVIEW,
        ConsensusOutcome.ALERT,
    ),
    now: datetime | None = None,
) -> EvaluationManifestDocument:
    """Build a validated content-addressed manifest from explicit case truth."""

    payload = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "document_type": "evaluation_manifest",
        "created_at": _clock(now),
        "name": name,
        "positive_outcomes": tuple(positive_outcomes),
        "cases": tuple(cases),
    }
    identity_payload = {
        key: value for key, value in payload.items() if key != "created_at"
    }
    serialized = {
        "schema_version": identity_payload["schema_version"],
        "document_type": identity_payload["document_type"],
        "name": identity_payload["name"],
        "positive_outcomes": [
            outcome.value for outcome in identity_payload["positive_outcomes"]
        ],
        "cases": [case.model_dump(mode="json") for case in identity_payload["cases"]],
    }
    payload["manifest_id"] = (
        f"evaluation-manifest-{_sha256(_canonical_bytes(serialized))[:24]}"
    )
    return EvaluationManifestDocument.model_validate(payload)


def load_evaluation_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
    maximum_bytes: int = MAX_EVALUATION_MANIFEST_BYTES,
) -> LoadedEvaluationManifest:
    """Load a bounded regular JSON manifest and verify its optional checksum."""

    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise EvaluationManifestError(
                "invalid_manifest_path",
                "The evaluation manifest must be a regular local file.",
            )
        size = source.stat().st_size
        if size <= 0 or size > maximum_bytes:
            raise EvaluationManifestError(
                "invalid_manifest_size",
                "The evaluation manifest has an invalid size.",
            )
        document = source.read_bytes()
    except EvaluationManifestError:
        raise
    except OSError as error:
        raise EvaluationManifestError(
            "manifest_unreadable",
            "The evaluation manifest could not be read.",
        ) from error
    if len(document) != size:
        raise EvaluationManifestError(
            "manifest_changed",
            "The evaluation manifest changed while it was being read.",
        )
    digest = _sha256(document)
    if expected_sha256 is not None and digest != expected_sha256:
        raise EvaluationManifestError(
            "manifest_checksum_mismatch",
            "The evaluation manifest checksum does not match the expected value.",
        )
    try:
        manifest = EvaluationManifestDocument.model_validate_json(document)
    except (ValueError, TypeError) as error:
        raise EvaluationManifestError(
            "invalid_manifest",
            "The evaluation manifest failed contract validation.",
        ) from error
    return LoadedEvaluationManifest(manifest, digest, size)


def _divide(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def calculate_evaluation_metrics(
    cases: Sequence[EvaluationCaseResult],
) -> EvaluationMetrics:
    """Calculate binary and exact-outcome metrics over completed cases only."""

    if not cases:
        raise ValueError("evaluation metrics require at least one case")
    completed = tuple(case for case in cases if case.status == "completed")
    failed_count = len(cases) - len(completed)
    tp = sum(case.expected_positive and case.observed_positive is True for case in completed)
    fp = sum(not case.expected_positive and case.observed_positive is True for case in completed)
    fn = sum(case.expected_positive and case.observed_positive is False for case in completed)
    tn = sum(not case.expected_positive and case.observed_positive is False for case in completed)
    positive_count = tp + fn
    negative_count = fp + tn
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)
    balanced = (
        None
        if recall is None or specificity is None
        else (recall + specificity) / 2
    )
    labeled = tuple(case for case in completed if case.expected_outcome is not None)
    matches = sum(case.expected_outcome is case.observed_outcome for case in labeled)
    binary = BinaryEvaluationMetrics(
        evaluated_count=len(completed),
        positive_count=positive_count,
        negative_count=negative_count,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        accuracy=_divide(tp + tn, len(completed)),
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        balanced_accuracy=balanced,
        false_positive_rate=_divide(fp, fp + tn),
        false_negative_rate=_divide(fn, fn + tp),
    )
    return EvaluationMetrics(
        total_case_count=len(cases),
        completed_case_count=len(completed),
        failed_case_count=failed_count,
        completion_rate=len(completed) / len(cases),
        binary=binary,
        outcome_labeled_count=len(labeled),
        outcome_match_count=matches,
        outcome_accuracy=_divide(matches, len(labeled)),
    )


def _safe_error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    return code if isinstance(code, str) and _SAFE_ERROR_CODE.fullmatch(code) else "unexpected_error"


def _completed_case(
    case: EvaluationManifestCase,
    response: EvaluationResponse,
    positive_outcomes: tuple[ConsensusOutcome, ...],
) -> EvaluationCaseResult:
    outcome = ConsensusOutcome(response.outcome)
    return EvaluationCaseResult(
        case_id=case.case_id,
        clip_id=case.request.clip_id,
        categories=case.categories,
        expected_positive=case.expected_positive,
        expected_outcome=case.expected_outcome,
        status="completed",
        observed_positive=outcome in positive_outcomes,
        observed_outcome=outcome,
        risk_score=response.risk_score,
        risk_severity=RiskLevel(response.risk_severity),
        report_id=response.report_id,
        audit_id=response.audit_id,
        alert_id=response.alert_id,
    )


def run_evaluation_manifest(
    settings: AudioSentinelSettings,
    loaded: LoadedEvaluationManifest,
    *,
    runner: EvaluationRunner = run_evaluation,
    now: datetime | None = None,
) -> EvaluationRunDocument:
    """Evaluate every case through the shared Phase 8 application boundary."""

    if not isinstance(settings, AudioSentinelSettings) or not isinstance(
        loaded, LoadedEvaluationManifest
    ):
        raise EvaluationManifestError(
            "invalid_evaluation_input",
            "Validated settings and an evaluation manifest are required.",
        )
    results: list[EvaluationCaseResult] = []
    manifest = loaded.manifest
    for case in manifest.cases:
        try:
            response = EvaluationResponse.model_validate(runner(settings, case.request))
            results.append(
                _completed_case(case, response, manifest.positive_outcomes)
            )
        except Exception as error:  # one failed clip must remain visible in coverage
            results.append(
                EvaluationCaseResult(
                    case_id=case.case_id,
                    clip_id=case.request.clip_id,
                    categories=case.categories,
                    expected_positive=case.expected_positive,
                    expected_outcome=case.expected_outcome,
                    status="failed",
                    error_code=_safe_error_code(error),
                )
            )
    result_tuple = tuple(results)
    metrics = calculate_evaluation_metrics(result_tuple)
    payload = {
        "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
        "document_type": "evaluation_run",
        "created_at": _clock(now),
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": loaded.artifact_sha256,
        "manifest_size_bytes": loaded.artifact_size_bytes,
        "positive_outcomes": manifest.positive_outcomes,
        "decision_status": (
            "complete" if metrics.failed_case_count == 0 else "incomplete_due_to_failures"
        ),
        "metrics": metrics,
        "cases": result_tuple,
        "notification_delivery": "not_sent",
        "alert_delivery_authorized": False,
    }
    serialized = {
        key: (
            value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item.value
                if isinstance(item, ConsensusOutcome)
                else item
                for item in value
            ]
            if isinstance(value, tuple)
            else value
        )
        for key, value in payload.items()
        if key != "created_at"
    }
    payload["run_id"] = f"evaluation-run-{_sha256(_canonical_bytes(serialized))[:24]}"
    return EvaluationRunDocument.model_validate(payload)


def save_evaluation_run(report: EvaluationRunDocument, destination: Path) -> bool:
    """Write one run report or reuse a semantically identical prior result.

    Returns ``True`` only when an existing result was safely reused.
    """

    if not isinstance(report, EvaluationRunDocument):
        raise EvaluationManifestError(
            "invalid_evaluation_run",
            "A validated evaluation run is required.",
        )
    output = Path(destination)
    document = (
        json.dumps(
            report.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(document)
            stream.flush()
    except FileExistsError as error:
        try:
            existing_size = output.stat().st_size
            if existing_size <= 0 or existing_size > MAX_EVALUATION_RUN_BYTES:
                raise ValueError("existing evaluation output has an invalid size")
            existing_bytes = output.read_bytes()
            if len(existing_bytes) != existing_size:
                raise ValueError("existing evaluation output changed while reading")
            existing = EvaluationRunDocument.model_validate_json(existing_bytes)
        except (OSError, ValueError, TypeError) as load_error:
            raise EvaluationManifestError(
                "output_conflict",
                "The evaluation output exists but is not the same verified run.",
            ) from load_error
        existing_semantic = existing.model_dump(
            mode="json", exclude={"created_at"}
        )
        requested_semantic = report.model_dump(
            mode="json", exclude={"created_at"}
        )
        if existing_semantic == requested_semantic:
            return True
        raise EvaluationManifestError(
            "output_conflict",
            "The evaluation output exists but is not the same verified run.",
        ) from error
    except OSError as error:
        raise EvaluationManifestError(
            "output_unwritable",
            "The evaluation output could not be written.",
        ) from error
    return False


def evaluation_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public B9.1 JSON Schemas."""

    return {
        "evaluation-manifest.schema.json": EvaluationManifestDocument.model_json_schema(),
        "evaluation-run.schema.json": EvaluationRunDocument.model_json_schema(),
    }


def write_evaluation_schemas(output_directory: Path) -> dict[str, Path]:
    """Write the public B9.1 JSON Schemas for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for filename, document in evaluation_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written[filename] = destination
    return written
