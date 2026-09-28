"""B8.1 versioned final JSON report contract and verified local persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import errno
import hashlib
import json
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.acoustic_evidence import (
    AcousticEvidenceDocument,
    AcousticEvidenceError,
    load_acoustic_evidence,
)
from audio_sentinel.config import Paths
from audio_sentinel.consensus_contracts import (
    ConsensusDecisionDocument,
    ConsensusOutcome,
    ConsensusRiskReference,
)
from audio_sentinel.consensus_rules import (
    ConsensusAgreementEvaluation,
    risk_assessment_sha256,
)
from audio_sentinel.contracts import ProcessingScope, RiskLevel
from audio_sentinel.evaluator import OfflineEvaluationResult
from audio_sentinel.feature_persistence import _read_bytes, _safe_file
from audio_sentinel.language_contracts import LanguageEvidenceDocument
from audio_sentinel.persistence import _is_link, _remove_stage
from audio_sentinel.preparation import Identifier, PreparedAudioManifest
from audio_sentinel.risk_contracts import (
    RiskAssessmentDocument,
    RiskEvidenceKind,
    RiskEvidenceReference,
    RiskInputStatus,
)
from audio_sentinel.speech_contracts import SpeechEvidenceDocument


FINAL_REPORT_SCHEMA_VERSION = "1.0"
FINAL_REPORT_FORMAT_VERSION = "1.0"
FINAL_REPORT_DIRECTORY = "final-reports"
FINAL_REPORT_FILENAME = "report.json"

_DOCUMENT_TYPE_BY_KIND = {
    RiskEvidenceKind.ACOUSTIC: "acoustic_event_candidates",
    RiskEvidenceKind.SPEECH: "speech_evidence_candidates",
    RiskEvidenceKind.LANGUAGE: "language_evidence",
}


class FinalReportError(RuntimeError):
    """Stable report failure code plus a safe, non-sensitive message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class FinalReportRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class FinalReportSource(FinalReportRecord):
    """Privacy-minimized source identity with content provenance, never a path."""

    clip_id: Identifier
    consent_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    processing_scope: ProcessingScope
    sample_rate_hz: int = Field(ge=8_000, le=192_000, strict=True)
    num_samples: int = Field(gt=0, strict=True)
    raw_audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    preparation_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_scope(self) -> "FinalReportSource":
        if self.processing_scope is ProcessingScope.NONE:
            raise ValueError("final reports require an authorized processing scope")
        return self


class FinalReportEvidenceReceipt(FinalReportRecord):
    """Hash-pinned evidence receipt without text, audio, tensors, or local paths."""

    kind: RiskEvidenceKind
    input_status: RiskInputStatus
    document_type: Literal[
        "acoustic_event_candidates",
        "speech_evidence_candidates",
        "language_evidence",
    ]
    evidence_id: Identifier | None = None
    evidence_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )

    @model_validator(mode="after")
    def validate_receipt(self) -> "FinalReportEvidenceReceipt":
        if self.document_type != _DOCUMENT_TYPE_BY_KIND[self.kind]:
            raise ValueError("evidence kind does not match document_type")
        has_id = self.evidence_id is not None
        has_hash = self.evidence_sha256 is not None
        if has_id != has_hash:
            raise ValueError("evidence identity and hash must be present together")
        if self.input_status in {
            RiskInputStatus.PRESENT,
            RiskInputStatus.NO_ACCEPTED_TEXT,
        } and not has_id:
            raise ValueError("completed evidence status requires a hash-pinned artifact")
        if self.input_status in {
            RiskInputStatus.NOT_PERMITTED,
            RiskInputStatus.MISSING,
            RiskInputStatus.NOT_APPLICABLE,
        } and has_id:
            raise ValueError("unavailable evidence status cannot reference an artifact")
        return self


class FinalReportSummary(FinalReportRecord):
    """Small inspection view derived entirely from the embedded trusted documents."""

    acoustic_event_count: int = Field(ge=0, strict=True)
    acoustic_max_peak_score: float | None = Field(default=None, ge=0, le=1)
    speech_segment_count: int | None = Field(default=None, ge=0, strict=True)
    speech_accepted_transcript_count: int | None = Field(
        default=None, ge=0, strict=True
    )
    speech_review_required_transcript_count: int | None = Field(
        default=None, ge=0, strict=True
    )
    language_finding_count: int | None = Field(default=None, ge=0, strict=True)
    risk_score: float = Field(ge=0, le=100)
    risk_severity: RiskLevel
    outcome: ConsensusOutcome
    review_required: bool
    alert_candidate: bool
    notification_delivery: Literal["not_sent"] = "not_sent"
    alert_delivery_authorized: Literal[False] = False


class FinalReportDocument(FinalReportRecord):
    """Portable final evaluation receipt; never notification authorization."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/final-report.schema.json"
        },
    )

    schema_version: Literal["1.0"] = FINAL_REPORT_SCHEMA_VERSION
    document_type: Literal["offline_evaluation_report"] = (
        "offline_evaluation_report"
    )
    format_version: Literal["1.0"] = FINAL_REPORT_FORMAT_VERSION
    report_id: Identifier
    created_at: datetime
    source: FinalReportSource
    evidence: tuple[FinalReportEvidenceReceipt, ...] = Field(
        min_length=3, max_length=3
    )
    summary: FinalReportSummary
    risk_assessment: RiskAssessmentDocument
    agreement: ConsensusAgreementEvaluation
    decision: ConsensusDecisionDocument

    @model_validator(mode="after")
    def validate_document(self) -> "FinalReportDocument":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if tuple(item.kind for item in self.evidence) != tuple(RiskEvidenceKind):
            raise ValueError("evidence receipts must use complete canonical ordering")
        _validate_risk_assessment_identity(self.risk_assessment)
        _validate_report_alignment(self)
        if self.report_id != _report_id(self):
            raise ValueError("report_id must match canonical report content")
        return self


class FinalReportPersistencePolicy(FinalReportRecord):
    """Resource limits for one final report JSON file."""

    max_document_bytes: int = Field(default=16_777_216, gt=0, strict=True)


@dataclass(frozen=True)
class SavedFinalReport:
    directory: Path
    report_path: Path
    report: FinalReportDocument
    reused: bool


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise FinalReportError(
            "invalid_time", "Final report creation time must be timezone-aware."
        )
    return value.astimezone(UTC)


def _canonical_hash(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _artifact_hash(document: BaseModel) -> str:
    return _canonical_hash(document.model_dump(mode="json"))


def _risk_assessment_identity(assessment: RiskAssessmentDocument) -> str:
    payload = {
        "scoring_policy": assessment.scoring_policy.model_dump(mode="json"),
        "inputs": assessment.inputs.model_dump(mode="json"),
        "score": assessment.score,
        "severity": assessment.severity.value,
        "reason_codes": [reason.value for reason in assessment.reason_codes],
        "missing_data": assessment.missing_data.model_dump(mode="json"),
        "human_review_required": assessment.human_review_required,
    }
    return _canonical_hash(payload)


def _validate_risk_assessment_identity(assessment: RiskAssessmentDocument) -> None:
    if assessment.assessment_id != _risk_assessment_identity(assessment):
        raise ValueError("assessment_id must match canonical risk content")


def _decision_identity(decision: ConsensusDecisionDocument) -> str:
    payload = {
        "policy": decision.policy.model_dump(mode="json"),
        "risk_assessment": decision.risk_assessment.model_dump(mode="json"),
        "branch_states": [
            state.model_dump(mode="json") for state in decision.branch_states
        ],
        "outcome": decision.outcome.value,
        "reason_codes": [reason.value for reason in decision.reason_codes],
    }
    return f"decision-{_canonical_hash(payload)}"


def _report_id(report: FinalReportDocument) -> str:
    payload = report.model_dump(mode="json", exclude={"report_id", "created_at"})
    return f"report-{_canonical_hash(payload)}"


def _expected_risk_reference(
    assessment: RiskAssessmentDocument,
) -> ConsensusRiskReference:
    return ConsensusRiskReference(
        assessment_id=assessment.assessment_id,
        assessment_sha256=risk_assessment_sha256(assessment),
        clip_id=assessment.inputs.source.clip_id,
        score=assessment.score,
        severity=assessment.severity,
        human_review_required=assessment.human_review_required,
        reason_codes=assessment.reason_codes,
        missing_branches=assessment.missing_data.missing_branches,
        not_permitted_branches=assessment.missing_data.not_permitted_branches,
    )


def _branch_input(assessment: RiskAssessmentDocument, kind: RiskEvidenceKind):
    return {
        RiskEvidenceKind.ACOUSTIC: assessment.inputs.acoustic,
        RiskEvidenceKind.SPEECH: assessment.inputs.speech,
        RiskEvidenceKind.LANGUAGE: assessment.inputs.language,
    }[kind]


def _expected_summary(report: FinalReportDocument) -> FinalReportSummary:
    inputs = report.risk_assessment.inputs
    speech_permitted = inputs.speech.status is not RiskInputStatus.NOT_PERMITTED
    language_permitted = inputs.language.status is not RiskInputStatus.NOT_PERMITTED
    return FinalReportSummary(
        acoustic_event_count=inputs.acoustic.event_count,
        acoustic_max_peak_score=inputs.acoustic.max_peak_score,
        speech_segment_count=inputs.speech.segment_count if speech_permitted else None,
        speech_accepted_transcript_count=(
            inputs.speech.accepted_transcript_count if speech_permitted else None
        ),
        speech_review_required_transcript_count=(
            inputs.speech.review_required_transcript_count
            if speech_permitted
            else None
        ),
        language_finding_count=(
            inputs.language.finding_count if language_permitted else None
        ),
        risk_score=report.risk_assessment.score,
        risk_severity=report.risk_assessment.severity,
        outcome=report.decision.outcome,
        review_required=report.decision.review_required,
        alert_candidate=report.decision.alert_candidate,
    )


def _validate_report_alignment(report: FinalReportDocument) -> None:
    assessment = report.risk_assessment
    agreement = report.agreement
    decision = report.decision

    if report.source.model_dump(mode="python", exclude={
        "raw_audio_sha256", "preparation_manifest_sha256"
    }) != assessment.inputs.source.model_dump(mode="python"):
        raise ValueError("report source must match the risk source")

    for receipt in report.evidence:
        branch = _branch_input(assessment, receipt.kind)
        if receipt.input_status is not branch.status:
            raise ValueError("evidence receipt status must match the risk input")
        reference: RiskEvidenceReference | None = branch.evidence
        if reference is not None:
            if (
                receipt.evidence_id != reference.evidence_id
                or receipt.evidence_sha256 != reference.evidence_sha256
            ):
                raise ValueError("evidence receipt must match the risk reference")
        elif branch.status is not RiskInputStatus.NO_ACCEPTED_TEXT and (
            receipt.evidence_id is not None or receipt.evidence_sha256 is not None
        ):
            raise ValueError("non-referenced risk input cannot add report evidence")

    assessment_hash = risk_assessment_sha256(assessment)
    if (
        agreement.assessment_id != assessment.assessment_id
        or agreement.assessment_sha256 != assessment_hash
    ):
        raise ValueError("agreement must reference the embedded risk assessment")
    if decision.risk_assessment != _expected_risk_reference(assessment):
        raise ValueError("decision must reference the embedded risk assessment")
    if decision.branch_states != agreement.branch_states:
        raise ValueError("decision branch states must match the agreement")
    if decision.decision_id != _decision_identity(decision):
        raise ValueError("decision_id must match canonical decision content")
    if report.summary != _expected_summary(report):
        raise ValueError("report summary must match the embedded documents")


def _validated_result(
    result: OfflineEvaluationResult,
) -> tuple[
    PreparedAudioManifest,
    AcousticEvidenceDocument,
    SpeechEvidenceDocument | None,
    LanguageEvidenceDocument | None,
    RiskAssessmentDocument,
    ConsensusAgreementEvaluation,
    ConsensusDecisionDocument,
]:
    if not isinstance(result, OfflineEvaluationResult):
        raise FinalReportError(
            "invalid_evaluation", "A completed A8.1 evaluation result is required."
        )
    try:
        manifest = PreparedAudioManifest.model_validate(
            result.prepared.manifest.model_dump(mode="python")
        )
        acoustic = AcousticEvidenceDocument.model_validate(
            result.acoustic_evidence.evidence.model_dump(mode="python")
        )
        speech = (
            None
            if result.speech is None
            else SpeechEvidenceDocument.model_validate(
                result.speech.evidence.model_dump(mode="python")
            )
        )
        language = (
            None
            if result.language is None
            else LanguageEvidenceDocument.model_validate(
                result.language.evidence.model_dump(mode="python")
            )
        )
        assessment = RiskAssessmentDocument.model_validate(
            result.risk_assessment.model_dump(mode="python")
        )
        agreement = ConsensusAgreementEvaluation.model_validate(
            result.agreement.model_dump(mode="python")
        )
        decision = ConsensusDecisionDocument.model_validate(
            result.decision.model_dump(mode="python")
        )
    except Exception as error:
        raise FinalReportError(
            "invalid_evaluation", "Evaluation artifacts failed contract validation."
        ) from error
    if (
        manifest != result.prepared.manifest
        or acoustic != result.acoustic_evidence.evidence
        or (speech is None) != (result.speech is None)
        or (
            speech is not None
            and result.speech is not None
            and speech != result.speech.evidence
        )
        or (language is None) != (result.language is None)
        or (
            language is not None
            and result.language is not None
            and language != result.language.evidence
        )
        or (result.speech_segmentation is None) != (speech is None)
        or assessment != result.risk_assessment
        or agreement != result.agreement
        or decision != result.decision
        or result.risk_inputs != assessment.inputs
    ):
        raise FinalReportError(
            "invalid_evaluation", "Evaluation artifacts failed integrity validation."
        )
    _validate_evidence_alignment(manifest, acoustic, speech, language, assessment)
    return manifest, acoustic, speech, language, assessment, agreement, decision


def _validate_evidence_alignment(
    manifest: PreparedAudioManifest,
    acoustic: AcousticEvidenceDocument,
    speech: SpeechEvidenceDocument | None,
    language: LanguageEvidenceDocument | None,
    assessment: RiskAssessmentDocument,
) -> None:
    source = assessment.inputs.source
    if (
        acoustic.source.clip_id != source.clip_id
        or acoustic.source.raw_audio_sha256 != manifest.source.sha256
        or acoustic.sample_rate_hz != source.sample_rate_hz
        or manifest.num_frames != source.num_samples
        or manifest.clip.clip_id != source.clip_id
        or manifest.clip.consent.consent_id != source.consent_id
        or manifest.clip.consent.processing_scope is not source.processing_scope
    ):
        raise FinalReportError(
            "source_mismatch", "Evaluation evidence does not share one source snapshot."
        )

    documents: dict[RiskEvidenceKind, BaseModel | None] = {
        RiskEvidenceKind.ACOUSTIC: acoustic,
        RiskEvidenceKind.SPEECH: speech,
        RiskEvidenceKind.LANGUAGE: language,
    }
    for kind, document in documents.items():
        branch = _branch_input(assessment, kind)
        if document is None:
            if branch.status is not RiskInputStatus.NOT_PERMITTED:
                raise FinalReportError(
                    "missing_evidence", "A permitted report evidence branch is absent."
                )
            continue
        if branch.status in {
            RiskInputStatus.NOT_PERMITTED,
            RiskInputStatus.MISSING,
            RiskInputStatus.NOT_APPLICABLE,
        }:
            raise FinalReportError(
                "evidence_mismatch",
                "An unavailable risk branch cannot carry a report artifact.",
            )
        evidence_id = getattr(document, "evidence_id")
        document_hash = _artifact_hash(document)
        if branch.evidence is not None and (
            branch.evidence.evidence_id != evidence_id
            or branch.evidence.evidence_sha256 != document_hash
        ):
            raise FinalReportError(
                "evidence_mismatch", "Risk inputs do not match their evidence artifacts."
            )
        if (
            branch.status is RiskInputStatus.NO_ACCEPTED_TEXT
            and kind is not RiskEvidenceKind.LANGUAGE
        ):
            raise FinalReportError(
                "evidence_mismatch", "Only language may use no_accepted_text status."
            )

    if speech is not None and (
        speech.source.clip_id != source.clip_id
        or speech.source.consent_id != source.consent_id
        or speech.source.sample_rate_hz != source.sample_rate_hz
        or speech.source.num_samples != source.num_samples
        or speech.source.preparation_manifest_path
        != acoustic.source.preparation_manifest_path
        or speech.source.preparation_manifest_sha256
        != acoustic.source.preparation_manifest_sha256
        or speech.source.raw_audio_sha256 != acoustic.source.raw_audio_sha256
    ):
        raise FinalReportError(
            "source_mismatch", "Speech evidence differs from the report source."
        )
    if language is not None and (
        speech is None
        or language.source.clip_id != source.clip_id
        or language.source.consent_id != source.consent_id
        or language.source.sample_rate_hz != source.sample_rate_hz
        or language.source.num_samples != source.num_samples
        or language.source.speech_evidence_id != speech.evidence_id
        or language.source.speech_evidence_sha256 != _artifact_hash(speech)
    ):
        raise FinalReportError(
            "source_mismatch", "Language evidence differs from the speech source."
        )


def _receipt(
    kind: RiskEvidenceKind,
    status: RiskInputStatus,
    document: BaseModel | None,
) -> FinalReportEvidenceReceipt:
    return FinalReportEvidenceReceipt(
        kind=kind,
        input_status=status,
        document_type=_DOCUMENT_TYPE_BY_KIND[kind],
        evidence_id=None if document is None else getattr(document, "evidence_id"),
        evidence_sha256=None if document is None else _artifact_hash(document),
    )


def build_final_report(
    result: OfflineEvaluationResult,
    *,
    now: datetime | None = None,
) -> FinalReportDocument:
    """Build one portable report from a complete, internally aligned evaluation."""

    try:
        (
            manifest,
            acoustic,
            speech,
            language,
            assessment,
            agreement,
            decision,
        ) = _validated_result(result)
        created_at = _clock(now)
        source = FinalReportSource(
            clip_id=manifest.clip.clip_id,
            consent_id=manifest.clip.consent.consent_id,
            processing_scope=manifest.clip.consent.processing_scope,
            sample_rate_hz=manifest.clip.sample_rate_hz,
            num_samples=manifest.num_frames,
            raw_audio_sha256=manifest.source.sha256,
            preparation_manifest_sha256=(
                acoustic.source.preparation_manifest_sha256
            ),
        )
        evidence = (
            _receipt(
                RiskEvidenceKind.ACOUSTIC,
                assessment.inputs.acoustic.status,
                acoustic,
            ),
            _receipt(
                RiskEvidenceKind.SPEECH,
                assessment.inputs.speech.status,
                speech,
            ),
            _receipt(
                RiskEvidenceKind.LANGUAGE,
                assessment.inputs.language.status,
                language,
            ),
        )
        placeholder = "report-" + "0" * 64
        provisional = FinalReportDocument.model_construct(
            schema_version=FINAL_REPORT_SCHEMA_VERSION,
            document_type="offline_evaluation_report",
            format_version=FINAL_REPORT_FORMAT_VERSION,
            report_id=placeholder,
            created_at=created_at,
            source=source,
            evidence=evidence,
            summary=FinalReportSummary(
                acoustic_event_count=assessment.inputs.acoustic.event_count,
                acoustic_max_peak_score=assessment.inputs.acoustic.max_peak_score,
                speech_segment_count=(
                    None
                    if assessment.inputs.speech.status
                    is RiskInputStatus.NOT_PERMITTED
                    else assessment.inputs.speech.segment_count
                ),
                speech_accepted_transcript_count=(
                    None
                    if assessment.inputs.speech.status
                    is RiskInputStatus.NOT_PERMITTED
                    else assessment.inputs.speech.accepted_transcript_count
                ),
                speech_review_required_transcript_count=(
                    None
                    if assessment.inputs.speech.status
                    is RiskInputStatus.NOT_PERMITTED
                    else assessment.inputs.speech.review_required_transcript_count
                ),
                language_finding_count=(
                    None
                    if assessment.inputs.language.status
                    is RiskInputStatus.NOT_PERMITTED
                    else assessment.inputs.language.finding_count
                ),
                risk_score=assessment.score,
                risk_severity=assessment.severity,
                outcome=decision.outcome,
                review_required=decision.review_required,
                alert_candidate=decision.alert_candidate,
            ),
            risk_assessment=assessment,
            agreement=agreement,
            decision=decision,
        )
        return FinalReportDocument.model_validate(
            provisional.model_copy(
                update={"report_id": _report_id(provisional)}
            ).model_dump(mode="python")
        )
    except FinalReportError:
        raise
    except Exception as error:
        raise FinalReportError(
            "invalid_evaluation",
            "Evaluation artifacts could not form a valid final report.",
        ) from error


def _output_parent(paths: Paths) -> Path:
    try:
        root = paths.root.resolve(strict=True)
        processed = Path(os.path.abspath(paths.processed_data))
        parts = processed.relative_to(root).parts
        if not parts:
            raise ValueError("root is not an output directory")
        for other in (paths.raw_data, paths.interim_data, paths.models):
            other = other.resolve()
            if processed.is_relative_to(other) or other.is_relative_to(processed):
                raise ValueError("output overlaps protected project data")
        current = root
        for part in (*parts, FINAL_REPORT_DIRECTORY):
            current = current / part
            if _is_link(current):
                raise ValueError("linked output")
            current.mkdir(exist_ok=True)
        return current.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise FinalReportError(
            "invalid_output_path",
            "Final report output must be separate from source and model data without linked directories.",
        ) from error


def _document_bytes(document: FinalReportDocument) -> bytes:
    return (document.model_dump_json(indent=2) + "\n").encode("utf-8")


def _load(
    paths: Paths,
    report_path: str | Path,
    policy: FinalReportPersistencePolicy,
) -> FinalReportDocument:
    try:
        path = _safe_file(paths.root, paths.processed_data, report_path)
        raw = _read_bytes(path, policy.max_document_bytes)
        report = FinalReportDocument.model_validate_json(raw)
        expected = (
            paths.processed_data
            / FINAL_REPORT_DIRECTORY
            / report.report_id
            / FINAL_REPORT_FILENAME
        )
        if path != expected or {item.name for item in path.parent.iterdir()} != {
            FINAL_REPORT_FILENAME
        }:
            raise FinalReportError(
                "output_conflict",
                "Report identity does not match its bundle path or inventory.",
            )
        return report
    except FinalReportError:
        raise
    except MemoryError as error:
        raise FinalReportError(
            "insufficient_memory", "Not enough memory to load the final report."
        ) from error
    except Exception as error:
        code = getattr(error, "code", "invalid_document")
        raise FinalReportError(
            code, "Final report could not be read or verified."
        ) from error


def load_final_report(
    paths: Paths,
    report_path: str | Path,
    *,
    policy: FinalReportPersistencePolicy | None = None,
) -> FinalReportDocument:
    """Reload a path relative to processed_data and verify report integrity."""

    resolved = FinalReportPersistencePolicy.model_validate(
        (policy or FinalReportPersistencePolicy()).model_dump(mode="python")
    )
    return _load(paths, report_path, resolved)


def save_final_report(
    paths: Paths,
    result: OfflineEvaluationResult,
    *,
    policy: FinalReportPersistencePolicy | None = None,
    now: datetime | None = None,
) -> SavedFinalReport:
    """Atomically persist and reload one immutable final report bundle."""

    resolved = FinalReportPersistencePolicy.model_validate(
        (policy or FinalReportPersistencePolicy()).model_dump(mode="python")
    )
    stage: Path | None = None
    parent: Path | None = None
    try:
        report = build_final_report(result, now=now)
        try:
            evidence_relative = result.acoustic_evidence.evidence_path.relative_to(
                paths.processed_data
            )
        except ValueError as error:
            raise FinalReportError(
                "source_mismatch",
                "Acoustic evidence is outside the configured processed directory.",
            ) from error
        try:
            current_acoustic = load_acoustic_evidence(
                paths, evidence_relative, now=now
            )
        except AcousticEvidenceError as error:
            raise FinalReportError(
                "source_changed",
                "Acoustic evidence could not be reverified before report persistence.",
            ) from error
        if current_acoustic != result.acoustic_evidence.evidence:
            raise FinalReportError(
                "source_changed", "Acoustic evidence changed before report persistence."
            )

        raw = _document_bytes(report)
        if len(raw) > resolved.max_document_bytes:
            raise FinalReportError(
                "output_too_large",
                "Final report exceeds the configured document byte limit.",
            )
        parent = _output_parent(paths)
        destination = parent / report.report_id
        relative = (
            f"{FINAL_REPORT_DIRECTORY}/{report.report_id}/{FINAL_REPORT_FILENAME}"
        )
        if _is_link(destination) or (
            destination.exists() and not destination.is_dir()
        ):
            raise FinalReportError(
                "output_conflict",
                "Final report destination is not a regular bundle directory.",
            )
        stage = Path(mkdtemp(prefix=".pending-", dir=parent)).resolve()
        staged_path = stage / FINAL_REPORT_FILENAME
        with staged_path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if FinalReportDocument.model_validate_json(
            _read_bytes(staged_path, resolved.max_document_bytes)
        ) != report:
            raise FinalReportError(
                "verification_failed", "Final report JSON failed readback."
            )

        reused = destination.exists()
        if not reused:
            try:
                stage.rename(destination)
                stage = None
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) and not isinstance(
                    error, FileExistsError
                ):
                    raise
                reused = True
        if reused:
            existing = _load(paths, relative, resolved)
            comparable = existing.model_copy(update={"created_at": report.created_at})
            if comparable != report:
                raise FinalReportError(
                    "output_conflict",
                    "Existing final report differs; it was not overwritten.",
                )
            report = existing
        loaded = _load(paths, relative, resolved)
        if loaded != report:
            raise FinalReportError(
                "verification_failed", "Persisted final report failed verification."
            )
        return SavedFinalReport(
            directory=destination,
            report_path=destination / FINAL_REPORT_FILENAME,
            report=report,
            reused=reused,
        )
    except FinalReportError:
        raise
    except MemoryError as error:
        raise FinalReportError(
            "insufficient_memory", "Not enough memory to persist the final report."
        ) from error
    except OSError as error:
        raise FinalReportError(
            "write_failed", "Final report could not be written or verified."
        ) from error
    finally:
        if stage is not None and parent is not None:
            _remove_stage(stage, parent)


def final_report_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public JSON Schema for the B8.1 final report."""

    return {"final-report.schema.json": FinalReportDocument.model_json_schema()}


def write_final_report_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the portable final-report schema for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in final_report_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        exported[filename] = destination
    return exported
