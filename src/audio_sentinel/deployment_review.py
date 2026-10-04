"""A10.4 final privacy, security, and deployment review contract."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEPLOYMENT_REVIEW_SCHEMA_VERSION = "1.0"
DEPLOYMENT_REVIEW_FORMAT_VERSION = "1.0"
MAX_DEPLOYMENT_REVIEW_BYTES = 2_097_152
MAX_REVIEW_EVIDENCE_BYTES = 16_777_216


class DeploymentReviewError(RuntimeError):
    """Stable A10.4 failure with a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ReviewDomain(str, Enum):
    PRIVACY = "privacy"
    SECURITY = "security"
    DEPLOYMENT = "deployment"


class FindingDisposition(str, Enum):
    VERIFIED_CONTROL = "verified_control"
    CONDITIONAL_CONTROL = "conditional_control"
    RELEASE_BLOCKER = "release_blocker"


class RiskSeverity(str, Enum):
    INFORMATIONAL = "informational"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class ReviewEvidenceArtifact(ReviewRecord):
    evidence_id: str = Field(pattern=r"^EV-[0-9]{2}$")
    path: str = Field(min_length=3, max_length=240)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(gt=0, le=MAX_REVIEW_EVIDENCE_BYTES, strict=True)
    purpose: str = Field(min_length=12, max_length=240)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in value:
            raise ValueError("evidence paths must be normalized repository-relative paths")
        return value


class DeploymentReviewFinding(ReviewRecord):
    finding_id: str = Field(pattern=r"^(PRIV|SEC|DEP)-[0-9]{2}$")
    domain: ReviewDomain
    disposition: FindingDisposition
    severity: RiskSeverity
    title: str = Field(min_length=8, max_length=120)
    conclusion: str = Field(min_length=24, max_length=900)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    required_action: str | None = Field(default=None, min_length=24, max_length=900)

    @model_validator(mode="after")
    def validate_disposition(self) -> "DeploymentReviewFinding":
        prefix = {
            ReviewDomain.PRIVACY: "PRIV-",
            ReviewDomain.SECURITY: "SEC-",
            ReviewDomain.DEPLOYMENT: "DEP-",
        }[self.domain]
        if not self.finding_id.startswith(prefix):
            raise ValueError("finding identity must match its review domain")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("finding evidence identities must be unique")
        if self.disposition is FindingDisposition.RELEASE_BLOCKER:
            if self.required_action is None:
                raise ValueError("release blockers require a concrete action")
            if self.severity not in {RiskSeverity.HIGH, RiskSeverity.CRITICAL}:
                raise ValueError("release blockers must be high or critical severity")
        elif self.required_action is not None:
            raise ValueError("only release blockers may define a required action")
        return self


class DeploymentReviewCounts(ReviewRecord):
    total_findings: int = Field(ge=1, strict=True)
    verified_controls: int = Field(ge=0, strict=True)
    conditional_controls: int = Field(ge=0, strict=True)
    release_blockers: int = Field(ge=1, strict=True)
    privacy_findings: int = Field(ge=1, strict=True)
    security_findings: int = Field(ge=1, strict=True)
    deployment_findings: int = Field(ge=1, strict=True)


class FinalDeploymentReview(ReviewRecord):
    """Portable final review. It is evidence, not a production approval."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": (
                "https://audio-sentinel.local/schemas/v1/"
                "final-deployment-review.schema.json"
            )
        },
    )

    schema_version: Literal["1.0"] = DEPLOYMENT_REVIEW_SCHEMA_VERSION
    document_type: Literal["final_deployment_review"] = "final_deployment_review"
    format_version: Literal["1.0"] = DEPLOYMENT_REVIEW_FORMAT_VERSION
    review_id: str = Field(pattern=r"^deployment-review-[a-f0-9]{64}$")
    reviewed_at: datetime
    review_status: Literal["completed"] = "completed"
    release_decision: Literal["not_approved_for_production"] = (
        "not_approved_for_production"
    )
    allowed_use: Literal["controlled_local_research_and_offline_evaluation_only"] = (
        "controlled_local_research_and_offline_evaluation_only"
    )
    evidence: tuple[ReviewEvidenceArtifact, ...] = Field(min_length=1, max_length=64)
    findings: tuple[DeploymentReviewFinding, ...] = Field(min_length=1, max_length=64)
    counts: DeploymentReviewCounts
    production_approved: Literal[False] = False
    independent_security_review_completed: Literal[False] = False
    representative_field_validation_completed: Literal[False] = False
    notification_delivery_authorized: Literal[False] = False
    external_action_authorized: Literal[False] = False

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_review(self) -> "FinalDeploymentReview":
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        finding_ids = tuple(item.finding_id for item in self.findings)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence identities must be unique")
        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("finding identities must be unique")
        if evidence_ids != tuple(sorted(evidence_ids)):
            raise ValueError("evidence inventory must be ordered by identity")
        if finding_ids != tuple(sorted(finding_ids)):
            raise ValueError("finding inventory must be ordered by identity")
        available = set(evidence_ids)
        if any(not set(item.evidence_ids) <= available for item in self.findings):
            raise ValueError("every finding must reference declared evidence")
        if {item.domain for item in self.findings} != set(ReviewDomain):
            raise ValueError("the final review must cover all three domains")
        expected = _summarize_findings(self.findings)
        if self.counts != expected:
            raise ValueError("review counts do not match the finding inventory")
        if self.counts.release_blockers < 1:
            raise ValueError("a non-production decision requires explicit blockers")
        if self.review_id != _review_id(self):
            raise ValueError("review_id must match canonical review content")
        return self


class _EvidenceDefinition(NamedTuple):
    evidence_id: str
    path: str
    purpose: str


class _FindingDefinition(NamedTuple):
    finding_id: str
    domain: ReviewDomain
    disposition: FindingDisposition
    severity: RiskSeverity
    title: str
    conclusion: str
    evidence_ids: tuple[str, ...]
    required_action: str | None = None


_EVIDENCE_DEFINITIONS = (
    _EvidenceDefinition("EV-01", "docs/authorized-use-and-limitations.md", "Consent, human-review, prohibited-use, and release boundary."),
    _EvidenceDefinition("EV-02", "docs/retention-and-deletion.md", "Implemented retention scope and documented deletion limitations."),
    _EvidenceDefinition("EV-03", "docs/local-api.md", "Loopback API inputs, exclusions, and trust boundary."),
    _EvidenceDefinition("EV-04", "docs/live-device-streaming-interface.md", "Authorized-device, consent, format, and session contract."),
    _EvidenceDefinition("EV-05", "docs/streaming-ingestion.md", "Enrollment, HMAC authentication, revocation, and ingestion behavior."),
    _EvidenceDefinition("EV-06", "docs/live-rolling-windows.md", "Memory-only rolling-window and retry boundary."),
    _EvidenceDefinition("EV-07", "docs/stream-resilience.md", "Bounded buffering, backpressure, discard, and reconnect behavior."),
    _EvidenceDefinition("EV-08", "docs/live-performance-measurement.md", "Measured live-path evidence and explicit measurement exclusions."),
    _EvidenceDefinition("EV-09", "docs/evaluation-findings.md", "Held-out acoustic results and limits on empirical claims."),
    _EvidenceDefinition("EV-10", "src/audio_sentinel/api.py", "Executable loopback-only API boundary."),
    _EvidenceDefinition("EV-11", "src/audio_sentinel/streaming_ingestion.py", "Executable local authentication, revocation, and stream intake."),
    _EvidenceDefinition("EV-12", "src/audio_sentinel/stream_resilience.py", "Executable bounded queue, explicit discard, and reconnect policy."),
    _EvidenceDefinition("EV-13", "src/audio_sentinel/final_report.py", "Privacy-minimized integrity-checked final report contract."),
    _EvidenceDefinition("EV-14", "outputs/a10_3_measurement/live-performance-report.json", "Checked-in A10.3 latency and reliability measurement receipt."),
    _EvidenceDefinition("EV-15", "tests/test_api.py", "Automated checks for local API access and input exclusions."),
    _EvidenceDefinition("EV-16", "tests/test_streaming_ingestion.py", "Automated authentication, replay, revocation, and audio-retention checks."),
)


_FINDING_DEFINITIONS = (
    _FindingDefinition(
        "DEP-01", ReviewDomain.DEPLOYMENT, FindingDisposition.CONDITIONAL_CONTROL,
        RiskSeverity.MEDIUM, "Measured local live-path regression",
        "The checked-in A10.3 run reconciles delivery, retry, backpressure, windows, and one local-alert timing probe, but it is an accelerated in-process regression only.",
        ("EV-08", "EV-14"),
    ),
    _FindingDefinition(
        "DEP-02", ReviewDomain.DEPLOYMENT, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.CRITICAL, "No production live-device path",
        "The repository has no microphone or driver capture adapter, production network transport, background service supervision, or approved live model orchestration.",
        ("EV-04", "EV-05", "EV-06", "EV-07", "EV-08"),
        "Implement and independently test the intended hardware capture, transport, service lifecycle, and approved model path in the target environment.",
    ),
    _FindingDefinition(
        "DEP-03", ReviewDomain.DEPLOYMENT, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.HIGH, "Performance evidence is not field representative",
        "The latency report excludes real hardware, network conditions, production inference, soak behavior, and statistically stable tail latency.",
        ("EV-08", "EV-14"),
        "Run representative hardware, concurrency, failure-recovery, soak, and end-to-end latency tests with predeclared acceptance thresholds.",
    ),
    _FindingDefinition(
        "DEP-04", ReviewDomain.DEPLOYMENT, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.CRITICAL, "No notification or reviewer operations",
        "Alerts remain local candidates; reviewer identity, assignment, disposition, delivery, escalation, and external action are deliberately not implemented or authorized.",
        ("EV-01", "EV-08"),
        "Design, approve, and test a separately controlled human-review and notification workflow with authentication, audit, escalation, and fail-safe behavior.",
    ),
    _FindingDefinition(
        "DEP-05", ReviewDomain.DEPLOYMENT, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.HIGH, "No production operations package",
        "The prototype does not provide production observability, service-level objectives, incident response, disaster recovery, signed releases, or a supported deployment runbook.",
        ("EV-01", "EV-08"),
        "Create and approve operational monitoring, incident response, recovery, release-signing, rollback, ownership, and support procedures.",
    ),
    _FindingDefinition(
        "DEP-06", ReviewDomain.DEPLOYMENT, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.CRITICAL, "Detection evidence is insufficient for deployment",
        "The small balanced acoustic evaluation is not representative of deployment prevalence, environments, speech, fairness, real incidents, or alert effectiveness.",
        ("EV-01", "EV-09"),
        "Complete independent representative validation, calibration, subgroup analysis, failure analysis, and human-factors testing before setting production thresholds.",
    ),
    _FindingDefinition(
        "PRIV-01", ReviewDomain.PRIVACY, FindingDisposition.VERIFIED_CONTROL,
        RiskSeverity.INFORMATIONAL, "Consent and least-scope gates are explicit",
        "The contracts distinguish denied, withdrawn, acoustic-only, and speech-authorized processing and require current consent at processing boundaries.",
        ("EV-01", "EV-04", "EV-05"),
    ),
    _FindingDefinition(
        "PRIV-02", ReviewDomain.PRIVACY, FindingDisposition.VERIFIED_CONTROL,
        RiskSeverity.INFORMATIONAL, "Live PCM is bounded and memory-only",
        "The live ingestion, rolling-window, and resilience layers bound audio in memory and require explicit delivery or zeroing and discard; they do not persist PCM.",
        ("EV-05", "EV-06", "EV-07", "EV-16"),
    ),
    _FindingDefinition(
        "PRIV-03", ReviewDomain.PRIVACY, FindingDisposition.VERIFIED_CONTROL,
        RiskSeverity.INFORMATIONAL, "Final outputs are privacy minimized",
        "Final reports omit raw audio, transcript text, recipients, credentials, proofs, and absolute local paths while retaining reviewable evidence and integrity metadata.",
        ("EV-01", "EV-13", "EV-14"),
    ),
    _FindingDefinition(
        "PRIV-04", ReviewDomain.PRIVACY, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.HIGH, "Data lifecycle coverage is incomplete",
        "Retention cleanup is manual and limited to verified final bundles; it is not secure erasure and does not cover raw recordings, intermediate speech text, backups, exports, or external copies.",
        ("EV-01", "EV-02"),
        "Implement policy-owned inventory, retention, legal hold, deletion verification, backup handling, and secure disposal across every data class and copy.",
    ),
    _FindingDefinition(
        "PRIV-05", ReviewDomain.PRIVACY, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.CRITICAL, "Organizational privacy approval is absent",
        "Software validation cannot establish lawful recording authority, authentic consent, notice, purpose limitation, data-subject rights, or jurisdiction-specific compliance.",
        ("EV-01", "EV-04"),
        "Obtain documented legal, privacy, ethics, and organizational approval for the exact people, places, devices, purposes, retention, and operator workflow.",
    ),
    _FindingDefinition(
        "SEC-01", ReviewDomain.SECURITY, FindingDisposition.VERIFIED_CONTROL,
        RiskSeverity.INFORMATIONAL, "External API exposure is constrained",
        "The evaluation endpoint rejects non-loopback clients and accepts a local recording reference rather than uploads, URLs, recipients, transports, or notification authority.",
        ("EV-03", "EV-10", "EV-15"),
    ),
    _FindingDefinition(
        "SEC-02", ReviewDomain.SECURITY, FindingDisposition.VERIFIED_CONTROL,
        RiskSeverity.INFORMATIONAL, "Device sessions are authenticated and revocable",
        "Live intake binds expiring enrollment, one-time HMAC challenges, single-use receipts, consent, exact PCM geometry, and ordered chunks while rechecking revocation.",
        ("EV-04", "EV-05", "EV-11", "EV-16"),
    ),
    _FindingDefinition(
        "SEC-03", ReviewDomain.SECURITY, FindingDisposition.CONDITIONAL_CONTROL,
        RiskSeverity.MEDIUM, "Integrity and atomic persistence are implemented",
        "Canonical hashes, strict schemas, bounded reads, and atomic writes detect accidental or unauthorized content changes, but hashes do not prove authorship or provide a signed chain of custody.",
        ("EV-01", "EV-02", "EV-13", "EV-14"),
    ),
    _FindingDefinition(
        "SEC-04", ReviewDomain.SECURITY, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.CRITICAL, "Credential lifecycle is not production managed",
        "HMAC possession is verified, but the credential resolver is an external integration point and the project does not provide hardware-backed storage, provisioning, rotation, recovery, or compromise response.",
        ("EV-04", "EV-05", "EV-11"),
        "Integrate an approved secret manager or hardware-backed keystore with authenticated provisioning, rotation, revocation, recovery, and compromise procedures.",
    ),
    _FindingDefinition(
        "SEC-05", ReviewDomain.SECURITY, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.CRITICAL, "Host security remains outside the application",
        "Loopback is not authentication against another local user or compromised process, and the project does not enforce operating-system identities, filesystem ACLs, encryption, process isolation, or endpoint hardening.",
        ("EV-01", "EV-03", "EV-10"),
        "Define and independently verify the production trust boundary, authenticated operator access, least-privilege service identity, encryption, ACLs, isolation, and host hardening.",
    ),
    _FindingDefinition(
        "SEC-06", ReviewDomain.SECURITY, FindingDisposition.RELEASE_BLOCKER,
        RiskSeverity.HIGH, "Independent security assurance is absent",
        "Repository tests cover important misuse and tampering cases, but there is no independent threat model, penetration test, dependency vulnerability gate, software bill of materials, or signed release evidence.",
        ("EV-01", "EV-15", "EV-16"),
        "Complete an independent threat model and security assessment, add dependency and supply-chain gates, publish an SBOM, and sign approved release artifacts.",
    ),
)


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _review_id(review: FinalDeploymentReview) -> str:
    payload = review.model_dump(mode="json", exclude={"review_id"})
    return f"deployment-review-{_canonical_hash(payload)}"


def _summarize_findings(
    findings: tuple[DeploymentReviewFinding, ...],
) -> DeploymentReviewCounts:
    return DeploymentReviewCounts(
        total_findings=len(findings),
        verified_controls=sum(
            item.disposition is FindingDisposition.VERIFIED_CONTROL for item in findings
        ),
        conditional_controls=sum(
            item.disposition is FindingDisposition.CONDITIONAL_CONTROL for item in findings
        ),
        release_blockers=sum(
            item.disposition is FindingDisposition.RELEASE_BLOCKER for item in findings
        ),
        privacy_findings=sum(item.domain is ReviewDomain.PRIVACY for item in findings),
        security_findings=sum(item.domain is ReviewDomain.SECURITY for item in findings),
        deployment_findings=sum(item.domain is ReviewDomain.DEPLOYMENT for item in findings),
    )


def _resolve_evidence(project_root: Path, definition: _EvidenceDefinition) -> ReviewEvidenceArtifact:
    root = project_root.resolve(strict=True)
    candidate = (root / Path(definition.path)).resolve(strict=True)
    if candidate == root or root not in candidate.parents or not candidate.is_file():
        raise DeploymentReviewError("invalid_evidence", "Review evidence must be a repository file.")
    size = candidate.stat().st_size
    if not 0 < size <= MAX_REVIEW_EVIDENCE_BYTES:
        raise DeploymentReviewError("invalid_evidence_size", "Review evidence size is invalid.")
    content = candidate.read_bytes()
    if len(content) != size:
        raise DeploymentReviewError("evidence_changed", "Review evidence changed while read.")
    return ReviewEvidenceArtifact(
        evidence_id=definition.evidence_id,
        path=definition.path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=size,
        purpose=definition.purpose,
    )


def build_final_deployment_review(
    project_root: Path,
    *,
    reviewed_at: datetime | None = None,
) -> FinalDeploymentReview:
    """Build the fixed-scope final review from hashed repository evidence."""

    try:
        timestamp = reviewed_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        evidence = tuple(
            _resolve_evidence(project_root, definition)
            for definition in _EVIDENCE_DEFINITIONS
        )
        findings = tuple(
            DeploymentReviewFinding(
                finding_id=item.finding_id,
                domain=item.domain,
                disposition=item.disposition,
                severity=item.severity,
                title=item.title,
                conclusion=item.conclusion,
                evidence_ids=item.evidence_ids,
                required_action=item.required_action,
            )
            for item in _FINDING_DEFINITIONS
        )
        provisional = FinalDeploymentReview.model_construct(
            schema_version=DEPLOYMENT_REVIEW_SCHEMA_VERSION,
            document_type="final_deployment_review",
            format_version=DEPLOYMENT_REVIEW_FORMAT_VERSION,
            review_id="deployment-review-" + "0" * 64,
            reviewed_at=timestamp.astimezone(UTC),
            review_status="completed",
            release_decision="not_approved_for_production",
            allowed_use="controlled_local_research_and_offline_evaluation_only",
            evidence=evidence,
            findings=findings,
            counts=_summarize_findings(findings),
            production_approved=False,
            independent_security_review_completed=False,
            representative_field_validation_completed=False,
            notification_delivery_authorized=False,
            external_action_authorized=False,
        )
        return FinalDeploymentReview.model_validate(
            provisional.model_copy(update={"review_id": _review_id(provisional)}).model_dump(mode="python")
        )
    except DeploymentReviewError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise DeploymentReviewError("review_failed", "The final review could not be built.") from error


def verify_review_evidence(
    review: FinalDeploymentReview,
    project_root: Path,
) -> tuple[str, ...]:
    """Return evidence IDs whose current repository files no longer match."""

    validated = FinalDeploymentReview.model_validate(review.model_dump(mode="python"))
    root = project_root.resolve(strict=True)
    changed: list[str] = []
    for artifact in validated.evidence:
        try:
            candidate = (root / Path(artifact.path)).resolve(strict=True)
            if candidate == root or root not in candidate.parents or not candidate.is_file():
                changed.append(artifact.evidence_id)
                continue
            size = candidate.stat().st_size
            content = candidate.read_bytes()
            if (
                size != artifact.size_bytes
                or len(content) != size
                or hashlib.sha256(content).hexdigest() != artifact.sha256
            ):
                changed.append(artifact.evidence_id)
        except OSError:
            changed.append(artifact.evidence_id)
    return tuple(changed)


def save_final_deployment_review(
    review: FinalDeploymentReview,
    destination: Path,
    *,
    replace: bool = False,
) -> None:
    """Persist a validated final review atomically; replacement is explicit."""

    if not isinstance(review, FinalDeploymentReview):
        raise TypeError("review must be FinalDeploymentReview")
    validated = FinalDeploymentReview.model_validate(review.model_dump(mode="python"))
    document = (validated.model_dump_json(indent=2) + "\n").encode("utf-8")
    if len(document) > MAX_DEPLOYMENT_REVIEW_BYTES:
        raise DeploymentReviewError("review_too_large", "The final review exceeds its size limit.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise DeploymentReviewError("output_exists", "The final review already exists.")
    stage = destination.with_name(f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with stage.open("xb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(stage, destination)
        else:
            try:
                os.link(stage, destination)
            except FileExistsError as error:
                raise DeploymentReviewError("output_exists", "The final review already exists.") from error
            stage.unlink()
    except DeploymentReviewError:
        raise
    except OSError as error:
        raise DeploymentReviewError("write_failed", "The final review could not be saved.") from error
    finally:
        try:
            stage.unlink(missing_ok=True)
        except OSError:
            pass


def load_final_deployment_review(path: Path) -> FinalDeploymentReview:
    """Load and revalidate a bounded final review."""

    try:
        size = path.stat().st_size
        if not 0 < size <= MAX_DEPLOYMENT_REVIEW_BYTES:
            raise DeploymentReviewError("invalid_review_size", "The final review size is invalid.")
        document = path.read_bytes()
        if len(document) != size:
            raise DeploymentReviewError("review_changed", "The final review changed while read.")
        return FinalDeploymentReview.model_validate_json(document)
    except DeploymentReviewError:
        raise
    except Exception as error:
        raise DeploymentReviewError("invalid_review", "The final review failed validation.") from error


def deployment_review_schema_document() -> dict[str, object]:
    return FinalDeploymentReview.model_json_schema()
