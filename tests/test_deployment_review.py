"""A10.4 tests for the final privacy, security, and deployment review."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from pydantic import ValidationError

from audio_sentinel.deployment_review import (
    DeploymentReviewError,
    DeploymentReviewFinding,
    FinalDeploymentReview,
    FindingDisposition,
    ReviewDomain,
    ReviewEvidenceArtifact,
    RiskSeverity,
    build_final_deployment_review,
    deployment_review_schema_document,
    load_final_deployment_review,
    save_final_deployment_review,
    verify_review_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 4, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def review() -> FinalDeploymentReview:
    return build_final_deployment_review(ROOT, reviewed_at=NOW)


def test_review_records_explicit_non_production_decision(
    review: FinalDeploymentReview,
) -> None:
    assert review.release_decision == "not_approved_for_production"
    assert review.allowed_use == "controlled_local_research_and_offline_evaluation_only"
    assert review.production_approved is False
    assert review.independent_security_review_completed is False
    assert review.representative_field_validation_completed is False
    assert review.notification_delivery_authorized is False
    assert review.external_action_authorized is False


def test_review_covers_all_domains_and_reconciles_counts(
    review: FinalDeploymentReview,
) -> None:
    assert {item.domain for item in review.findings} == set(ReviewDomain)
    assert review.counts.total_findings == len(review.findings) == 17
    assert review.counts.verified_controls == 5
    assert review.counts.conditional_controls == 2
    assert review.counts.release_blockers == 10
    assert review.counts.privacy_findings == 5
    assert review.counts.security_findings == 6
    assert review.counts.deployment_findings == 6


def test_every_blocker_has_high_severity_and_required_action(
    review: FinalDeploymentReview,
) -> None:
    blockers = [
        item
        for item in review.findings
        if item.disposition is FindingDisposition.RELEASE_BLOCKER
    ]
    assert blockers
    assert all(item.required_action for item in blockers)
    assert all(
        item.severity in {RiskSeverity.HIGH, RiskSeverity.CRITICAL}
        for item in blockers
    )


def test_every_finding_references_hashed_repository_evidence(
    review: FinalDeploymentReview,
) -> None:
    available = {item.evidence_id for item in review.evidence}
    assert all(set(item.evidence_ids) <= available for item in review.findings)
    assert verify_review_evidence(review, ROOT) == ()
    for item in review.evidence:
        assert (ROOT / item.path).is_file()
        assert len(item.sha256) == 64


def test_review_is_portable_and_contains_no_secret_or_local_path(
    review: FinalDeploymentReview,
) -> None:
    document = review.model_dump_json()
    assert str(ROOT) not in document
    assert "C:\\" not in document
    assert "credential_value" not in document
    assert "shared_secret" not in document
    assert "hmac_proof" not in document
    assert "transcript_text" not in document
    assert "raw_audio" not in document


def test_review_identity_and_counts_reject_tampering(
    review: FinalDeploymentReview,
) -> None:
    bad_identity = review.model_dump(mode="python")
    bad_identity["review_id"] = "deployment-review-" + "0" * 64
    with pytest.raises(ValidationError):
        FinalDeploymentReview.model_validate(bad_identity)

    bad_counts = review.model_dump(mode="python")
    bad_counts["counts"]["release_blockers"] -= 1
    with pytest.raises(ValidationError):
        FinalDeploymentReview.model_validate(bad_counts)


def test_invalid_finding_and_evidence_contracts_are_rejected() -> None:
    with pytest.raises(ValidationError):
        DeploymentReviewFinding(
            finding_id="PRIV-99",
            domain=ReviewDomain.PRIVACY,
            disposition=FindingDisposition.RELEASE_BLOCKER,
            severity=RiskSeverity.HIGH,
            title="Missing required action",
            conclusion="This blocker deliberately omits its required remediation action.",
            evidence_ids=("EV-01",),
        )
    with pytest.raises(ValidationError):
        ReviewEvidenceArtifact(
            evidence_id="EV-99",
            path="../outside.txt",
            sha256="0" * 64,
            size_bytes=1,
            purpose="Invalid path traversal evidence.",
        )


def test_naive_review_time_is_rejected() -> None:
    with pytest.raises(DeploymentReviewError) as captured:
        build_final_deployment_review(ROOT, reviewed_at=datetime(2026, 10, 4))
    assert captured.value.code == "review_failed"


def test_evidence_change_is_detected(
    review: FinalDeploymentReview,
    tmp_path: Path,
) -> None:
    for item in review.evidence:
        source = ROOT / item.path
        destination = tmp_path / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    changed = review.evidence[0]
    with (tmp_path / changed.path).open("ab") as handle:
        handle.write(b"\nchanged after review\n")
    assert verify_review_evidence(review, tmp_path) == (changed.evidence_id,)


def test_save_load_and_explicit_replacement(
    review: FinalDeploymentReview,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "final-review.json"
    save_final_deployment_review(review, destination)
    assert load_final_deployment_review(destination) == review
    with pytest.raises(DeploymentReviewError) as captured:
        save_final_deployment_review(review, destination)
    assert captured.value.code == "output_exists"
    save_final_deployment_review(review, destination, replace=True)
    assert load_final_deployment_review(destination) == review


def test_invalid_saved_review_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "invalid-review.json"
    destination.write_text("{}", encoding="utf-8")
    with pytest.raises(DeploymentReviewError) as captured:
        load_final_deployment_review(destination)
    assert captured.value.code == "invalid_review"


def test_checked_in_schema_matches_runtime_contract() -> None:
    checked_in = json.loads(
        (ROOT / "docs/schemas/v1/final-deployment-review.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert checked_in == deployment_review_schema_document()


def test_review_script_writes_reloadable_report(tmp_path: Path) -> None:
    output = tmp_path / "generated-review.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_final_deployment_review.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    summary = json.loads(completed.stdout)
    loaded = load_final_deployment_review(output)
    assert summary["review_id"] == loaded.review_id
    assert summary["release_decision"] == "not_approved_for_production"
    assert summary["release_blockers"] == loaded.counts.release_blockers
    assert summary["production_approved"] is False


def test_module_import_has_no_network_or_model_runtime_dependency() -> None:
    code = """
import sys
class RejectImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'onnxruntime'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RejectImports())
def reject(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError('Unexpected network access')
sys.addaudithook(reject)
from audio_sentinel.deployment_review import FinalDeploymentReview
assert FinalDeploymentReview.__name__ == 'FinalDeploymentReview'
"""
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        capture_output=True,
        timeout=30,
    )
