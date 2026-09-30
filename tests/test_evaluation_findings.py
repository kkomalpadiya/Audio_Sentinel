"""A9.1 pinned manifest, observed metric, and privacy checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from audio_sentinel.consensus_contracts import ConsensusOutcome
from audio_sentinel.evaluation_manifest import (
    EvaluationRunDocument,
    load_evaluation_manifest,
)
from audio_sentinel.evaluation_service import EvaluationScope


MANIFEST_SHA256 = "24da0b0fe6a352f00c1b79e9830db8ce79e8cd3f942776a6ea91be8c5dffefe9"
RUN_SHA256 = "a2fd00c159c50c7a320b79c979eaa9b6e7f20eb4f02b651fcfd8f59a123b7d33"


def test_manifest_builder_reproduces_pinned_balanced_selection(
    project_root: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "a9-1-manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_a9_1_evaluation_manifest.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    artifact = output.read_bytes()
    loaded = load_evaluation_manifest(output, expected_sha256=MANIFEST_SHA256)
    manifest = loaded.manifest

    assert hashlib.sha256(artifact).hexdigest() == MANIFEST_SHA256
    assert manifest.manifest_id == "evaluation-manifest-b643e0f3913f8223415cf662"
    assert len(manifest.cases) == 32
    assert sum(case.expected_positive for case in manifest.cases) == 16
    assert sum(not case.expected_positive for case in manifest.cases) == 16
    assert manifest.positive_outcomes == (
        ConsensusOutcome.LOG,
        ConsensusOutcome.REVIEW,
        ConsensusOutcome.ALERT,
    )
    assert all(
        case.request.processing_scope is EvaluationScope.ACOUSTIC_ONLY
        and case.request.acoustic_threshold == 0.5
        for case in manifest.cases
    )
    for dataset_id in ("esc50", "urbansound8k"):
        dataset_cases = [case for case in manifest.cases if case.categories[0] == dataset_id]
        assert sum(case.expected_positive for case in dataset_cases) == 8
        negatives = [case for case in dataset_cases if not case.expected_positive]
        assert len(negatives) == 8
        assert len({case.categories[1] for case in negatives}) == 8
    assert "scores" not in artifact.decode("utf-8")


def test_checked_in_run_is_complete_reproducible_and_privacy_minimized(
    project_root: Path,
) -> None:
    path = project_root / "outputs" / "a9_1_evaluation" / "evaluation-run.json"
    artifact = path.read_bytes()
    report = EvaluationRunDocument.model_validate_json(artifact)

    assert hashlib.sha256(artifact).hexdigest() == RUN_SHA256
    assert report.run_id == "evaluation-run-33c6a7a6fd06d11f4722ef03"
    assert report.manifest_sha256 == MANIFEST_SHA256
    assert report.decision_status == "complete"
    assert report.metrics.completion_rate == 1.0
    assert report.metrics.failed_case_count == 0
    assert (
        report.metrics.binary.true_positive,
        report.metrics.binary.false_positive,
        report.metrics.binary.false_negative,
        report.metrics.binary.true_negative,
    ) == (13, 2, 3, 14)
    assert report.metrics.binary.precision == 13 / 15
    assert report.metrics.binary.recall == 13 / 16
    assert report.metrics.binary.specificity == 14 / 16
    assert report.metrics.binary.accuracy == 27 / 32
    assert report.notification_delivery == "not_sent"
    assert report.alert_delivery_authorized is False

    serialized = artifact.decode("utf-8")
    assert "audio_path" not in serialized
    assert "consent_id" not in serialized
    assert "transcript" not in serialized.lower()
    assert str(project_root) not in serialized


def test_observed_error_cases_match_documented_categories(project_root: Path) -> None:
    report = EvaluationRunDocument.model_validate_json(
        (
            project_root
            / "outputs"
            / "a9_1_evaluation"
            / "evaluation-run.json"
        ).read_bytes()
    )
    false_positives = {
        (case.categories[0], case.categories[1])
        for case in report.cases
        if not case.expected_positive and case.observed_positive
    }
    false_negatives = {
        (case.categories[0], case.categories[1])
        for case in report.cases
        if case.expected_positive and not case.observed_positive
    }

    assert false_positives == {("esc50", "cow"), ("esc50", "wind")}
    assert false_negatives == {
        ("esc50", "glass_breaking"),
        ("urbansound8k", "siren"),
    }
    assert sum(
        case.expected_positive
        and not case.observed_positive
        and case.categories[1] == "glass_breaking"
        for case in report.cases
    ) == 2


def test_checked_in_run_json_contains_no_nonstandard_numbers(project_root: Path) -> None:
    document = json.loads(
        (
            project_root
            / "outputs"
            / "a9_1_evaluation"
            / "evaluation-run.json"
        ).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard number: {value}")
        ),
    )
    assert document["metrics"]["total_case_count"] == 32
