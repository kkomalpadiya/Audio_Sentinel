"""B9.1 evaluation-manifest, runner, metric, and schema tests."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.consensus_contracts import ConsensusOutcome
from audio_sentinel.evaluation_manifest import (
    EvaluationCaseResult,
    EvaluationManifestCase,
    EvaluationManifestDocument,
    EvaluationManifestError,
    EvaluationRunDocument,
    build_evaluation_manifest,
    calculate_evaluation_metrics,
    evaluation_schema_documents,
    load_evaluation_manifest,
    run_evaluation_manifest,
    save_evaluation_run,
)
from audio_sentinel.evaluation_service import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationScope,
)


NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def _request(index: int) -> EvaluationRequest:
    return EvaluationRequest(
        audio_path=f"evaluation/case-{index}.wav",
        clip_id=f"clip-{index}",
        consent_id=f"consent-{index}",
        processing_scope=EvaluationScope.ACOUSTIC_ONLY,
        device_authorized=True,
        granted_at=datetime(2026, 9, 1, tzinfo=UTC),
        source_dataset="b9-fixture",
        acoustic_threshold=0.5,
    )


def _case(
    index: int,
    *,
    expected_positive: bool,
    expected_outcome: ConsensusOutcome | None = None,
) -> EvaluationManifestCase:
    return EvaluationManifestCase(
        case_id=f"case-{index}",
        request=_request(index),
        expected_positive=expected_positive,
        expected_outcome=expected_outcome,
        categories=("synthetic", "positive" if expected_positive else "negative"),
    )


def _manifest() -> EvaluationManifestDocument:
    return build_evaluation_manifest(
        "B9 fixture",
        (
            _case(1, expected_positive=True, expected_outcome=ConsensusOutcome.ALERT),
            _case(2, expected_positive=False, expected_outcome=ConsensusOutcome.NO_ACTION),
            _case(3, expected_positive=True, expected_outcome=ConsensusOutcome.REVIEW),
            _case(4, expected_positive=False, expected_outcome=ConsensusOutcome.LOG),
        ),
        now=NOW,
    )


def _write_manifest(tmp_path: Path, manifest: EvaluationManifestDocument) -> Path:
    path = tmp_path / "evaluation-manifest.json"
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _response(
    index: int,
    outcome: ConsensusOutcome,
    *,
    receipt_suffix: str = "",
) -> EvaluationResponse:
    report_id = f"report-{index}{receipt_suffix}"
    audit_id = f"audit-{index}{receipt_suffix}"
    is_alert = outcome is ConsensusOutcome.ALERT
    return EvaluationResponse(
        clip_id=f"clip-{index}",
        processing_scope=EvaluationScope.ACOUSTIC_ONLY,
        outcome=outcome.value,
        risk_score={
            ConsensusOutcome.NO_ACTION: 0.0,
            ConsensusOutcome.LOG: 10.0,
            ConsensusOutcome.REVIEW: 40.0,
            ConsensusOutcome.ALERT: 90.0,
        }[outcome],
        risk_severity={
            ConsensusOutcome.NO_ACTION: "none",
            ConsensusOutcome.LOG: "low",
            ConsensusOutcome.REVIEW: "medium",
            ConsensusOutcome.ALERT: "critical",
        }[outcome],
        review_required=outcome in {ConsensusOutcome.REVIEW, ConsensusOutcome.ALERT},
        alert_candidate=is_alert,
        report_id=report_id,
        report_path=f"final-reports/{report_id}/report.json",
        report_reused=False,
        audit_id=audit_id,
        audit_path=f"alert-audit/{audit_id}/audit.json",
        audit_reused=False,
        alert_id=f"alert-{index}{receipt_suffix}" if is_alert else None,
        alert_path=f"alert-audit/{audit_id}/alert.json" if is_alert else None,
    )


def _completed(
    index: int,
    expected: bool,
    observed: bool,
    *,
    expected_outcome: ConsensusOutcome | None = None,
    observed_outcome: ConsensusOutcome | None = None,
) -> EvaluationCaseResult:
    outcome = observed_outcome or (
        ConsensusOutcome.REVIEW if observed else ConsensusOutcome.NO_ACTION
    )
    return EvaluationCaseResult(
        case_id=f"case-{index}",
        clip_id=f"clip-{index}",
        categories=(),
        expected_positive=expected,
        expected_outcome=expected_outcome,
        status="completed",
        observed_positive=observed,
        observed_outcome=outcome,
        risk_score=40.0 if observed else 0.0,
        risk_severity="medium" if observed else "none",
        report_id=f"report-{index}",
        audit_id=f"audit-{index}",
        alert_id=f"alert-{index}" if outcome is ConsensusOutcome.ALERT else None,
    )


def test_manifest_identity_is_repeatable_and_excludes_creation_time() -> None:
    cases = (_case(1, expected_positive=True, expected_outcome=ConsensusOutcome.REVIEW),)
    first = build_evaluation_manifest("repeatable", cases, now=NOW)
    second = build_evaluation_manifest(
        "repeatable",
        cases,
        now=datetime(2026, 9, 30, 12, tzinfo=UTC),
    )

    assert first.manifest_id == second.manifest_id
    assert first.created_at != second.created_at
    assert first.cases[0].request.audio_path == "evaluation/case-1.wav"


def test_manifest_rejects_duplicate_cases_and_inconsistent_truth() -> None:
    manifest = _manifest().model_dump(mode="python")
    manifest["cases"][1]["case_id"] = manifest["cases"][0]["case_id"]
    with pytest.raises(ValidationError, match="case IDs must be unique"):
        EvaluationManifestDocument.model_validate(manifest)

    inconsistent = _manifest().model_dump(mode="python")
    inconsistent["cases"][0]["expected_positive"] = False
    with pytest.raises(ValidationError, match="expected_positive"):
        EvaluationManifestDocument.model_validate(inconsistent)


def test_manifest_rejects_noncanonical_positive_outcomes() -> None:
    with pytest.raises(ValidationError, match="canonical outcome order"):
        build_evaluation_manifest(
            "bad order",
            (_case(1, expected_positive=True),),
            positive_outcomes=(ConsensusOutcome.ALERT, ConsensusOutcome.REVIEW),
            now=NOW,
        )


def test_loader_verifies_size_checksum_and_semantic_identity(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, _manifest())
    document = path.read_bytes()
    loaded = load_evaluation_manifest(
        path,
        expected_sha256=hashlib.sha256(document).hexdigest(),
    )
    assert loaded.manifest == _manifest()
    assert loaded.artifact_size_bytes == len(document)

    with pytest.raises(EvaluationManifestError) as checksum_error:
        load_evaluation_manifest(path, expected_sha256="0" * 64)
    assert checksum_error.value.code == "manifest_checksum_mismatch"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["name"] = "changed without a new identity"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationManifestError) as identity_error:
        load_evaluation_manifest(path)
    assert identity_error.value.code == "invalid_manifest"


def test_binary_metrics_cover_all_confusion_cells_and_exact_outcomes() -> None:
    cases = (
        _completed(
            1,
            True,
            True,
            expected_outcome=ConsensusOutcome.REVIEW,
            observed_outcome=ConsensusOutcome.REVIEW,
        ),
        _completed(2, False, True),
        _completed(3, True, False),
        _completed(
            4,
            False,
            False,
            expected_outcome=ConsensusOutcome.LOG,
            observed_outcome=ConsensusOutcome.NO_ACTION,
        ),
    )
    metrics = calculate_evaluation_metrics(cases)

    assert (
        metrics.binary.true_positive,
        metrics.binary.false_positive,
        metrics.binary.false_negative,
        metrics.binary.true_negative,
    ) == (1, 1, 1, 1)
    assert metrics.binary.accuracy == 0.5
    assert metrics.binary.precision == 0.5
    assert metrics.binary.recall == 0.5
    assert metrics.binary.specificity == 0.5
    assert metrics.binary.f1 == 0.5
    assert metrics.binary.balanced_accuracy == 0.5
    assert metrics.binary.false_positive_rate == 0.5
    assert metrics.binary.false_negative_rate == 0.5
    assert (metrics.outcome_labeled_count, metrics.outcome_match_count) == (2, 1)
    assert metrics.outcome_accuracy == 0.5


def test_failures_remain_visible_and_are_not_counted_as_predictions() -> None:
    cases = (
        _completed(1, True, True),
        EvaluationCaseResult(
            case_id="case-2",
            clip_id="clip-2",
            categories=(),
            expected_positive=False,
            status="failed",
            error_code="recording_not_found",
        ),
    )
    metrics = calculate_evaluation_metrics(cases)

    assert metrics.total_case_count == 2
    assert metrics.completed_case_count == 1
    assert metrics.failed_case_count == 1
    assert metrics.completion_rate == 0.5
    assert metrics.binary.evaluated_count == 1
    assert metrics.binary.true_positive == 1


def test_undefined_rates_are_null_instead_of_zero() -> None:
    metrics = calculate_evaluation_metrics((_completed(1, False, False),))

    assert metrics.binary.precision is None
    assert metrics.binary.recall is None
    assert metrics.binary.f1 is None
    assert metrics.binary.false_negative_rate is None
    assert metrics.binary.specificity == 1.0


def test_runner_uses_shared_request_boundary_and_redacts_failures(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    path = _write_manifest(tmp_path, manifest)
    loaded = load_evaluation_manifest(path)
    settings = AudioSentinelSettings.from_project_root(tmp_path)
    received: list[EvaluationRequest] = []

    def runner(
        supplied_settings: AudioSentinelSettings,
        request: EvaluationRequest,
    ) -> EvaluationResponse:
        assert supplied_settings is settings
        received.append(request)
        index = int(str(request.clip_id).split("-")[-1])
        if index == 4:
            raise RuntimeError("private transcript and C:\\secret\\recording.wav")
        outcomes = {
            1: ConsensusOutcome.ALERT,
            2: ConsensusOutcome.REVIEW,
            3: ConsensusOutcome.NO_ACTION,
        }
        return _response(index, outcomes[index])

    report = run_evaluation_manifest(settings, loaded, runner=runner, now=NOW)

    assert received == list(manifest.cases[index].request for index in range(4))
    assert report.decision_status == "incomplete_due_to_failures"
    assert report.metrics.failed_case_count == 1
    assert report.cases[-1].error_code == "unexpected_error"
    assert (report.metrics.binary.true_positive, report.metrics.binary.false_positive) == (1, 1)
    assert report.metrics.binary.false_negative == 1
    serialized = report.model_dump_json()
    assert "private transcript" not in serialized
    assert "secret" not in serialized
    assert "audio_path" not in serialized
    assert report.notification_delivery == "not_sent"
    assert report.alert_delivery_authorized is False


def test_run_identity_excludes_time_and_changes_with_results(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, _manifest())
    loaded = load_evaluation_manifest(path)
    settings = AudioSentinelSettings.from_project_root(tmp_path)

    def first_runner(
        _settings: AudioSentinelSettings,
        request: EvaluationRequest,
    ) -> EvaluationResponse:
        index = int(str(request.clip_id).split("-")[-1])
        return _response(index, ConsensusOutcome.ALERT if index in {1, 3} else ConsensusOutcome.NO_ACTION)

    first = run_evaluation_manifest(settings, loaded, runner=first_runner, now=NOW)
    repeated = run_evaluation_manifest(
        settings,
        loaded,
        runner=first_runner,
        now=datetime(2026, 9, 30, 12, tzinfo=UTC),
    )
    assert first.run_id == repeated.run_id
    assert first.created_at != repeated.created_at

    def changed_runner(
        _settings: AudioSentinelSettings,
        request: EvaluationRequest,
    ) -> EvaluationResponse:
        index = int(str(request.clip_id).split("-")[-1])
        return _response(index, ConsensusOutcome.NO_ACTION)

    changed = run_evaluation_manifest(settings, loaded, runner=changed_runner, now=NOW)
    assert changed.run_id != first.run_id


def test_run_identity_and_reuse_exclude_local_artifact_receipts(
    tmp_path: Path,
) -> None:
    path = _write_manifest(tmp_path, _manifest())
    loaded = load_evaluation_manifest(path)
    settings = AudioSentinelSettings.from_project_root(tmp_path)

    def runner_with_suffix(suffix: str):
        def runner(
            _settings: AudioSentinelSettings,
            request: EvaluationRequest,
        ) -> EvaluationResponse:
            index = int(str(request.clip_id).split("-")[-1])
            outcome = (
                ConsensusOutcome.ALERT
                if index in {1, 3}
                else ConsensusOutcome.NO_ACTION
            )
            return _response(index, outcome, receipt_suffix=suffix)

        return runner

    first = run_evaluation_manifest(
        settings,
        loaded,
        runner=runner_with_suffix("-first"),
        now=NOW,
    )
    repeated = run_evaluation_manifest(
        settings,
        loaded,
        runner=runner_with_suffix("-second"),
        now=datetime(2026, 9, 30, 12, tzinfo=UTC),
    )
    assert first.run_id == repeated.run_id
    assert first.cases[0].report_id != repeated.cases[0].report_id

    output = tmp_path / "evaluation" / "repeatable.json"
    assert save_evaluation_run(first, output) is False
    original = output.read_bytes()
    assert save_evaluation_run(repeated, output) is True
    assert output.read_bytes() == original


def test_save_run_is_valid_json_and_reuses_only_same_semantic_run(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, _manifest())
    loaded = load_evaluation_manifest(path)
    settings = AudioSentinelSettings.from_project_root(tmp_path)

    def runner(
        _settings: AudioSentinelSettings,
        request: EvaluationRequest,
    ) -> EvaluationResponse:
        index = int(str(request.clip_id).split("-")[-1])
        return _response(index, ConsensusOutcome.ALERT if index in {1, 3} else ConsensusOutcome.NO_ACTION)

    report = run_evaluation_manifest(settings, loaded, runner=runner, now=NOW)
    output = tmp_path / "evaluation" / "run.json"
    assert save_evaluation_run(report, output) is False
    saved = EvaluationRunDocument.model_validate_json(output.read_bytes())
    assert saved == report
    original = output.read_bytes()

    repeated = report.model_copy(update={"created_at": datetime(2026, 9, 30, tzinfo=UTC)})
    assert save_evaluation_run(repeated, output) is True
    assert output.read_bytes() == original

    conflicting = output.parent / "conflict.json"
    conflicting.write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluationManifestError) as caught:
        save_evaluation_run(report, conflicting)
    assert caught.value.code == "output_conflict"


def test_checked_in_schemas_match_generated_documents(project_root: Path) -> None:
    generated = evaluation_schema_documents()
    schema_root = project_root / "docs" / "schemas" / "v1"
    for filename, document in generated.items():
        checked_in = json.loads((schema_root / filename).read_text(encoding="utf-8"))
        assert checked_in == document


def test_public_contracts_are_strict_and_immutable() -> None:
    manifest = _manifest()
    with pytest.raises(ValidationError):
        EvaluationManifestDocument.model_validate(
            {**manifest.model_dump(mode="python"), "unknown": True}
        )
    with pytest.raises(ValidationError):
        manifest.name = "changed"
