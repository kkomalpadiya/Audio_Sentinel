"""A10.3 tests for live latency, reliability, and alert timing."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from audio_sentinel.live_performance import (
    COMPLETE_LIVE_PERFORMANCE_COVERAGE,
    LiveLatencyKind,
    LiveLatencyObservation,
    LivePerformanceError,
    LivePerformanceReport,
    LiveReliabilityCounters,
    LiveRuntimeDescriptor,
    build_live_performance_report,
    live_performance_schema_document,
    load_live_performance_report,
    run_in_process_live_benchmark,
    save_live_performance_report,
    summarize_latency,
    summarize_reliability,
)


NOW = datetime(2026, 10, 3, 12, tzinfo=UTC)
RUNTIME = LiveRuntimeDescriptor(
    python_implementation="CPython",
    python_version="3.12.0",
    operating_system="TestOS",
    machine="test-machine",
)


def observations() -> tuple[LiveLatencyObservation, ...]:
    return (
        LiveLatencyObservation(
            observation_id="connection:1",
            kind=LiveLatencyKind.CONNECTION_OPEN,
            elapsed_ms=4,
        ),
        LiveLatencyObservation(
            observation_id="enqueue:0",
            kind=LiveLatencyKind.BUFFER_ENQUEUE,
            elapsed_ms=1,
            sequence_number=0,
        ),
        LiveLatencyObservation(
            observation_id="enqueue:1",
            kind=LiveLatencyKind.BUFFER_ENQUEUE,
            elapsed_ms=3,
            sequence_number=1,
        ),
        LiveLatencyObservation(
            observation_id="acceptance:0",
            kind=LiveLatencyKind.CAPTURE_TO_ACCEPTANCE,
            elapsed_ms=5,
            sequence_number=0,
            attempt_number=2,
        ),
        LiveLatencyObservation(
            observation_id="acceptance:1",
            kind=LiveLatencyKind.CAPTURE_TO_ACCEPTANCE,
            elapsed_ms=7,
            sequence_number=1,
        ),
        LiveLatencyObservation(
            observation_id="window:1",
            kind=LiveLatencyKind.CAPTURE_TO_WINDOW,
            elapsed_ms=8,
            sequence_number=1,
            window_id="window-1",
        ),
        LiveLatencyObservation(
            observation_id="decision:1",
            kind=LiveLatencyKind.CAPTURE_TO_DECISION,
            elapsed_ms=9,
            sequence_number=1,
            window_id="window-1",
        ),
        LiveLatencyObservation(
            observation_id="alert:1",
        kind=LiveLatencyKind.CAPTURE_TO_LOCAL_ALERT_PROBE,
            elapsed_ms=10,
            sequence_number=1,
            window_id="window-1",
        ),
    )


def counters() -> LiveReliabilityCounters:
    return LiveReliabilityCounters(
        chunks_enqueued=2,
        delivery_attempts=3,
        chunks_accepted=2,
        deferred_deliveries=1,
        failed_deliveries=0,
        backpressure_rejections=1,
        chunks_discarded=0,
        chunks_queued_at_end=0,
        duplicate_acceptances=0,
        sequence_gaps=0,
        windows_emitted=1,
        windows_processed=1,
        window_processing_failures=0,
        alert_candidates=1,
        local_alert_probe_completions=1,
    )


def report() -> LivePerformanceReport:
    return build_live_performance_report(
        observations(),
        counters(),
        run_duration_ms=12,
        created_at=NOW,
        runtime=RUNTIME,
    )


def test_latency_summary_uses_documented_linear_percentiles() -> None:
    summary = summarize_latency(observations(), LiveLatencyKind.BUFFER_ENQUEUE)

    assert summary.sample_count == 2
    assert summary.minimum_ms == 1
    assert summary.median_ms == 2
    assert summary.mean_ms == 2
    assert summary.p95_ms == 2.9
    assert summary.maximum_ms == 3
    assert summary.percentile_method == "linear_interpolation"


def test_reliability_rates_keep_explicit_denominators() -> None:
    summary = summarize_reliability(counters())

    assert summary.chunk_delivery_rate == 1
    assert summary.delivery_attempt_success_rate == 0.666667
    assert summary.window_processing_rate == 1
    assert summary.alert_probe_completion_rate == 1
    assert summary.loss_rate == 0


def test_zero_denominators_remain_unavailable() -> None:
    empty = LiveReliabilityCounters(
        chunks_enqueued=0,
        delivery_attempts=0,
        chunks_accepted=0,
        deferred_deliveries=0,
        failed_deliveries=0,
        backpressure_rejections=0,
        chunks_discarded=0,
        chunks_queued_at_end=0,
        duplicate_acceptances=0,
        sequence_gaps=0,
        windows_emitted=0,
        windows_processed=0,
        window_processing_failures=0,
        alert_candidates=0,
        local_alert_probe_completions=0,
    )

    summary = summarize_reliability(empty)

    assert summary.chunk_delivery_rate is None
    assert summary.delivery_attempt_success_rate is None
    assert summary.window_processing_rate is None
    assert summary.alert_probe_completion_rate is None
    assert summary.loss_rate is None


@pytest.mark.parametrize(
    "updates",
    [
        {"delivery_attempts": 2},
        {"chunks_enqueued": 3},
        {"windows_emitted": 2},
        {"alert_candidates": 2},
        {"local_alert_probe_completions": 2},
    ],
)
def test_reliability_counters_must_reconcile(updates: dict[str, int]) -> None:
    values = counters().model_dump()
    values.update(updates)

    with pytest.raises(ValidationError):
        LiveReliabilityCounters(**values)


@pytest.mark.parametrize(
    "kind,sequence_number,window_id",
    [
        (LiveLatencyKind.CONNECTION_OPEN, 0, None),
        (LiveLatencyKind.BUFFER_ENQUEUE, None, None),
        (LiveLatencyKind.CAPTURE_TO_ACCEPTANCE, 0, "window-1"),
        (LiveLatencyKind.CAPTURE_TO_WINDOW, 0, None),
        (LiveLatencyKind.CAPTURE_TO_DECISION, None, "window-1"),
        (LiveLatencyKind.CAPTURE_TO_LOCAL_ALERT_PROBE, None, None),
    ],
)
def test_observation_identity_matches_stage(
    kind: LiveLatencyKind,
    sequence_number: int | None,
    window_id: str | None,
) -> None:
    with pytest.raises(ValidationError):
        LiveLatencyObservation(
            observation_id="invalid-observation",
            kind=kind,
            elapsed_ms=1,
            sequence_number=sequence_number,
            window_id=window_id,
        )


def test_report_is_self_validating_and_audio_free() -> None:
    result = report()
    document = result.model_dump_json()

    assert result.coverage == COMPLETE_LIVE_PERFORMANCE_COVERAGE
    assert result.report_id.startswith("live-performance-")
    assert result.notification_delivery == "not_sent"
    assert result.alert_delivery_authorized is False
    assert result.external_action_authorized is False
    assert result.raw_audio_persisted is False
    assert '"payload"' not in document
    assert "credential" not in document
    assert "consent_id" not in document
    assert "device_id" not in document
    assert "session_id" not in document


def test_report_rejects_tampered_identity_and_summaries() -> None:
    valid = report()
    bad_identity = valid.model_dump(mode="python")
    bad_identity["report_id"] = "live-performance-" + "0" * 64
    with pytest.raises(ValidationError):
        LivePerformanceReport.model_validate(bad_identity)

    bad_latency = valid.model_dump(mode="python")
    bad_latency["latency"][0]["maximum_ms"] = 99
    with pytest.raises(ValidationError):
        LivePerformanceReport.model_validate(bad_latency)


def test_report_save_load_and_explicit_replacement(tmp_path: Path) -> None:
    destination = tmp_path / "live-performance.json"
    original = report()

    save_live_performance_report(original, destination)
    assert load_live_performance_report(destination) == original
    with pytest.raises(LivePerformanceError) as captured:
        save_live_performance_report(original, destination)
    assert captured.value.code == "output_exists"

    save_live_performance_report(original, destination, replace=True)
    assert load_live_performance_report(destination) == original


def test_invalid_saved_report_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(LivePerformanceError) as captured:
        load_live_performance_report(path)
    assert captured.value.code == "invalid_report"


def test_real_in_process_benchmark_covers_retry_backpressure_and_alert_timing() -> None:
    measured = run_in_process_live_benchmark(created_at=NOW)
    counts = measured.reliability.counters
    distributions = {item.kind: item for item in measured.latency}

    assert counts.chunks_enqueued == counts.chunks_accepted == 12
    assert counts.delivery_attempts == 13
    assert counts.deferred_deliveries == 1
    assert counts.backpressure_rejections == 1
    assert counts.chunks_discarded == counts.chunks_queued_at_end == 0
    assert counts.duplicate_acceptances == counts.sequence_gaps == 0
    assert counts.windows_emitted == counts.windows_processed > 0
    assert counts.window_processing_failures == 0
    assert counts.alert_candidates == counts.local_alert_probe_completions == 1
    assert measured.reliability.chunk_delivery_rate == 1
    assert measured.reliability.loss_rate == 0
    assert all(item.sample_count > 0 for item in distributions.values())


def test_checked_in_schema_matches_runtime_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_in = json.loads(
        (root / "docs/schemas/v1/live-performance-report.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert checked_in == live_performance_schema_document()


def test_measurement_script_writes_reloadable_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "measured.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/measure_live_performance.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    summary = json.loads(completed.stdout)
    loaded = load_live_performance_report(output)
    assert summary["report_id"] == loaded.report_id
    assert summary["notification_delivery"] == "not_sent"


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
from audio_sentinel.live_performance import LivePerformanceReport
assert LivePerformanceReport.__name__ == 'LivePerformanceReport'
"""
    environment = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        capture_output=True,
        timeout=30,
    )
