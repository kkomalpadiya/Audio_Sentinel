"""A10.3 latency, reliability, and local-alert timing measurements."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.config import Paths
from audio_sentinel.contracts import ConsentRecord, ConsentStatus, ProcessingScope
from audio_sentinel.live_streaming import (
    AuthorizedStreamSession,
    DeviceEnrollment,
    EnrollmentStatus,
    StreamAudioChunk,
    StreamAudioFormat,
    StreamChunkReceipt,
    StreamCloseReceipt,
)
from audio_sentinel.live_windows import (
    LiveAudioWindow,
    LiveWindowCloseSummary,
    RollingLiveWindowSink,
    RollingWindowSettings,
)
from audio_sentinel.stream_resilience import (
    BackpressureMode,
    BufferedStreamingClient,
    StreamBufferSettings,
    StreamResilienceError,
)
from audio_sentinel.streaming_ingestion import (
    FileDeviceEnrollmentRegistry,
    LocalDeviceAuthenticator,
    LocalStreamingClient,
    LocalStreamingIngestionService,
)


LIVE_PERFORMANCE_SCHEMA_VERSION = "1.0"
LIVE_PERFORMANCE_FORMAT_VERSION = "1.0"
MAX_LIVE_PERFORMANCE_BYTES = 4_194_304
MAX_LIVE_PERFORMANCE_OBSERVATIONS = 100_000


class LivePerformanceError(RuntimeError):
    """Stable A10.3 failure with a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class LiveLatencyKind(str, Enum):
    CONNECTION_OPEN = "connection_open"
    BUFFER_ENQUEUE = "buffer_enqueue"
    CAPTURE_TO_ACCEPTANCE = "capture_to_acceptance"
    CAPTURE_TO_WINDOW = "capture_to_window"
    CAPTURE_TO_DECISION = "capture_to_decision"
    CAPTURE_TO_LOCAL_ALERT_PROBE = "capture_to_local_alert_probe"


class LivePerformanceCoverage(str, Enum):
    AUTHENTICATED_STREAM_OPEN = "authenticated_stream_open"
    BOUNDED_BUFFERING = "bounded_buffering"
    DEFERRED_DELIVERY_RETRY = "deferred_delivery_retry"
    BACKPRESSURE_REJECTION = "backpressure_rejection"
    ROLLING_WINDOWS = "rolling_windows"
    DETERMINISTIC_LOCAL_ALERT_PROBE = "deterministic_local_alert_probe"


COMPLETE_LIVE_PERFORMANCE_COVERAGE = tuple(LivePerformanceCoverage)


class LivePerformanceRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class LiveLatencyObservation(LivePerformanceRecord):
    """One audio-free duration measured with a monotonic clock."""

    observation_id: str = Field(
        min_length=3,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    kind: LiveLatencyKind
    elapsed_ms: float = Field(ge=0, le=600_000)
    sequence_number: int | None = Field(default=None, ge=0, strict=True)
    window_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    attempt_number: int = Field(default=1, ge=1, strict=True)
    clock: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"
    raw_audio_persisted: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity(self) -> "LiveLatencyObservation":
        chunk_kinds = {
            LiveLatencyKind.BUFFER_ENQUEUE,
            LiveLatencyKind.CAPTURE_TO_ACCEPTANCE,
        }
        window_kinds = {
            LiveLatencyKind.CAPTURE_TO_WINDOW,
            LiveLatencyKind.CAPTURE_TO_DECISION,
            LiveLatencyKind.CAPTURE_TO_LOCAL_ALERT_PROBE,
        }
        if self.kind is LiveLatencyKind.CONNECTION_OPEN:
            if self.sequence_number is not None or self.window_id is not None:
                raise ValueError("connection timing cannot identify a chunk or window")
        elif self.kind in chunk_kinds:
            if self.sequence_number is None or self.window_id is not None:
                raise ValueError("chunk timing requires only sequence_number")
        elif self.kind in window_kinds:
            if self.sequence_number is None or self.window_id is None:
                raise ValueError("window timing requires sequence_number and window_id")
        return self


class LiveLatencyDistribution(LivePerformanceRecord):
    """A duration distribution with explicit count and percentile method."""

    kind: LiveLatencyKind
    sample_count: int = Field(ge=0, strict=True)
    minimum_ms: float | None = Field(default=None, ge=0)
    median_ms: float | None = Field(default=None, ge=0)
    mean_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)
    maximum_ms: float | None = Field(default=None, ge=0)
    percentile_method: Literal["linear_interpolation"] = "linear_interpolation"

    @model_validator(mode="after")
    def validate_distribution(self) -> "LiveLatencyDistribution":
        values = (
            self.minimum_ms,
            self.median_ms,
            self.mean_ms,
            self.p95_ms,
            self.maximum_ms,
        )
        if self.sample_count == 0:
            if any(value is not None for value in values):
                raise ValueError("empty distributions cannot contain latency values")
            return self
        if any(value is None for value in values):
            raise ValueError("nonempty distributions require all latency values")
        assert self.minimum_ms is not None
        assert self.median_ms is not None
        assert self.p95_ms is not None
        assert self.maximum_ms is not None
        if not (
            self.minimum_ms
            <= self.median_ms
            <= self.p95_ms
            <= self.maximum_ms
        ):
            raise ValueError("latency percentiles must be nondecreasing")
        return self


class LiveReliabilityCounters(LivePerformanceRecord):
    """Exact event counts; rejected chunks are outside the enqueue denominator."""

    chunks_enqueued: int = Field(ge=0, strict=True)
    delivery_attempts: int = Field(ge=0, strict=True)
    chunks_accepted: int = Field(ge=0, strict=True)
    deferred_deliveries: int = Field(ge=0, strict=True)
    failed_deliveries: int = Field(ge=0, strict=True)
    backpressure_rejections: int = Field(ge=0, strict=True)
    chunks_discarded: int = Field(ge=0, strict=True)
    chunks_queued_at_end: int = Field(ge=0, strict=True)
    duplicate_acceptances: int = Field(ge=0, strict=True)
    sequence_gaps: int = Field(ge=0, strict=True)
    windows_emitted: int = Field(ge=0, strict=True)
    windows_processed: int = Field(ge=0, strict=True)
    window_processing_failures: int = Field(ge=0, strict=True)
    alert_candidates: int = Field(ge=0, strict=True)
    local_alert_probe_completions: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_accounting(self) -> "LiveReliabilityCounters":
        if self.delivery_attempts != (
            self.chunks_accepted
            + self.deferred_deliveries
            + self.failed_deliveries
        ):
            raise ValueError("delivery attempts must reconcile to all outcomes")
        if self.chunks_enqueued != (
            self.chunks_accepted
            + self.chunks_discarded
            + self.chunks_queued_at_end
        ):
            raise ValueError("enqueued chunks must have an explicit final disposition")
        if self.windows_emitted != (
            self.windows_processed + self.window_processing_failures
        ):
            raise ValueError("emitted windows must have an explicit processing outcome")
        if self.alert_candidates > self.windows_processed:
            raise ValueError("alert candidates cannot exceed processed windows")
        if self.local_alert_probe_completions > self.alert_candidates:
            raise ValueError("alert probe completions cannot exceed alert candidates")
        return self


class LiveReliabilitySummary(LivePerformanceRecord):
    counters: LiveReliabilityCounters
    chunk_delivery_rate: float | None = Field(default=None, ge=0, le=1)
    delivery_attempt_success_rate: float | None = Field(default=None, ge=0, le=1)
    window_processing_rate: float | None = Field(default=None, ge=0, le=1)
    alert_probe_completion_rate: float | None = Field(default=None, ge=0, le=1)
    loss_rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_rates(self) -> "LiveReliabilitySummary":
        expected = summarize_reliability(self.counters)
        for name in (
            "chunk_delivery_rate",
            "delivery_attempt_success_rate",
            "window_processing_rate",
            "alert_probe_completion_rate",
            "loss_rate",
        ):
            if getattr(self, name) != getattr(expected, name):
                raise ValueError(f"{name} does not match the explicit counters")
        return self


class LiveBenchmarkDefinition(LivePerformanceRecord):
    """Public, identity-free description of the accelerated benchmark."""

    benchmark_id: Literal["a10.3-in-process-live-path-v1"] = (
        "a10.3-in-process-live-path-v1"
    )
    measurement_scope: Literal["accelerated_in_process_synthetic_pcm"] = (
        "accelerated_in_process_synthetic_pcm"
    )
    sample_rate_hz: int = Field(default=16_000, gt=0, strict=True)
    channels: Literal[1] = 1
    sample_width_bytes: Literal[2] = 2
    chunk_samples: int = Field(default=160, gt=0, strict=True)
    chunk_count: int = Field(default=12, gt=0, strict=True)
    window_seconds: tuple[float, ...] = (0.04,)
    window_overlap_ratio: float = Field(default=0.5, ge=0, lt=1)
    max_buffered_chunks: int = Field(default=4, gt=0, strict=True)
    backpressure_mode: Literal["reject"] = "reject"
    forced_deferred_deliveries: Literal[1] = 1
    deterministic_alert_probe_count: Literal[1] = 1

    @field_validator("window_seconds")
    @classmethod
    def validate_windows(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if (
            not values
            or tuple(sorted(values)) != values
            or len(set(values)) != len(values)
            or any(not math.isfinite(value) or value <= 0 for value in values)
        ):
            raise ValueError("window_seconds must be finite, positive, unique, and ordered")
        return values


class LiveRuntimeDescriptor(LivePerformanceRecord):
    python_implementation: str = Field(min_length=1, max_length=32)
    python_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    operating_system: str = Field(min_length=1, max_length=64)
    machine: str = Field(min_length=1, max_length=64)


class LivePerformanceReport(LivePerformanceRecord):
    """Audio-free result for one A10.3 in-process measurement run."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": (
                "https://audio-sentinel.local/schemas/v1/"
                "live-performance-report.schema.json"
            )
        },
    )

    schema_version: Literal["1.0"] = LIVE_PERFORMANCE_SCHEMA_VERSION
    document_type: Literal["live_performance_report"] = "live_performance_report"
    format_version: Literal["1.0"] = LIVE_PERFORMANCE_FORMAT_VERSION
    report_id: str = Field(pattern=r"^live-performance-[a-f0-9]{64}$")
    created_at: datetime
    measurement_status: Literal["completed"] = "completed"
    benchmark: LiveBenchmarkDefinition
    runtime: LiveRuntimeDescriptor
    run_duration_ms: float = Field(ge=0, le=600_000)
    coverage: tuple[LivePerformanceCoverage, ...]
    observations: tuple[LiveLatencyObservation, ...] = Field(
        max_length=MAX_LIVE_PERFORMANCE_OBSERVATIONS
    )
    latency: tuple[LiveLatencyDistribution, ...]
    reliability: LiveReliabilitySummary
    interpretation: Literal["regression_measurement_not_deployment_approval"] = (
        "regression_measurement_not_deployment_approval"
    )
    hardware_capture_measured: Literal[False] = False
    network_transport_measured: Literal[False] = False
    model_inference_measured: Literal[False] = False
    alert_effectiveness_measured: Literal[False] = False
    notification_delivery: Literal["not_sent"] = "not_sent"
    alert_delivery_authorized: Literal[False] = False
    external_action_authorized: Literal[False] = False
    raw_audio_persisted: Literal[False] = False

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> "LivePerformanceReport":
        if self.coverage != COMPLETE_LIVE_PERFORMANCE_COVERAGE:
            raise ValueError("completed reports require the complete A10.3 coverage set")
        if len({item.observation_id for item in self.observations}) != len(
            self.observations
        ):
            raise ValueError("observation identities must be unique")
        expected_latency = tuple(
            summarize_latency(self.observations, kind) for kind in LiveLatencyKind
        )
        if self.latency != expected_latency:
            raise ValueError("latency summaries do not match their observations")
        if self.reliability != summarize_reliability(
            self.reliability.counters
        ):
            raise ValueError("reliability summary does not match its counters")
        counts = self.reliability.counters
        by_kind = {
            kind: sum(item.kind is kind for item in self.observations)
            for kind in LiveLatencyKind
        }
        if by_kind[LiveLatencyKind.CONNECTION_OPEN] != 1:
            raise ValueError("completed reports require one connection measurement")
        if by_kind[LiveLatencyKind.BUFFER_ENQUEUE] != counts.chunks_enqueued:
            raise ValueError("enqueue observations must match enqueued chunks")
        if by_kind[LiveLatencyKind.CAPTURE_TO_ACCEPTANCE] != counts.chunks_accepted:
            raise ValueError("acceptance observations must match accepted chunks")
        if by_kind[LiveLatencyKind.CAPTURE_TO_WINDOW] != counts.windows_emitted:
            raise ValueError("window observations must match emitted windows")
        if by_kind[LiveLatencyKind.CAPTURE_TO_DECISION] != counts.windows_processed:
            raise ValueError("decision observations must match processed windows")
        if (
            by_kind[LiveLatencyKind.CAPTURE_TO_LOCAL_ALERT_PROBE]
            != counts.local_alert_probe_completions
        ):
            raise ValueError("alert observations must match completed alert probes")
        if self.report_id != _report_id(self):
            raise ValueError("report_id must match canonical report content")
        return self


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _report_id(report: LivePerformanceReport) -> str:
    payload = report.model_dump(mode="json", exclude={"report_id"})
    return f"live-performance-{_canonical_hash(payload)}"


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        raise ValueError("a percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latency(
    observations: tuple[LiveLatencyObservation, ...],
    kind: LiveLatencyKind,
) -> LiveLatencyDistribution:
    values = tuple(item.elapsed_ms for item in observations if item.kind is kind)
    if not values:
        return LiveLatencyDistribution(kind=kind, sample_count=0)
    return LiveLatencyDistribution(
        kind=kind,
        sample_count=len(values),
        minimum_ms=_rounded(min(values)),
        median_ms=_rounded(_percentile(values, 0.5)),
        mean_ms=_rounded(sum(values) / len(values)),
        p95_ms=_rounded(_percentile(values, 0.95)),
        maximum_ms=_rounded(max(values)),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else _rounded(numerator / denominator)


def summarize_reliability(
    counters: LiveReliabilityCounters,
) -> LiveReliabilitySummary:
    if not isinstance(counters, LiveReliabilityCounters):
        raise TypeError("counters must be LiveReliabilityCounters")
    return LiveReliabilitySummary.model_construct(
        counters=counters,
        chunk_delivery_rate=_rate(counters.chunks_accepted, counters.chunks_enqueued),
        delivery_attempt_success_rate=_rate(
            counters.chunks_accepted, counters.delivery_attempts
        ),
        window_processing_rate=_rate(
            counters.windows_processed, counters.windows_emitted
        ),
        alert_probe_completion_rate=_rate(
            counters.local_alert_probe_completions, counters.alert_candidates
        ),
        loss_rate=_rate(counters.chunks_discarded, counters.chunks_enqueued),
    )


def build_live_performance_report(
    observations: tuple[LiveLatencyObservation, ...],
    counters: LiveReliabilityCounters,
    *,
    benchmark: LiveBenchmarkDefinition | None = None,
    runtime: LiveRuntimeDescriptor | None = None,
    run_duration_ms: float,
    created_at: datetime | None = None,
) -> LivePerformanceReport:
    """Build a self-validating report from monotonic durations and exact counts."""

    try:
        validated_observations = tuple(
            LiveLatencyObservation.model_validate(item.model_dump(mode="python"))
            for item in observations
        )
        validated_counters = LiveReliabilityCounters.model_validate(
            counters.model_dump(mode="python")
        )
        created = created_at or datetime.now(UTC)
        if created.tzinfo is None or created.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        descriptor = runtime or LiveRuntimeDescriptor(
            python_implementation=platform.python_implementation(),
            python_version=platform.python_version(),
            operating_system=platform.system() or os.name,
            machine=platform.machine() or "unknown",
        )
        provisional = LivePerformanceReport.model_construct(
            schema_version=LIVE_PERFORMANCE_SCHEMA_VERSION,
            document_type="live_performance_report",
            format_version=LIVE_PERFORMANCE_FORMAT_VERSION,
            report_id="live-performance-" + "0" * 64,
            created_at=created.astimezone(UTC),
            measurement_status="completed",
            benchmark=benchmark or LiveBenchmarkDefinition(),
            runtime=descriptor,
            run_duration_ms=_rounded(run_duration_ms),
            coverage=COMPLETE_LIVE_PERFORMANCE_COVERAGE,
            observations=validated_observations,
            latency=tuple(
                summarize_latency(validated_observations, kind)
                for kind in LiveLatencyKind
            ),
            reliability=summarize_reliability(validated_counters),
            interpretation="regression_measurement_not_deployment_approval",
            hardware_capture_measured=False,
            network_transport_measured=False,
            model_inference_measured=False,
            alert_effectiveness_measured=False,
            notification_delivery="not_sent",
            alert_delivery_authorized=False,
            external_action_authorized=False,
            raw_audio_persisted=False,
        )
        return LivePerformanceReport.model_validate(
            provisional.model_copy(
                update={"report_id": _report_id(provisional)}
            ).model_dump(mode="python")
        )
    except (TypeError, ValueError) as error:
        raise LivePerformanceError(
            "invalid_measurement",
            "Live performance observations or counters were inconsistent.",
        ) from error


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    if end_ns < start_ns:
        raise LivePerformanceError(
            "clock_invalid", "The monotonic measurement clock moved backward."
        )
    return _rounded((end_ns - start_ns) / 1_000_000)


class _MeasuredWindowConsumer:
    def __init__(
        self,
        observations: list[LiveLatencyObservation],
        capture_started_ns: dict[int, int],
    ) -> None:
        self.observations = observations
        self.capture_started_ns = capture_started_ns
        self.window_ids: list[str] = []
        self.alert_ids: list[str] = []
        self.close_summary: LiveWindowCloseSummary | None = None

    def accept_window(
        self,
        session: AuthorizedStreamSession,
        window: LiveAudioWindow,
    ) -> None:
        del session
        sequence = window.record.emitted_after_sequence_number
        started_ns = self.capture_started_ns[sequence]
        emitted_ns = time.perf_counter_ns()
        public_window_id = f"window-{len(self.window_ids):06d}"
        self.observations.append(
            LiveLatencyObservation(
                observation_id=f"window:{len(self.window_ids):06d}",
                kind=LiveLatencyKind.CAPTURE_TO_WINDOW,
                elapsed_ms=_elapsed_ms(started_ns, emitted_ns),
                sequence_number=sequence,
                window_id=public_window_id,
            )
        )
        decision_payload = {
            "window_id": public_window_id,
            "start_sample": window.record.start_sample,
            "end_sample": window.record.end_sample,
            "probe_outcome": (
                "alert" if not self.alert_ids else "no_action"
            ),
        }
        decision_id = f"decision-{_canonical_hash(decision_payload)}"
        decision_ns = time.perf_counter_ns()
        self.observations.append(
            LiveLatencyObservation(
                observation_id=f"decision:{len(self.window_ids):06d}",
                kind=LiveLatencyKind.CAPTURE_TO_DECISION,
                elapsed_ms=_elapsed_ms(started_ns, decision_ns),
                sequence_number=sequence,
                window_id=public_window_id,
            )
        )
        if not self.alert_ids:
            alert_id = f"local-alert-probe-{_canonical_hash({'decision_id': decision_id})}"
            self.alert_ids.append(alert_id)
            alert_ns = time.perf_counter_ns()
            self.observations.append(
                LiveLatencyObservation(
                    observation_id=f"alert:{len(self.window_ids):06d}",
                    kind=LiveLatencyKind.CAPTURE_TO_LOCAL_ALERT_PROBE,
                    elapsed_ms=_elapsed_ms(started_ns, alert_ns),
                    sequence_number=sequence,
                    window_id=public_window_id,
                )
            )
        self.window_ids.append(public_window_id)

    def close_live_windows(
        self,
        session: AuthorizedStreamSession,
        receipt: StreamCloseReceipt,
        summary: LiveWindowCloseSummary,
    ) -> None:
        if session.session_id != receipt.session_id or receipt.session_id != summary.session_id:
            raise LivePerformanceError(
                "close_mismatch", "Live-window close accounting did not match the session."
            )
        self.close_summary = summary


class _FailFirstChunkSink:
    """One deterministic downstream refusal for retry measurement."""

    def __init__(self, downstream: RollingLiveWindowSink) -> None:
        self.downstream = downstream
        self.remaining_failures = 1

    def accept_chunk(
        self,
        session: AuthorizedStreamSession,
        chunk: StreamAudioChunk,
        receipt: StreamChunkReceipt,
    ) -> None:
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise RuntimeError("deterministic deferred-delivery probe")
        self.downstream.accept_chunk(session, chunk, receipt)

    def close_stream(
        self,
        session: AuthorizedStreamSession,
        receipt: StreamCloseReceipt,
    ) -> None:
        self.downstream.close_stream(session, receipt)


def run_in_process_live_benchmark(
    *,
    created_at: datetime | None = None,
) -> LivePerformanceReport:
    """Exercise A10.1-B10.2 in process with synthetic PCM and no persistence."""

    report_time = created_at or datetime.now(UTC)
    if report_time.tzinfo is None or report_time.utcoffset() is None:
        raise LivePerformanceError(
            "invalid_time", "Benchmark time must include a timezone."
        )
    report_time = report_time.astimezone(UTC)
    authorization_time = datetime.now(UTC)
    benchmark = LiveBenchmarkDefinition()
    observations: list[LiveLatencyObservation] = []
    capture_started_ns: dict[int, int] = {}
    pending_sequences: deque[int] = deque()
    attempt_counts = {index: 0 for index in range(benchmark.chunk_count)}
    rejected_by_backpressure = 0
    run_started_ns = time.perf_counter_ns()

    with tempfile.TemporaryDirectory(prefix="audio-sentinel-a10-3-") as directory:
        paths = Paths.from_root(Path(directory))
        paths.ensure_directories()
        secret = b"a10-3-in-process-benchmark-credential"
        audio_format = StreamAudioFormat(
            sample_rate_hz=benchmark.sample_rate_hz,
            channels=benchmark.channels,
            chunk_duration_ms=round(
                benchmark.chunk_samples * 1000 / benchmark.sample_rate_hz
            ),
        )
        enrollment = DeviceEnrollment(
            enrollment_id="enrollment-a10-3-benchmark",
            device_id="device-a10-3-benchmark",
            status=EnrollmentStatus.ACTIVE,
            authorization_reference="a10-3-local-regression-measurement",
            credential_fingerprint_sha256=hashlib.sha256(secret).hexdigest(),
            authorized_scopes=(ProcessingScope.ACOUSTIC_ONLY,),
            audio_formats=(audio_format,),
            enrolled_at=authorization_time - timedelta(minutes=1),
            expires_at=authorization_time + timedelta(hours=1),
        )
        consent = ConsentRecord(
            consent_id="consent-a10-3-benchmark",
            status=ConsentStatus.GRANTED,
            processing_scope=ProcessingScope.ACOUSTIC_ONLY,
            device_authorized=True,
            granted_at=authorization_time - timedelta(minutes=1),
            expires_at=authorization_time + timedelta(hours=1),
        )
        registry = FileDeviceEnrollmentRegistry(paths)
        registry.register(enrollment)
        authenticator = LocalDeviceAuthenticator(lambda _record: secret)
        collector = _MeasuredWindowConsumer(observations, capture_started_ns)
        windows = RollingLiveWindowSink(
            collector,
            settings=RollingWindowSettings(
                target_sample_rate_hz=benchmark.sample_rate_hz,
                window_seconds=benchmark.window_seconds,
                window_overlap_ratio=benchmark.window_overlap_ratio,
                close_tail_policy="pad",
            ),
        )
        deferred_sink = _FailFirstChunkSink(windows)
        service = LocalStreamingIngestionService(
            registry,
            authenticator,
            deferred_sink,
        )

        def factory() -> LocalStreamingClient:
            return LocalStreamingClient(
                service,
                enrollment_id=enrollment.enrollment_id,
                device_id=enrollment.device_id,
                credential_provider=lambda: secret,
                session_id_factory=lambda: "session-a10-3-benchmark",
            )

        buffered = BufferedStreamingClient(
            factory,
            consent=consent,
            requested_scope=ProcessingScope.ACOUSTIC_ONLY,
            audio_format=audio_format,
            settings=StreamBufferSettings(
                max_buffered_chunks=benchmark.max_buffered_chunks,
                max_buffered_bytes=(
                    benchmark.max_buffered_chunks
                    * benchmark.chunk_samples
                    * benchmark.channels
                    * benchmark.sample_width_bytes
                ),
                backpressure_mode=BackpressureMode.REJECT,
            ),
        )
        connect_started_ns = time.perf_counter_ns()
        buffered.connect()
        observations.append(
            LiveLatencyObservation(
                observation_id="connection:000001",
                kind=LiveLatencyKind.CONNECTION_OPEN,
                elapsed_ms=_elapsed_ms(
                    connect_started_ns, time.perf_counter_ns()
                ),
            )
        )

        def enqueue(sequence_number: int) -> None:
            started_ns = time.perf_counter_ns()
            captured_at = datetime.now(UTC)
            value = sequence_number.to_bytes(2, "little", signed=False)
            buffered.enqueue_pcm(
                value * benchmark.chunk_samples,
                captured_at=captured_at,
            )
            completed_ns = time.perf_counter_ns()
            capture_started_ns[sequence_number] = started_ns
            pending_sequences.append(sequence_number)
            observations.append(
                LiveLatencyObservation(
                    observation_id=f"enqueue:{sequence_number:06d}",
                    kind=LiveLatencyKind.BUFFER_ENQUEUE,
                    elapsed_ms=_elapsed_ms(started_ns, completed_ns),
                    sequence_number=sequence_number,
                )
            )

        def flush(*, expect_deferred: bool = False) -> None:
            if not pending_sequences:
                return
            snapshot = tuple(pending_sequences)
            if expect_deferred:
                attempt_counts[snapshot[0]] += 1
                try:
                    buffered.flush()
                except StreamResilienceError as error:
                    if error.code != "delivery_deferred":
                        raise
                else:
                    raise LivePerformanceError(
                        "retry_probe_failed",
                        "The benchmark retry probe did not defer delivery.",
                    )
                return
            receipt = buffered.flush()
            completed_ns = time.perf_counter_ns()
            if receipt.delivered_chunks != len(snapshot):
                raise LivePerformanceError(
                    "delivery_mismatch",
                    "The benchmark did not deliver the expected FIFO inventory.",
                )
            for sequence_number in snapshot:
                attempt_counts[sequence_number] += 1
                pending_sequences.popleft()
                observations.append(
                    LiveLatencyObservation(
                        observation_id=f"acceptance:{sequence_number:06d}",
                        kind=LiveLatencyKind.CAPTURE_TO_ACCEPTANCE,
                        elapsed_ms=_elapsed_ms(
                            capture_started_ns[sequence_number], completed_ns
                        ),
                        sequence_number=sequence_number,
                        attempt_number=attempt_counts[sequence_number],
                    )
                )

        for sequence_number in range(benchmark.max_buffered_chunks):
            enqueue(sequence_number)
        try:
            value = benchmark.max_buffered_chunks.to_bytes(2, "little")
            buffered.enqueue_pcm(
                value * benchmark.chunk_samples,
                captured_at=datetime.now(UTC),
            )
        except StreamResilienceError as error:
            if error.code != "backpressure":
                raise
            rejected_by_backpressure += 1
        else:
            raise LivePerformanceError(
                "backpressure_probe_failed",
                "The benchmark buffer accepted a chunk beyond its configured bound.",
            )
        flush(expect_deferred=True)
        flush()

        for sequence_number in range(
            benchmark.max_buffered_chunks, benchmark.chunk_count
        ):
            enqueue(sequence_number)
            if len(pending_sequences) == benchmark.max_buffered_chunks:
                flush()
        flush()
        buffered.close()
        status = buffered.status()

        if collector.close_summary is None:
            raise LivePerformanceError(
                "close_missing", "The rolling-window benchmark did not close cleanly."
            )
        if status.queued_chunks or status.discarded_chunks:
            raise LivePerformanceError(
                "delivery_incomplete", "The benchmark left PCM unresolved."
            )
        if (
            status.backpressure_events != rejected_by_backpressure
            or status.deferred_deliveries != benchmark.forced_deferred_deliveries
        ):
            raise LivePerformanceError(
                "probe_mismatch", "Backpressure or retry accounting did not match."
            )

        counters = LiveReliabilityCounters(
            chunks_enqueued=benchmark.chunk_count,
            delivery_attempts=sum(attempt_counts.values()),
            chunks_accepted=status.delivered_chunks,
            deferred_deliveries=status.deferred_deliveries,
            failed_deliveries=0,
            backpressure_rejections=rejected_by_backpressure,
            chunks_discarded=status.discarded_chunks,
            chunks_queued_at_end=status.queued_chunks,
            duplicate_acceptances=0,
            sequence_gaps=0,
            windows_emitted=len(collector.window_ids),
            windows_processed=len(collector.window_ids),
            window_processing_failures=0,
            alert_candidates=len(collector.alert_ids),
            local_alert_probe_completions=len(collector.alert_ids),
        )

    return build_live_performance_report(
        tuple(observations),
        counters,
        benchmark=benchmark,
        run_duration_ms=_elapsed_ms(run_started_ns, time.perf_counter_ns()),
        created_at=report_time,
    )


def save_live_performance_report(
    report: LivePerformanceReport,
    destination: Path,
    *,
    replace: bool = False,
) -> None:
    """Persist a validated report atomically; replacement must be explicit."""

    if not isinstance(report, LivePerformanceReport):
        raise TypeError("report must be LivePerformanceReport")
    validated = LivePerformanceReport.model_validate(
        report.model_dump(mode="python")
    )
    document = (validated.model_dump_json(indent=2) + "\n").encode("utf-8")
    if len(document) > MAX_LIVE_PERFORMANCE_BYTES:
        raise LivePerformanceError(
            "report_too_large", "The live performance report exceeds its size limit."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise LivePerformanceError(
            "output_exists", "The live performance report already exists."
        )
    stage = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
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
                raise LivePerformanceError(
                    "output_exists", "The live performance report already exists."
                ) from error
            stage.unlink()
    except LivePerformanceError:
        raise
    except OSError as error:
        raise LivePerformanceError(
            "write_failed", "The live performance report could not be saved."
        ) from error
    finally:
        try:
            stage.unlink(missing_ok=True)
        except OSError:
            pass


def load_live_performance_report(path: Path) -> LivePerformanceReport:
    """Load and revalidate a bounded report."""

    try:
        size = path.stat().st_size
        if not 0 < size <= MAX_LIVE_PERFORMANCE_BYTES:
            raise LivePerformanceError(
                "invalid_report_size", "The live performance report size is invalid."
            )
        document = path.read_bytes()
        if len(document) != size:
            raise LivePerformanceError(
                "report_changed", "The live performance report changed while read."
            )
        return LivePerformanceReport.model_validate_json(document)
    except LivePerformanceError:
        raise
    except Exception as error:
        raise LivePerformanceError(
            "invalid_report", "The live performance report failed validation."
        ) from error


def live_performance_schema_document() -> dict[str, object]:
    return LivePerformanceReport.model_json_schema()
