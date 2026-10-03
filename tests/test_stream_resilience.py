"""B10.2 tests for bounded buffering, reconnect, and backpressure."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from threading import Event

import numpy as np
import pytest

from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import ConsentRecord, ConsentStatus, ProcessingScope
from audio_sentinel.live_streaming import (
    AuthorizedStreamSession,
    DeviceEnrollment,
    EnrollmentStatus,
    StreamAudioFormat,
    StreamChunkReceipt,
    StreamCloseReason,
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
    BufferDiscardReason,
    BufferedConnectionState,
    BufferedStreamingClient,
    StreamBufferSettings,
    StreamingClientConnection,
    StreamResilienceError,
)
from audio_sentinel.streaming_ingestion import (
    FileDeviceEnrollmentRegistry,
    LocalDeviceAuthenticator,
    LocalStreamingClient,
    LocalStreamingIngestionService,
    StreamingIngestionError,
)


NOW = datetime(2026, 10, 3, 12, tzinfo=UTC)
FORMAT = StreamAudioFormat(
    sample_rate_hz=16_000,
    channels=1,
    chunk_duration_ms=10,
)


def consent(*, consent_id: str = "consent-b10-2-001") -> ConsentRecord:
    return ConsentRecord(
        consent_id=consent_id,
        status=ConsentStatus.GRANTED,
        processing_scope=ProcessingScope.ACOUSTIC_ONLY,
        device_authorized=True,
        granted_at=NOW - timedelta(minutes=5),
        expires_at=NOW + timedelta(minutes=10),
    )


def pcm(samples: int, value: int = 0) -> bytes:
    return np.full(samples, value, dtype="<i2").tobytes()


class ScriptedClient:
    def __init__(self, session_id: str, *, active_consent: ConsentRecord) -> None:
        self.session_id = session_id
        self.active_consent = active_consent
        self._session: AuthorizedStreamSession | None = None
        self.send_errors: deque[str] = deque()
        self.close_errors: deque[str] = deque()
        self.sent: list[tuple[bytes, datetime]] = []
        self.closed: list[StreamCloseReason] = []
        self.next_sequence = 0
        self.next_sample = 0
        self.send_started: Event | None = None
        self.send_release: Event | None = None
        self.bad_receipt = False

    @property
    def session(self) -> AuthorizedStreamSession | None:
        return self._session

    def open(
        self,
        *,
        consent: ConsentRecord,
        requested_scope: ProcessingScope,
        audio_format: StreamAudioFormat,
    ) -> AuthorizedStreamSession:
        assert consent == self.active_consent
        self._session = AuthorizedStreamSession(
            session_id=self.session_id,
            enrollment_id="enrollment-b10-2-001",
            enrollment_sha256="a" * 64,
            authentication_id=f"authentication-{self.session_id}",
            device_id="device-b10-2-001",
            consent_id=consent.consent_id,
            processing_scope=requested_scope,
            audio_format=audio_format,
            authorized_at=NOW - timedelta(seconds=1),
            expires_at=NOW + timedelta(minutes=1),
            raw_audio_persistence_allowed=False,
        )
        return self._session

    def send_pcm(
        self,
        payload: bytes,
        *,
        captured_at: datetime | None = None,
    ) -> StreamChunkReceipt:
        if self.send_started is not None:
            self.send_started.set()
        if self.send_release is not None:
            assert self.send_release.wait(2)
        if self.send_errors:
            code = self.send_errors.popleft()
            raise StreamingIngestionError(code, "safe scripted failure")
        assert self._session is not None and captured_at is not None
        count = len(payload) // (
            self._session.audio_format.channels
            * self._session.audio_format.sample_width_bytes
        )
        receipt = StreamChunkReceipt(
            session_id=self._session.session_id,
            sequence_number=self.next_sequence,
            start_sample=self.next_sample,
            end_sample=self.next_sample + count,
            sample_count=count,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            captured_at=captured_at,
            accepted_at=NOW,
        )
        self.sent.append((payload, captured_at))
        self.next_sequence += 1
        self.next_sample += count
        if self.bad_receipt:
            return receipt.model_copy(update={"payload_sha256": "0" * 64})
        return receipt

    def close(
        self,
        reason: StreamCloseReason = StreamCloseReason.CLIENT_REQUEST,
    ) -> StreamCloseReceipt:
        if self.close_errors:
            code = self.close_errors.popleft()
            raise StreamingIngestionError(code, "safe scripted close failure")
        if self._session is None:
            raise StreamingIngestionError(
                "client_session_missing", "No active scripted session."
            )
        receipt = StreamCloseReceipt(
            session_id=self._session.session_id,
            reason=reason,
            next_sequence_number=self.next_sequence,
            next_start_sample=self.next_sample,
            closed_at=NOW,
        )
        self.closed.append(reason)
        self._session = None
        return receipt


def controller(
    *,
    buffer_settings: StreamBufferSettings | None = None,
    current: list[datetime] | None = None,
) -> tuple[BufferedStreamingClient, list[ScriptedClient]]:
    clients: list[ScriptedClient] = []
    active_consent = consent()
    clock = current or [NOW]

    def factory() -> ScriptedClient:
        client = ScriptedClient(
            f"session-b10-2-{len(clients) + 1:03d}",
            active_consent=active_consent,
        )
        clients.append(client)
        return client

    buffered = BufferedStreamingClient(
        factory,
        consent=active_consent,
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
        settings=buffer_settings,
        clock=lambda: clock[0],
    )
    return buffered, clients


def test_settings_require_capacity_for_one_authorized_chunk() -> None:
    with pytest.raises(ValueError):
        BufferedStreamingClient(
            lambda: ScriptedClient("session-small", active_consent=consent()),
            consent=consent(),
            requested_scope=ProcessingScope.ACOUSTIC_ONLY,
            audio_format=FORMAT,
            settings=StreamBufferSettings(max_buffered_bytes=319),
            clock=lambda: NOW,
        )


def test_connect_exposes_audio_free_status_and_runtime_protocol() -> None:
    buffered, clients = controller()
    opened = buffered.connect()
    status = buffered.status()

    assert isinstance(clients[0], StreamingClientConnection)
    assert opened.previous_session_id is None
    assert opened.session_id == "session-b10-2-001"
    assert opened.carried_chunks == opened.carried_bytes == 0
    assert status.state is BufferedConnectionState.CONNECTED
    assert status.connection_index == 1 and status.queued_chunks == 0
    assert status.raw_audio_persisted is False


def test_fifo_enqueue_and_flush_remove_only_verified_deliveries() -> None:
    buffered, clients = controller()
    buffered.connect()
    first = buffered.enqueue_pcm(pcm(80, 1), captured_at=NOW)
    second = buffered.enqueue_pcm(pcm(160, 2), captured_at=NOW)

    result = buffered.flush()

    assert [item[0] for item in clients[0].sent] == [pcm(80, 1), pcm(160, 2)]
    assert first.enqueue_index == 0 and second.enqueue_index == 1
    assert result.delivered_chunks == 2 and result.delivered_bytes == 480
    assert result.remaining_chunks == result.remaining_bytes == 0
    assert (result.final_sequence_number, result.final_end_sample) == (1, 240)
    assert buffered.status().delivered_chunks == 2


def test_flush_limit_preserves_fifo_remainder() -> None:
    buffered, _clients = controller()
    buffered.connect()
    for value in (1, 2, 3):
        buffered.enqueue_pcm(pcm(40, value), captured_at=NOW)

    result = buffered.flush(max_chunks=2)

    assert result.delivered_chunks == 2
    assert result.remaining_chunks == 1 and result.remaining_bytes == 80
    assert buffered.status().queued_chunks == 1


def test_reject_backpressure_never_silently_drops_or_replaces_audio() -> None:
    buffered, _clients = controller(
        buffer_settings=StreamBufferSettings(
            max_buffered_chunks=1,
            max_buffered_bytes=320,
            backpressure_mode=BackpressureMode.REJECT,
        )
    )
    buffered.connect()
    first = buffered.enqueue_pcm(pcm(160, 1), captured_at=NOW)

    with pytest.raises(StreamResilienceError) as captured:
        buffered.enqueue_pcm(pcm(1, 2), captured_at=NOW)
    assert captured.value.code == "backpressure"
    status = buffered.status()
    assert status.queued_chunks == 1 and status.queued_bytes == 320
    assert status.backpressure_events == 1
    assert buffered.discard_buffer(BufferDiscardReason.OPERATOR_REQUEST).ticket_ids == (
        first.ticket_id,
    )


def test_block_backpressure_unblocks_when_flush_frees_capacity() -> None:
    buffered, _clients = controller(
        buffer_settings=StreamBufferSettings(
            max_buffered_chunks=1,
            max_buffered_bytes=320,
            backpressure_mode=BackpressureMode.BLOCK,
            block_timeout_seconds=2,
        )
    )
    buffered.connect()
    buffered.enqueue_pcm(pcm(160, 1), captured_at=NOW)
    started = Event()

    def enqueue_second() -> str:
        started.set()
        return buffered.enqueue_pcm(
            pcm(80, 2), captured_at=NOW, timeout_seconds=2
        ).ticket_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(enqueue_second)
        assert started.wait(1)
        flushed = buffered.flush(max_chunks=1)
        ticket_id = future.result(timeout=2)

    assert flushed.delivered_chunks == 1
    assert ticket_id.startswith("buffer-000000000001-")
    assert buffered.status().queued_chunks == 1


def test_block_backpressure_times_out_without_mutating_queue() -> None:
    buffered, _clients = controller(
        buffer_settings=StreamBufferSettings(
            max_buffered_chunks=1,
            max_buffered_bytes=320,
            backpressure_mode=BackpressureMode.BLOCK,
            block_timeout_seconds=0.02,
        )
    )
    buffered.connect()
    buffered.enqueue_pcm(pcm(160), captured_at=NOW)

    with pytest.raises(StreamResilienceError) as captured:
        buffered.enqueue_pcm(pcm(1), captured_at=NOW)
    assert captured.value.code == "backpressure_timeout"
    assert buffered.status().queued_chunks == 1


def test_sink_backpressure_defers_oldest_chunk_and_retry_is_exact() -> None:
    buffered, clients = controller()
    buffered.connect()
    ticket = buffered.enqueue_pcm(pcm(80, 7), captured_at=NOW)
    clients[0].send_errors.append("sink_failed")

    with pytest.raises(StreamResilienceError) as captured:
        buffered.flush()
    assert captured.value.code == "delivery_deferred"
    assert captured.value.cause_code == "sink_failed"
    assert buffered.status().queued_chunks == 1

    result = buffered.flush()
    assert result.delivered_chunks == 1
    assert clients[0].sent[0][0] == pcm(80, 7)
    assert hashlib.sha256(clients[0].sent[0][0]).hexdigest() == ticket.payload_sha256


def test_lost_session_retains_pcm_and_requires_explicit_discard_before_reconnect() -> None:
    buffered, clients = controller()
    buffered.connect()
    buffered.enqueue_pcm(pcm(80, 4), captured_at=NOW)
    clients[0].send_errors.append("session_not_active")

    with pytest.raises(StreamResilienceError) as captured:
        buffered.flush()
    assert captured.value.code == "reconnect_required"
    assert buffered.state is BufferedConnectionState.DISCONNECTED
    assert buffered.status().queued_chunks == 1
    with pytest.raises(StreamResilienceError) as captured:
        buffered.reconnect()
    assert captured.value.code == "buffer_not_empty"

    discarded = buffered.discard_buffer(BufferDiscardReason.CONNECTION_LOST)
    reconnect = buffered.reconnect()
    assert discarded.discarded_chunks == 1 and discarded.discarded_bytes == 160
    assert reconnect.previous_session_id == "session-b10-2-001"
    assert reconnect.session_id == "session-b10-2-002"
    assert reconnect.carried_chunks == reconnect.carried_bytes == 0


def test_clean_reconnect_closes_old_session_and_resets_capture_order() -> None:
    buffered, clients = controller()
    buffered.connect()
    buffered.enqueue_pcm(pcm(40), captured_at=NOW)
    buffered.flush()

    receipt = buffered.reconnect(reason=StreamCloseReason.DEVICE_DISCONNECTED)
    buffered.enqueue_pcm(pcm(40), captured_at=NOW - timedelta(milliseconds=1))

    assert clients[0].closed == [StreamCloseReason.DEVICE_DISCONNECTED]
    assert receipt.previous_session_id == "session-b10-2-001"
    assert receipt.session_id == "session-b10-2-002"
    assert buffered.status().reconnects == 1


def test_reconnect_can_use_refreshed_consent() -> None:
    clients: list[ScriptedClient] = []
    initial = consent(consent_id="consent-b10-2-old")
    refreshed = consent(consent_id="consent-b10-2-new")

    def factory() -> ScriptedClient:
        expected = initial if not clients else refreshed
        client = ScriptedClient(
            f"session-consent-{len(clients) + 1}", active_consent=expected
        )
        clients.append(client)
        return client

    buffered = BufferedStreamingClient(
        factory,
        consent=initial,
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
        clock=lambda: NOW,
    )
    buffered.connect()

    reconnect = buffered.reconnect(consent=refreshed)
    assert reconnect.session_id == "session-consent-2"
    assert buffered.session is not None
    assert buffered.session.consent_id == refreshed.consent_id


@pytest.mark.parametrize(
    ("document", "code"),
    [(b"", "invalid_payload"), (b"\x00", "invalid_payload"), (pcm(161), "chunk_too_large")],
)
def test_invalid_or_oversized_pcm_is_rejected_before_queueing(
    document: bytes,
    code: str,
) -> None:
    buffered, _clients = controller()
    buffered.connect()

    with pytest.raises(StreamResilienceError) as captured:
        buffered.enqueue_pcm(document, captured_at=NOW)
    assert captured.value.code == code
    assert buffered.status().queued_chunks == 0


def test_capture_times_must_be_current_authorized_and_nondecreasing() -> None:
    current = [NOW]
    buffered, _clients = controller(current=current)
    buffered.connect()
    buffered.enqueue_pcm(pcm(1), captured_at=NOW)

    with pytest.raises(StreamResilienceError) as captured:
        buffered.enqueue_pcm(pcm(1), captured_at=NOW - timedelta(milliseconds=500))
    assert captured.value.code == "capture_out_of_order"
    with pytest.raises(StreamResilienceError) as captured:
        buffered.enqueue_pcm(pcm(1), captured_at=NOW + timedelta(seconds=1))
    assert captured.value.code == "invalid_time"


def test_expired_connection_rejects_new_buffering_and_requires_reconnect() -> None:
    current = [NOW]
    buffered, _clients = controller(current=current)
    buffered.connect()
    current[0] = NOW + timedelta(minutes=2)

    with pytest.raises(StreamResilienceError) as captured:
        buffered.enqueue_pcm(pcm(1), captured_at=NOW)
    assert captured.value.code == "authorization_expired"
    assert buffered.state is BufferedConnectionState.DISCONNECTED
    assert buffered.status().queued_chunks == 0


def test_mismatched_delivery_receipt_retains_pcm_and_disconnects() -> None:
    buffered, clients = controller()
    buffered.connect()
    buffered.enqueue_pcm(pcm(80), captured_at=NOW)
    clients[0].bad_receipt = True

    with pytest.raises(StreamResilienceError) as captured:
        buffered.flush()
    assert captured.value.code == "receipt_mismatch"
    assert buffered.state is BufferedConnectionState.DISCONNECTED
    assert buffered.status().queued_chunks == 1


def test_close_requires_empty_buffer_and_is_idempotent_after_success() -> None:
    buffered, _clients = controller()
    buffered.connect()
    buffered.enqueue_pcm(pcm(40), captured_at=NOW)

    with pytest.raises(StreamResilienceError) as captured:
        buffered.close()
    assert captured.value.code == "buffer_not_empty"
    buffered.flush()
    receipt = buffered.close()

    assert receipt is not None and receipt.reason is StreamCloseReason.CLIENT_REQUEST
    assert buffered.state is BufferedConnectionState.CLOSED
    assert buffered.close() is None


def test_close_and_reconnect_defer_when_downstream_close_is_backpressured() -> None:
    buffered, clients = controller()
    buffered.connect()
    clients[0].close_errors.extend(["sink_failed", "sink_failed"])

    with pytest.raises(StreamResilienceError) as captured:
        buffered.close()
    assert captured.value.code == "close_deferred"
    assert buffered.state is BufferedConnectionState.CONNECTED
    with pytest.raises(StreamResilienceError) as captured:
        buffered.reconnect()
    assert captured.value.code == "reconnect_deferred"
    assert buffered.state is BufferedConnectionState.DISCONNECTED


def test_concurrent_flush_is_rejected_while_first_delivery_is_inflight() -> None:
    buffered, clients = controller()
    buffered.connect()
    buffered.enqueue_pcm(pcm(80), captured_at=NOW)
    clients[0].send_started = Event()
    clients[0].send_release = Event()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(buffered.flush)
        assert clients[0].send_started.wait(1)
        with pytest.raises(StreamResilienceError) as captured:
            buffered.flush()
        assert captured.value.code == "flush_in_progress"
        clients[0].send_release.set()
        assert first.result(timeout=2).delivered_chunks == 1


def test_invalid_factory_result_fails_closed() -> None:
    buffered = BufferedStreamingClient(
        lambda: object(),
        consent=consent(),
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
        clock=lambda: NOW,
    )

    with pytest.raises(StreamResilienceError) as captured:
        buffered.connect()
    assert captured.value.code == "connection_failed"
    assert buffered.state is BufferedConnectionState.DISCONNECTED


class WindowCollector:
    def __init__(self) -> None:
        self.windows: list[LiveAudioWindow] = []
        self.summaries: list[LiveWindowCloseSummary] = []

    def accept_window(self, session, window) -> None:
        self.windows.append(window)

    def close_live_windows(self, session, receipt, summary) -> None:
        self.summaries.append(summary)


def test_real_ingestion_and_rolling_pipeline_preserve_buffered_fifo(
    temporary_settings: AudioSentinelSettings,
) -> None:
    secret = b"b10-2-real-integration-credential"
    active_consent = consent(consent_id="consent-b10-2-real")
    enrollment = DeviceEnrollment(
        enrollment_id="enrollment-b10-2-real",
        device_id="device-b10-2-real",
        status=EnrollmentStatus.ACTIVE,
        authorization_reference="approval-b10-2-real",
        credential_fingerprint_sha256=hashlib.sha256(secret).hexdigest(),
        authorized_scopes=(ProcessingScope.ACOUSTIC_ONLY,),
        audio_formats=(FORMAT,),
        enrolled_at=NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=1),
    )
    temporary_settings.ensure_directories()
    registry = FileDeviceEnrollmentRegistry(temporary_settings.paths)
    registry.register(enrollment)
    authenticator = LocalDeviceAuthenticator(lambda _record: secret, clock=lambda: NOW)
    collector = WindowCollector()
    windows = RollingLiveWindowSink(
        collector,
        settings=RollingWindowSettings(
            target_sample_rate_hz=16_000,
            window_seconds=(0.01,),
            window_overlap_ratio=0,
            close_tail_policy="pad",
        ),
    )
    service = LocalStreamingIngestionService(
        registry, authenticator, windows, clock=lambda: NOW
    )
    client_number = [0]

    def factory() -> LocalStreamingClient:
        client_number[0] += 1
        return LocalStreamingClient(
            service,
            enrollment_id=enrollment.enrollment_id,
            device_id=enrollment.device_id,
            credential_provider=lambda: secret,
            clock=lambda: NOW,
            session_id_factory=lambda: f"session-b10-2-real-{client_number[0]}",
        )

    buffered = BufferedStreamingClient(
        factory,
        consent=active_consent,
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
        clock=lambda: NOW,
    )
    buffered.connect()
    first = np.arange(80, dtype="<i2").tobytes()
    second = np.arange(80, 160, dtype="<i2").tobytes()
    buffered.enqueue_pcm(first, captured_at=NOW)
    buffered.enqueue_pcm(second, captured_at=NOW)
    buffered.flush()
    buffered.close()

    assert len(collector.windows) == 1
    np.testing.assert_array_equal(
        collector.windows[0].samples,
        np.arange(160, dtype=np.float32) / 32768.0,
    )
    assert collector.summaries[0].total_input_samples == 160
    files = [item for item in temporary_settings.paths.root.rglob("*") if item.is_file()]
    assert {item.suffix for item in files} == {".json"}


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
from audio_sentinel.stream_resilience import BufferedStreamingClient
assert BufferedStreamingClient.__name__ == 'BufferedStreamingClient'
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
