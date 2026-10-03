"""A10.2 tests for deterministic rolling live-window adaptation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from audio_sentinel.config import AudioSentinelSettings, AudioSettings
from audio_sentinel.contracts import ConsentRecord, ConsentStatus, ProcessingScope
from audio_sentinel.live_streaming import (
    AuthorizedStreamSession,
    DeviceEnrollment,
    EnrollmentStatus,
    StreamAudioChunk,
    StreamAudioFormat,
    StreamChunkReceipt,
    StreamCloseReason,
    StreamCloseReceipt,
)
from audio_sentinel.live_windows import (
    LiveAudioWindow,
    LiveWindowCloseSummary,
    LiveWindowConsumer,
    RollingLiveWindowSink,
    RollingWindowError,
    RollingWindowSettings,
)
from audio_sentinel.streaming_ingestion import StreamingAudioSink
from audio_sentinel import streaming_ingestion


NOW = datetime(2026, 1, 15, 12, tzinfo=UTC)


class CollectingConsumer:
    def __init__(
        self,
        *,
        fail_window_calls: set[int] | None = None,
        fail_close_calls: set[int] | None = None,
    ) -> None:
        self.windows: list[LiveAudioWindow] = []
        self.closes: list[tuple[StreamCloseReceipt, LiveWindowCloseSummary]] = []
        self.window_calls = 0
        self.close_calls = 0
        self.fail_window_calls = fail_window_calls or set()
        self.fail_close_calls = fail_close_calls or set()

    def accept_window(
        self,
        session: AuthorizedStreamSession,
        window: LiveAudioWindow,
    ) -> None:
        assert window.record.session_id == session.session_id
        self.window_calls += 1
        if self.window_calls in self.fail_window_calls:
            raise RuntimeError("injected window failure")
        self.windows.append(window)

    def close_live_windows(
        self,
        session: AuthorizedStreamSession,
        receipt: StreamCloseReceipt,
        summary: LiveWindowCloseSummary,
    ) -> None:
        assert summary.session_id == session.session_id == receipt.session_id
        self.close_calls += 1
        if self.close_calls in self.fail_close_calls:
            raise RuntimeError("injected close failure")
        self.closes.append((receipt, summary))


def session(
    *,
    session_id: str = "session-a10-2-001",
    rate: int = 16_000,
    channels: int = 1,
) -> AuthorizedStreamSession:
    return AuthorizedStreamSession(
        session_id=session_id,
        enrollment_id="enrollment-a10-2-001",
        enrollment_sha256="a" * 64,
        authentication_id="authentication-a10-2-001",
        device_id="device-a10-2-001",
        consent_id="consent-a10-2-001",
        processing_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=StreamAudioFormat(
            sample_rate_hz=rate,
            channels=channels,
            chunk_duration_ms=1_000,
        ),
        authorized_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
        raw_audio_persistence_allowed=False,
    )


def settings(
    *,
    windows: tuple[float, ...] = (0.02,),
    overlap: float = 0.5,
    tail: str = "pad",
    **limits: int,
) -> RollingWindowSettings:
    return RollingWindowSettings(
        target_sample_rate_hz=16_000,
        window_seconds=windows,
        window_overlap_ratio=overlap,
        close_tail_policy=tail,
        **limits,
    )


def payload(values: np.ndarray | list[int]) -> bytes:
    return np.asarray(values, dtype="<i2").tobytes()


def handoff(
    active: AuthorizedStreamSession,
    *,
    sequence: int,
    start: int,
    values: np.ndarray | list[int],
) -> tuple[StreamAudioChunk, StreamChunkReceipt]:
    document = payload(values)
    samples = np.asarray(values)
    sample_count = len(samples) if samples.ndim == 1 else samples.shape[0]
    captured = NOW + timedelta(milliseconds=sequence)
    chunk = StreamAudioChunk(
        session_id=active.session_id,
        sequence_number=sequence,
        start_sample=start,
        sample_count=sample_count,
        captured_at=captured,
        payload=document,
    )
    receipt = StreamChunkReceipt(
        session_id=active.session_id,
        sequence_number=sequence,
        start_sample=start,
        end_sample=start + sample_count,
        sample_count=sample_count,
        payload_sha256=hashlib.sha256(document).hexdigest(),
        captured_at=captured,
        accepted_at=captured + timedelta(milliseconds=1),
    )
    return chunk, receipt


def close_receipt(
    active: AuthorizedStreamSession,
    *,
    next_sequence: int,
    next_start: int,
) -> StreamCloseReceipt:
    return StreamCloseReceipt(
        session_id=active.session_id,
        reason=StreamCloseReason.CLIENT_REQUEST,
        next_sequence_number=next_sequence,
        next_start_sample=next_start,
        closed_at=NOW + timedelta(seconds=2),
    )


def accept(
    sink: RollingLiveWindowSink,
    active: AuthorizedStreamSession,
    *,
    sequence: int,
    start: int,
    values: np.ndarray | list[int],
) -> tuple[StreamAudioChunk, StreamChunkReceipt]:
    chunk, receipt = handoff(
        active,
        sequence=sequence,
        start=start,
        values=values,
    )
    sink.accept_chunk(active, chunk, receipt)
    return chunk, receipt


def test_settings_copy_offline_window_grid_and_validate_memory() -> None:
    offline = AudioSettings(
        target_sample_rate_hz=16_000,
        window_seconds=(0.02, 0.04),
        window_overlap_ratio=0.25,
        tail_policy="drop",
    )
    live = RollingWindowSettings.from_audio_settings(offline)

    assert live.window_sample_counts(0.02) == offline.window_sample_counts(0.02)
    assert live.window_seconds == offline.window_seconds
    assert live.close_tail_policy == "drop"
    with pytest.raises(ValueError):
        RollingWindowSettings(window_seconds=(0.04, 0.02))
    with pytest.raises(ValueError):
        RollingWindowSettings(
            window_seconds=(1.0,),
            max_buffer_bytes=63_999,
        )
    with pytest.raises(ValueError):
        RollingWindowSettings.from_audio_settings(
            AudioSettings(convert_to_mono=False)
        )


def test_irregular_chunks_emit_the_offline_grid_without_waiting_for_close() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(consumer, settings=settings())
    active = session()
    source = np.arange(480, dtype=np.int16)

    accept(sink, active, sequence=0, start=0, values=source[:100])
    assert consumer.windows == []
    accept(sink, active, sequence=1, start=100, values=source[100:320])
    assert [(w.record.start_sample, w.record.end_sample) for w in consumer.windows] == [
        (0, 320)
    ]
    accept(sink, active, sequence=2, start=320, values=source[320:])

    assert [(w.record.start_sample, w.record.end_sample) for w in consumer.windows] == [
        (0, 320),
        (160, 480),
    ]
    np.testing.assert_array_equal(
        consumer.windows[1].samples,
        source[160:480].astype(np.float32) / 32768.0,
    )
    assert all(not item.samples.flags.writeable for item in consumer.windows)


def test_close_pads_only_the_missing_tail_on_the_same_grid() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(consumer, settings=settings())
    active = session()
    source = np.arange(400, dtype=np.int16)
    accept(sink, active, sequence=0, start=0, values=source)

    sink.close_stream(
        active,
        close_receipt(active, next_sequence=1, next_start=400),
    )

    assert [(w.record.start_sample, w.record.end_sample) for w in consumer.windows] == [
        (0, 320),
        (160, 400),
    ]
    tail = consumer.windows[-1]
    assert tail.record.padding_samples == 80
    np.testing.assert_array_equal(
        tail.samples[:240], source[160:].astype(np.float32) / 32768.0
    )
    assert np.count_nonzero(tail.samples[240:]) == 0
    summary = consumer.closes[0][1]
    assert summary.total_input_samples == 400
    assert summary.total_emitted_windows == 2
    assert summary.groups[0].padded_windows == 1
    assert summary.groups[0].uncovered_tail_samples == 0


def test_drop_tail_reports_uncovered_samples_without_emitting_padding() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(consumer, settings=settings(tail="drop"))
    active = session()
    accept(sink, active, sequence=0, start=0, values=np.arange(400))

    sink.close_stream(
        active,
        close_receipt(active, next_sequence=1, next_start=400),
    )

    assert len(consumer.windows) == 1
    summary = consumer.closes[0][1]
    assert summary.groups[0].padded_windows == 0
    assert summary.groups[0].uncovered_tail_samples == 80


def test_multiple_window_groups_emit_in_deterministic_readiness_order() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(
        consumer,
        settings=settings(windows=(0.01, 0.02)),
    )
    active = session()
    accept(sink, active, sequence=0, start=0, values=np.arange(320))

    assert [
        (w.record.group_index, w.record.start_sample, w.record.end_sample)
        for w in consumer.windows
    ] == [
        (0, 0, 160),
        (0, 80, 240),
        (0, 160, 320),
        (1, 0, 320),
    ]
    assert len({item.record.window_id for item in consumer.windows}) == 4


def test_stereo_pcm_is_downmixed_to_owned_float32_mono() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(
        consumer,
        settings=settings(windows=(0.01,), overlap=0),
    )
    active = session(channels=2)
    stereo = np.column_stack(
        (np.arange(160, dtype=np.int16), np.arange(160, dtype=np.int16))
    )
    accept(sink, active, sequence=0, start=0, values=stereo)

    window = consumer.windows[0]
    assert window.samples.dtype == np.float32 and window.samples.shape == (160,)
    np.testing.assert_array_equal(
        window.samples,
        np.arange(160, dtype=np.float32) / 32768.0,
    )


def test_sample_rate_mismatch_fails_before_consumer_handoff() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(consumer, settings=settings())
    active = session(rate=8_000)
    chunk, receipt = handoff(
        active, sequence=0, start=0, values=np.arange(80)
    )

    with pytest.raises(RollingWindowError) as captured:
        sink.accept_chunk(active, chunk, receipt)
    assert captured.value.code == "sample_rate_mismatch"
    assert consumer.windows == []


def test_receipt_or_pcm_geometry_mismatch_fails_closed() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(consumer, settings=settings())
    active = session()
    chunk, receipt = handoff(
        active, sequence=0, start=0, values=np.arange(100)
    )

    with pytest.raises(RollingWindowError) as captured:
        sink.accept_chunk(
            active,
            chunk,
            receipt.model_copy(update={"payload_sha256": "0" * 64}),
        )
    assert captured.value.code == "handoff_mismatch"

    malformed = chunk.model_copy(update={"sample_count": 101})
    with pytest.raises(RollingWindowError) as captured:
        sink.accept_chunk(active, malformed, receipt)
    assert captured.value.code == "handoff_mismatch"
    assert consumer.windows == []


def test_window_consumer_failure_resumes_same_chunk_without_duplicates() -> None:
    consumer = CollectingConsumer(fail_window_calls={2})
    sink = RollingLiveWindowSink(
        consumer,
        settings=settings(windows=(0.01,), overlap=0.5),
    )
    active = session()
    chunk, receipt = handoff(
        active, sequence=0, start=0, values=np.arange(320)
    )

    with pytest.raises(RollingWindowError) as captured:
        sink.accept_chunk(active, chunk, receipt)
    assert captured.value.code == "consumer_failed"
    assert [w.record.start_sample for w in consumer.windows] == [0]

    sink.accept_chunk(active, chunk, receipt)
    assert [w.record.start_sample for w in consumer.windows] == [0, 80, 160]
    assert len({w.record.window_id for w in consumer.windows}) == 3


def test_failed_chunk_cannot_be_replaced_by_a_different_handoff() -> None:
    consumer = CollectingConsumer(fail_window_calls={1, 2})
    sink = RollingLiveWindowSink(
        consumer,
        settings=settings(windows=(0.01,), overlap=0),
    )
    active = session()
    first = handoff(active, sequence=0, start=0, values=np.arange(160))
    second = handoff(active, sequence=1, start=160, values=np.arange(160))

    with pytest.raises(RollingWindowError):
        sink.accept_chunk(active, *first)
    with pytest.raises(RollingWindowError) as captured:
        sink.accept_chunk(active, *second)
    assert captured.value.code == "chunk_pending"


def test_close_window_failure_resumes_without_reemitting_completed_tail() -> None:
    consumer = CollectingConsumer(fail_window_calls={2})
    sink = RollingLiveWindowSink(
        consumer,
        settings=settings(windows=(0.01, 0.02), overlap=0),
    )
    active = session()
    accept(sink, active, sequence=0, start=0, values=np.arange(100))
    closing = close_receipt(active, next_sequence=1, next_start=100)

    with pytest.raises(RollingWindowError) as captured:
        sink.close_stream(active, closing)
    assert captured.value.code == "consumer_failed"
    assert len(consumer.windows) == 1

    sink.close_stream(active, closing)
    assert len(consumer.windows) == 2
    assert len({item.record.window_id for item in consumer.windows}) == 2
    assert consumer.closes[0][1].total_emitted_windows == 2


def test_close_callback_failure_retries_summary_without_reemitting_windows() -> None:
    consumer = CollectingConsumer(fail_close_calls={1})
    sink = RollingLiveWindowSink(
        consumer,
        settings=settings(windows=(0.01,), overlap=0),
    )
    active = session()
    accept(sink, active, sequence=0, start=0, values=np.arange(100))
    closing = close_receipt(active, next_sequence=1, next_start=100)

    with pytest.raises(RollingWindowError) as captured:
        sink.close_stream(active, closing)
    assert captured.value.code == "consumer_failed"
    assert len(consumer.windows) == 1

    sink.close_stream(active, closing)
    assert len(consumer.windows) == 1
    assert len(consumer.closes) == 1
    assert sink.active_stream_count == 0


def test_empty_stream_close_emits_no_synthetic_silence() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(consumer, settings=settings())
    active = session()

    sink.close_stream(
        active,
        close_receipt(active, next_sequence=0, next_start=0),
    )

    assert consumer.windows == []
    assert consumer.closes[0][1].total_input_samples == 0
    assert consumer.closes[0][1].total_emitted_windows == 0


def test_stream_capacity_is_bounded_without_replacing_active_state() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(
        consumer,
        settings=settings(max_active_streams=1),
    )
    first = session(session_id="session-a10-2-first")
    second = session(session_id="session-a10-2-second")
    accept(sink, first, sequence=0, start=0, values=np.arange(100))
    second_chunk = handoff(second, sequence=0, start=0, values=np.arange(100))

    with pytest.raises(RollingWindowError) as captured:
        sink.accept_chunk(second, *second_chunk)
    assert captured.value.code == "stream_capacity_reached"
    assert sink.active_stream_count == 1


def test_buffer_limit_is_enforced_before_appending_new_pcm() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(
        consumer,
        settings=settings(
            windows=(0.02,),
            overlap=0.5,
            max_buffer_bytes=1_280,
        ),
    )
    active = session()
    accept(sink, active, sequence=0, start=0, values=np.arange(300))
    second = handoff(active, sequence=1, start=300, values=np.arange(300))

    with pytest.raises(RollingWindowError) as captured:
        sink.accept_chunk(active, *second)
    assert captured.value.code == "buffer_limit_exceeded"
    assert consumer.windows == []


def test_close_position_must_match_received_stream() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(consumer, settings=settings())
    active = session()
    accept(sink, active, sequence=0, start=0, values=np.arange(100))

    with pytest.raises(RollingWindowError) as captured:
        sink.close_stream(
            active,
            close_receipt(active, next_sequence=1, next_start=99),
        )
    assert captured.value.code == "sample_position_mismatch"
    assert consumer.closes == []


def test_rolling_sink_and_consumer_satisfy_runtime_protocols() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(consumer, settings=settings())

    assert isinstance(consumer, LiveWindowConsumer)
    assert isinstance(sink, StreamingAudioSink)


def test_ingestion_service_drives_rolling_windows_end_to_end(
    temporary_settings: AudioSentinelSettings,
) -> None:
    secret = b"a10-2-integration-credential-0001"
    audio_format = StreamAudioFormat(
        sample_rate_hz=16_000,
        channels=1,
        chunk_duration_ms=20,
    )
    consent = ConsentRecord(
        consent_id="consent-a10-2-integration",
        status=ConsentStatus.GRANTED,
        processing_scope=ProcessingScope.ACOUSTIC_ONLY,
        device_authorized=True,
        granted_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=10),
    )
    enrollment = DeviceEnrollment(
        enrollment_id="enrollment-a10-2-integration",
        device_id="device-a10-2-integration",
        status=EnrollmentStatus.ACTIVE,
        authorization_reference="approval-a10-2-integration",
        credential_fingerprint_sha256=hashlib.sha256(secret).hexdigest(),
        authorized_scopes=(ProcessingScope.ACOUSTIC_ONLY,),
        audio_formats=(audio_format,),
        enrolled_at=NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=1),
    )
    temporary_settings.ensure_directories()
    registry = streaming_ingestion.FileDeviceEnrollmentRegistry(
        temporary_settings.paths
    )
    registry.register(enrollment)
    authenticator = streaming_ingestion.LocalDeviceAuthenticator(
        lambda _record: secret,
        clock=lambda: NOW,
    )
    consumer = CollectingConsumer()
    window_sink = RollingLiveWindowSink(
        consumer,
        settings=settings(windows=(0.01,), overlap=0),
    )
    service = streaming_ingestion.LocalStreamingIngestionService(
        registry,
        authenticator,
        window_sink,
        clock=lambda: NOW,
    )
    client = streaming_ingestion.LocalStreamingClient(
        service,
        enrollment_id=enrollment.enrollment_id,
        device_id=enrollment.device_id,
        credential_provider=lambda: secret,
        clock=lambda: NOW,
        session_id_factory=lambda: "session-a10-2-integration",
    )

    client.open(
        consent=consent,
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=audio_format,
    )
    client.send_pcm(payload(np.arange(320, dtype=np.int16)))
    client.close()

    assert [item.record.start_sample for item in consumer.windows] == [0, 160]
    assert consumer.closes[0][1].total_input_samples == 320
    assert consumer.closes[0][1].total_emitted_windows == 2


def test_window_record_rejects_mutable_or_changed_samples() -> None:
    consumer = CollectingConsumer()
    sink = RollingLiveWindowSink(
        consumer,
        settings=settings(windows=(0.01,), overlap=0),
    )
    active = session()
    accept(sink, active, sequence=0, start=0, values=np.arange(160))
    emitted = consumer.windows[0]

    mutable = emitted.samples.copy()
    with pytest.raises(ValueError):
        LiveAudioWindow(record=emitted.record, samples=mutable)
    changed = emitted.samples.copy()
    changed[0] = 0.5
    changed.setflags(write=False)
    with pytest.raises(ValueError):
        LiveAudioWindow(record=emitted.record, samples=changed)


def test_module_import_has_no_network_or_model_runtime_dependency() -> None:
    code = """
import pathlib
import sys
class RejectImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'onnxruntime'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RejectImports())
def reject(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError('Unexpected import side effect: ' + event)
sys.addaudithook(reject)
from audio_sentinel.live_windows import RollingLiveWindowSink
assert RollingLiveWindowSink.__name__ == 'RollingLiveWindowSink'
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
