"""B10.2 bounded buffering, reconnect, and backpressure for live PCM clients."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import hashlib
from threading import Condition, Lock, RLock
import time
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.contracts import ConsentRecord, ProcessingScope
from audio_sentinel.live_streaming import (
    AuthorizedStreamSession,
    StreamAudioFormat,
    StreamChunkReceipt,
    StreamCloseReason,
    StreamCloseReceipt,
)
from audio_sentinel.streaming_ingestion import StreamingIngestionError


DEFAULT_MAX_BUFFERED_CHUNKS = 128
DEFAULT_MAX_BUFFERED_BYTES = 16_777_216
DEFAULT_BLOCK_TIMEOUT_SECONDS = 5.0


class StreamResilienceError(RuntimeError):
    """Stable B10.2 failure with an optional safe upstream failure code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        cause_code: str | None = None,
    ) -> None:
        self.code = code
        self.cause_code = cause_code
        super().__init__(message)


class BackpressureMode(str, Enum):
    REJECT = "reject"
    BLOCK = "block"


class BufferedConnectionState(str, Enum):
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CLOSED = "closed"


class BufferDiscardReason(str, Enum):
    OPERATOR_REQUEST = "operator_request"
    AUTHORIZATION_EXPIRED = "authorization_expired"
    CONNECTION_LOST = "connection_lost"
    SERVICE_SHUTDOWN = "service_shutdown"


class StreamBufferSettings(BaseModel):
    """Limits and full-queue behavior for one resilient live client."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_buffered_chunks: int = Field(
        default=DEFAULT_MAX_BUFFERED_CHUNKS,
        gt=0,
        strict=True,
    )
    max_buffered_bytes: int = Field(
        default=DEFAULT_MAX_BUFFERED_BYTES,
        gt=0,
        strict=True,
    )
    backpressure_mode: BackpressureMode = BackpressureMode.REJECT
    block_timeout_seconds: float = Field(
        default=DEFAULT_BLOCK_TIMEOUT_SECONDS,
        gt=0,
        le=60,
    )


class BufferedChunkTicket(BaseModel):
    """Audio-free identity returned when PCM enters the bounded queue."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )

    ticket_id: str = Field(pattern=r"^buffer-[0-9]{12}-[a-f0-9]{16}$")
    enqueue_index: int = Field(ge=0, strict=True)
    sample_count: int = Field(gt=0, strict=True)
    size_bytes: int = Field(gt=0, strict=True)
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    captured_at: datetime
    enqueued_at: datetime

    @field_validator("captured_at", "enqueued_at")
    @classmethod
    def validate_time(cls, value: datetime, info: object) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"{getattr(info, 'field_name', 'timestamp')} must include a timezone"
            )
        return value

    @model_validator(mode="after")
    def validate_order(self) -> "BufferedChunkTicket":
        if self.enqueued_at < self.captured_at:
            raise ValueError("enqueued_at cannot precede captured_at")
        return self


class StreamBufferStatus(BaseModel):
    """Audio-free current buffer and connection counters."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    state: BufferedConnectionState
    connection_index: int = Field(ge=0, strict=True)
    session_id: str | None = None
    queued_chunks: int = Field(ge=0, strict=True)
    queued_bytes: int = Field(ge=0, strict=True)
    oldest_captured_at: datetime | None = None
    delivered_chunks: int = Field(ge=0, strict=True)
    delivered_bytes: int = Field(ge=0, strict=True)
    discarded_chunks: int = Field(ge=0, strict=True)
    discarded_bytes: int = Field(ge=0, strict=True)
    backpressure_events: int = Field(ge=0, strict=True)
    deferred_deliveries: int = Field(ge=0, strict=True)
    reconnects: int = Field(ge=0, strict=True)
    raw_audio_persisted: Literal[False] = False


class BufferFlushReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    session_id: str = Field(min_length=3, max_length=128)
    delivered_chunks: int = Field(ge=0, strict=True)
    delivered_bytes: int = Field(ge=0, strict=True)
    remaining_chunks: int = Field(ge=0, strict=True)
    remaining_bytes: int = Field(ge=0, strict=True)
    final_sequence_number: int | None = Field(default=None, ge=0, strict=True)
    final_end_sample: int | None = Field(default=None, ge=1, strict=True)
    raw_audio_persisted: Literal[False] = False

    @model_validator(mode="after")
    def validate_final_position(self) -> "BufferFlushReceipt":
        if (self.final_sequence_number is None) != (self.final_end_sample is None):
            raise ValueError("final sequence and sample positions must appear together")
        return self


class BufferDiscardReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    reason: BufferDiscardReason
    discarded_chunks: int = Field(ge=0, strict=True)
    discarded_bytes: int = Field(ge=0, strict=True)
    ticket_ids: tuple[str, ...]
    discarded_at: datetime
    raw_audio_persisted: Literal[False] = False

    @field_validator("discarded_at")
    @classmethod
    def validate_discarded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("discarded_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_inventory(self) -> "BufferDiscardReceipt":
        if len(self.ticket_ids) != self.discarded_chunks:
            raise ValueError("ticket_ids must match discarded_chunks")
        return self


class StreamReconnectReceipt(BaseModel):
    """Proof of a clean session boundary; no audio crosses the boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    previous_session_id: str | None = None
    session_id: str = Field(min_length=3, max_length=128)
    connection_index: int = Field(gt=0, strict=True)
    reconnected_at: datetime
    carried_chunks: int = Field(default=0, ge=0, le=0, strict=True)
    carried_bytes: int = Field(default=0, ge=0, le=0, strict=True)
    raw_audio_persisted: Literal[False] = False

    @field_validator("reconnected_at")
    @classmethod
    def validate_reconnected_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reconnected_at must include a timezone")
        return value


@runtime_checkable
class StreamingClientConnection(Protocol):
    """Minimum authenticated connection used by the buffering controller."""

    @property
    def session(self) -> AuthorizedStreamSession | None: ...

    def open(
        self,
        *,
        consent: ConsentRecord,
        requested_scope: ProcessingScope,
        audio_format: StreamAudioFormat,
    ) -> AuthorizedStreamSession: ...

    def send_pcm(
        self,
        payload: bytes,
        *,
        captured_at: datetime | None = None,
    ) -> StreamChunkReceipt: ...

    def close(
        self,
        reason: StreamCloseReason = StreamCloseReason.CLIENT_REQUEST,
    ) -> StreamCloseReceipt: ...


@dataclass
class _BufferedChunk:
    ticket: BufferedChunkTicket
    payload: bytearray = field(repr=False)

    def erase(self) -> None:
        self.payload[:] = b"\x00" * len(self.payload)


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise StreamResilienceError(
            "invalid_time", f"{name} must include a valid UTC offset."
        )
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise StreamResilienceError(
            "invalid_time", f"{name} must include a valid UTC offset."
        ) from error


def _now(clock: Callable[[], datetime]) -> datetime:
    try:
        value = clock()
    except Exception as error:
        raise StreamResilienceError(
            "clock_failed", "The buffering clock was unavailable."
        ) from error
    if not isinstance(value, datetime):
        raise StreamResilienceError(
            "clock_failed", "The buffering clock returned an invalid value."
        )
    return _aware(value, "buffer time")


class BufferedStreamingClient:
    """Bounded FIFO and explicit reconnect controller for local live PCM.

    The queue retains raw PCM only in memory. A chunk leaves the queue only after
    the active connection returns a matching receipt. Reconnect never moves PCM
    from an earlier authorization into a newly authenticated session.
    """

    def __init__(
        self,
        client_factory: Callable[[], StreamingClientConnection],
        *,
        consent: ConsentRecord,
        requested_scope: ProcessingScope,
        audio_format: StreamAudioFormat,
        settings: StreamBufferSettings | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        if not isinstance(consent, ConsentRecord):
            raise TypeError("consent must be ConsentRecord")
        if not isinstance(requested_scope, ProcessingScope):
            raise TypeError("requested_scope must be ProcessingScope")
        if not isinstance(audio_format, StreamAudioFormat):
            raise TypeError("audio_format must be StreamAudioFormat")
        self._client_factory = client_factory
        self._consent = ConsentRecord.model_validate(consent.model_dump(mode="python"))
        self._requested_scope = requested_scope
        self._audio_format = StreamAudioFormat.model_validate(
            audio_format.model_dump(mode="python")
        )
        self._settings = StreamBufferSettings.model_validate(
            (settings or StreamBufferSettings()).model_dump()
        )
        if self._settings.max_buffered_bytes < self._audio_format.max_chunk_bytes:
            raise ValueError("max_buffered_bytes must hold one authorized maximum chunk")
        self._clock = clock
        self._condition = Condition(RLock())
        self._flush_lock = Lock()
        self._queue: deque[_BufferedChunk] = deque()
        self._queued_bytes = 0
        self._client: StreamingClientConnection | None = None
        self._state = BufferedConnectionState.NEW
        self._connection_index = 0
        self._next_enqueue_index = 0
        self._last_capture_at: datetime | None = None
        self._inflight = False
        self._delivered_chunks = 0
        self._delivered_bytes = 0
        self._discarded_chunks = 0
        self._discarded_bytes = 0
        self._backpressure_events = 0
        self._deferred_deliveries = 0
        self._reconnects = 0

    @property
    def settings(self) -> StreamBufferSettings:
        return self._settings

    @property
    def state(self) -> BufferedConnectionState:
        with self._condition:
            return self._state

    @property
    def session(self) -> AuthorizedStreamSession | None:
        with self._condition:
            return self._client.session if self._client is not None else None

    def status(self) -> StreamBufferStatus:
        with self._condition:
            session = self._client.session if self._client is not None else None
            oldest = self._queue[0].ticket.captured_at if self._queue else None
            return StreamBufferStatus(
                state=self._state,
                connection_index=self._connection_index,
                session_id=session.session_id if session is not None else None,
                queued_chunks=len(self._queue),
                queued_bytes=self._queued_bytes,
                oldest_captured_at=oldest,
                delivered_chunks=self._delivered_chunks,
                delivered_bytes=self._delivered_bytes,
                discarded_chunks=self._discarded_chunks,
                discarded_bytes=self._discarded_bytes,
                backpressure_events=self._backpressure_events,
                deferred_deliveries=self._deferred_deliveries,
                reconnects=self._reconnects,
            )

    def _new_client(self) -> StreamingClientConnection:
        try:
            client = self._client_factory()
        except Exception as error:
            raise StreamResilienceError(
                "connection_failed", "A streaming client connection was unavailable."
            ) from error
        if not isinstance(client, StreamingClientConnection):
            raise StreamResilienceError(
                "connection_failed",
                "The streaming client connection did not satisfy its contract.",
            )
        return client

    def _open_new_client(
        self,
        *,
        previous_session_id: str | None,
    ) -> StreamReconnectReceipt:
        current = _now(self._clock)
        client = self._new_client()
        try:
            session = client.open(
                consent=self._consent,
                requested_scope=self._requested_scope,
                audio_format=self._audio_format,
            )
        except Exception as error:
            code = getattr(error, "code", None)
            raise StreamResilienceError(
                "connection_failed",
                "A new authenticated stream could not be opened.",
                cause_code=code if isinstance(code, str) else None,
            ) from error
        if (
            not isinstance(session, AuthorizedStreamSession)
            or session != client.session
            or session.audio_format != self._audio_format
            or session.processing_scope is not self._requested_scope
            or session.consent_id != self._consent.consent_id
        ):
            try:
                client.close(StreamCloseReason.PROTOCOL_ERROR)
            except Exception:
                pass
            raise StreamResilienceError(
                "connection_mismatch",
                "The new stream does not match the requested authorization.",
            )
        with self._condition:
            self._client = client
            self._connection_index += 1
            self._state = BufferedConnectionState.CONNECTED
            self._last_capture_at = None
            if previous_session_id is not None:
                self._reconnects += 1
            self._condition.notify_all()
            return StreamReconnectReceipt(
                previous_session_id=previous_session_id,
                session_id=session.session_id,
                connection_index=self._connection_index,
                reconnected_at=current,
            )

    def connect(self) -> StreamReconnectReceipt:
        with self._condition:
            if self._state is not BufferedConnectionState.NEW:
                raise StreamResilienceError(
                    "connection_state_invalid",
                    "Initial connect is allowed only for a new buffering client.",
                )
            self._state = BufferedConnectionState.CONNECTING
        try:
            return self._open_new_client(previous_session_id=None)
        except Exception:
            with self._condition:
                self._state = BufferedConnectionState.DISCONNECTED
                self._condition.notify_all()
            raise

    def _has_capacity(self, size_bytes: int) -> bool:
        return (
            len(self._queue) < self._settings.max_buffered_chunks
            and self._queued_bytes + size_bytes <= self._settings.max_buffered_bytes
        )

    def enqueue_pcm(
        self,
        pcm: bytes,
        *,
        captured_at: datetime | None = None,
        timeout_seconds: float | None = None,
    ) -> BufferedChunkTicket:
        if not isinstance(pcm, bytes) or not pcm:
            raise StreamResilienceError(
                "invalid_payload", "Buffered PCM must be nonempty bytes."
            )
        bytes_per_frame = (
            self._audio_format.channels * self._audio_format.sample_width_bytes
        )
        if len(pcm) % bytes_per_frame:
            raise StreamResilienceError(
                "invalid_payload", "Buffered PCM is not aligned to its audio format."
            )
        sample_count = len(pcm) // bytes_per_frame
        if sample_count > self._audio_format.max_chunk_samples:
            raise StreamResilienceError(
                "chunk_too_large", "Buffered PCM exceeds the authorized chunk size."
            )
        if len(pcm) > self._settings.max_buffered_bytes:
            raise StreamResilienceError(
                "chunk_exceeds_buffer", "One PCM chunk exceeds the entire buffer limit."
            )

        current = _now(self._clock)
        captured = (
            _aware(captured_at, "capture time")
            if captured_at is not None
            else current
        )
        if captured > current:
            raise StreamResilienceError(
                "invalid_time", "Buffered PCM capture time cannot be in the future."
            )

        timeout = (
            self._settings.block_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or timeout > 60
        ):
            raise ValueError("timeout_seconds must be within (0, 60]")
        deadline = time.monotonic() + float(timeout)

        with self._condition:
            while True:
                if self._state is not BufferedConnectionState.CONNECTED:
                    raise StreamResilienceError(
                        "not_connected",
                        "PCM can be buffered only for an active authorized session.",
                    )
                session = self._client.session if self._client is not None else None
                if session is None:
                    self._state = BufferedConnectionState.DISCONNECTED
                    raise StreamResilienceError(
                        "not_connected", "The streaming connection has no active session."
                    )
                observed = _now(self._clock)
                if observed >= session.expires_at:
                    self._state = BufferedConnectionState.DISCONNECTED
                    self._condition.notify_all()
                    raise StreamResilienceError(
                        "authorization_expired",
                        "The active stream authorization expired before buffering.",
                    )
                if not session.authorized_at <= captured < session.expires_at:
                    raise StreamResilienceError(
                        "capture_not_authorized",
                        "PCM capture time falls outside the active stream authorization.",
                    )
                if self._last_capture_at is not None and captured < self._last_capture_at:
                    raise StreamResilienceError(
                        "capture_out_of_order",
                        "PCM capture times must be nondecreasing within a connection.",
                    )
                if self._has_capacity(len(pcm)):
                    break
                self._backpressure_events += 1
                if self._settings.backpressure_mode is BackpressureMode.REJECT:
                    raise StreamResilienceError(
                        "backpressure",
                        "The bounded PCM buffer is full; no audio was enqueued.",
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StreamResilienceError(
                        "backpressure_timeout",
                        "The bounded PCM buffer did not free capacity before timeout.",
                    )
                self._condition.wait(remaining)

            digest = hashlib.sha256(pcm).hexdigest()
            index = self._next_enqueue_index
            self._next_enqueue_index += 1
            ticket = BufferedChunkTicket(
                ticket_id=f"buffer-{index:012d}-{digest[:16]}",
                enqueue_index=index,
                sample_count=sample_count,
                size_bytes=len(pcm),
                payload_sha256=digest,
                captured_at=captured,
                enqueued_at=observed,
            )
            try:
                owned = bytearray(pcm)
            except MemoryError as error:
                raise StreamResilienceError(
                    "insufficient_memory", "The PCM chunk could not be buffered."
                ) from error
            self._queue.append(_BufferedChunk(ticket=ticket, payload=owned))
            self._queued_bytes += len(owned)
            self._last_capture_at = captured
            return ticket

    def flush(self, *, max_chunks: int | None = None) -> BufferFlushReceipt:
        if max_chunks is not None and (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or max_chunks <= 0
        ):
            raise ValueError("max_chunks must be a positive integer or None")
        if not self._flush_lock.acquire(blocking=False):
            raise StreamResilienceError(
                "flush_in_progress", "Another buffer flush is already in progress."
            )
        delivered_chunks = 0
        delivered_bytes = 0
        final_receipt: StreamChunkReceipt | None = None
        try:
            while max_chunks is None or delivered_chunks < max_chunks:
                with self._condition:
                    if self._state is not BufferedConnectionState.CONNECTED:
                        raise StreamResilienceError(
                            "not_connected", "Buffered PCM cannot flush while disconnected."
                        )
                    if not self._queue:
                        break
                    client = self._client
                    if client is None or client.session is None:
                        self._state = BufferedConnectionState.DISCONNECTED
                        raise StreamResilienceError(
                            "not_connected", "The active streaming connection was lost."
                        )
                    entry = self._queue[0]
                    self._inflight = True
                    pcm = bytes(entry.payload)
                    session_id = client.session.session_id

                try:
                    receipt = client.send_pcm(
                        pcm,
                        captured_at=entry.ticket.captured_at,
                    )
                except Exception as error:
                    upstream = getattr(error, "code", None)
                    cause_code = upstream if isinstance(upstream, str) else None
                    with self._condition:
                        self._inflight = False
                        if cause_code in {
                            "authorization_expired",
                            "session_not_active",
                            "client_session_missing",
                        }:
                            self._state = BufferedConnectionState.DISCONNECTED
                            code = "reconnect_required"
                            message = (
                                "The stream ended with queued PCM; explicitly discard it "
                                "before reconnecting under new authorization."
                            )
                        elif cause_code == "sink_failed":
                            self._deferred_deliveries += 1
                            code = "delivery_deferred"
                            message = (
                                "Downstream backpressure deferred the oldest PCM chunk."
                            )
                        else:
                            code = "delivery_failed"
                            message = "The oldest buffered PCM chunk was not accepted."
                        self._condition.notify_all()
                    raise StreamResilienceError(
                        code, message, cause_code=cause_code
                    ) from error

                expected = entry.ticket
                if (
                    not isinstance(receipt, StreamChunkReceipt)
                    or receipt.session_id != session_id
                    or receipt.sample_count != expected.sample_count
                    or receipt.payload_sha256 != expected.payload_sha256
                ):
                    with self._condition:
                        self._inflight = False
                        self._state = BufferedConnectionState.DISCONNECTED
                        self._condition.notify_all()
                    raise StreamResilienceError(
                        "receipt_mismatch",
                        "Delivery returned an inconsistent receipt; queued PCM was retained.",
                    )

                with self._condition:
                    if not self._queue or self._queue[0] is not entry:
                        self._inflight = False
                        self._state = BufferedConnectionState.DISCONNECTED
                        raise StreamResilienceError(
                            "buffer_inconsistent",
                            "The PCM queue changed during verified delivery.",
                        )
                    self._queue.popleft()
                    self._queued_bytes -= expected.size_bytes
                    entry.erase()
                    self._inflight = False
                    delivered_chunks += 1
                    delivered_bytes += expected.size_bytes
                    self._delivered_chunks += 1
                    self._delivered_bytes += expected.size_bytes
                    final_receipt = receipt
                    self._condition.notify_all()

            with self._condition:
                active = self._client.session if self._client is not None else None
                if active is None:
                    raise StreamResilienceError(
                        "not_connected", "The streaming connection ended during flush."
                    )
                return BufferFlushReceipt(
                    session_id=active.session_id,
                    delivered_chunks=delivered_chunks,
                    delivered_bytes=delivered_bytes,
                    remaining_chunks=len(self._queue),
                    remaining_bytes=self._queued_bytes,
                    final_sequence_number=(
                        final_receipt.sequence_number
                        if final_receipt is not None
                        else None
                    ),
                    final_end_sample=(
                        final_receipt.end_sample if final_receipt is not None else None
                    ),
                )
        finally:
            with self._condition:
                self._inflight = False
                self._condition.notify_all()
            self._flush_lock.release()

    def discard_buffer(
        self,
        reason: BufferDiscardReason,
    ) -> BufferDiscardReceipt:
        if not isinstance(reason, BufferDiscardReason):
            raise TypeError("reason must be BufferDiscardReason")
        current = _now(self._clock)
        with self._condition:
            if self._inflight:
                raise StreamResilienceError(
                    "flush_in_progress", "PCM cannot be discarded during delivery."
                )
            entries = tuple(self._queue)
            self._queue.clear()
            discarded_bytes = self._queued_bytes
            self._queued_bytes = 0
            for entry in entries:
                entry.erase()
            self._discarded_chunks += len(entries)
            self._discarded_bytes += discarded_bytes
            self._condition.notify_all()
            return BufferDiscardReceipt(
                reason=reason,
                discarded_chunks=len(entries),
                discarded_bytes=discarded_bytes,
                ticket_ids=tuple(entry.ticket.ticket_id for entry in entries),
                discarded_at=current,
            )

    def _close_previous(
        self,
        client: StreamingClientConnection | None,
        reason: StreamCloseReason,
    ) -> str | None:
        if client is None or client.session is None:
            return None
        previous_id = client.session.session_id
        try:
            client.close(reason)
        except Exception as error:
            upstream = getattr(error, "code", None)
            cause_code = upstream if isinstance(upstream, str) else None
            if cause_code != "session_not_active":
                raise StreamResilienceError(
                    "reconnect_deferred",
                    "The previous stream could not close before reconnect.",
                    cause_code=cause_code,
                ) from error
        return previous_id

    def reconnect(
        self,
        *,
        reason: StreamCloseReason = StreamCloseReason.DEVICE_DISCONNECTED,
        consent: ConsentRecord | None = None,
    ) -> StreamReconnectReceipt:
        if not isinstance(reason, StreamCloseReason):
            raise TypeError("reason must be StreamCloseReason")
        if consent is not None and not isinstance(consent, ConsentRecord):
            raise TypeError("consent must be ConsentRecord or None")
        with self._condition:
            if self._state in {
                BufferedConnectionState.NEW,
                BufferedConnectionState.CONNECTING,
                BufferedConnectionState.CLOSED,
            }:
                raise StreamResilienceError(
                    "connection_state_invalid",
                    "Reconnect requires a previously opened live connection.",
                )
            if self._queue or self._queued_bytes:
                raise StreamResilienceError(
                    "buffer_not_empty",
                    "Queued PCM cannot cross into a newly authorized stream.",
                )
            if self._inflight:
                raise StreamResilienceError(
                    "flush_in_progress", "Reconnect cannot begin during delivery."
                )
            client = self._client
            self._state = BufferedConnectionState.CONNECTING
        try:
            previous_id = self._close_previous(client, reason)
            if consent is not None:
                replacement = ConsentRecord.model_validate(
                    consent.model_dump(mode="python")
                )
                with self._condition:
                    self._consent = replacement
            return self._open_new_client(previous_session_id=previous_id)
        except Exception:
            with self._condition:
                self._state = BufferedConnectionState.DISCONNECTED
                self._condition.notify_all()
            raise

    def close(
        self,
        reason: StreamCloseReason = StreamCloseReason.CLIENT_REQUEST,
    ) -> StreamCloseReceipt | None:
        if not isinstance(reason, StreamCloseReason):
            raise TypeError("reason must be StreamCloseReason")
        with self._condition:
            if self._state is BufferedConnectionState.CLOSED:
                return None
            if self._queue or self._queued_bytes:
                raise StreamResilienceError(
                    "buffer_not_empty",
                    "Flush or explicitly discard queued PCM before closing.",
                )
            if self._inflight:
                raise StreamResilienceError(
                    "flush_in_progress", "Close cannot begin during delivery."
                )
            client = self._client
        receipt: StreamCloseReceipt | None = None
        if client is not None and client.session is not None:
            try:
                receipt = client.close(reason)
            except Exception as error:
                upstream = getattr(error, "code", None)
                cause_code = upstream if isinstance(upstream, str) else None
                if cause_code != "session_not_active":
                    raise StreamResilienceError(
                        "close_deferred",
                        "The active stream could not close cleanly.",
                        cause_code=cause_code,
                    ) from error
        with self._condition:
            self._client = None
            self._state = BufferedConnectionState.CLOSED
            self._condition.notify_all()
        return receipt
