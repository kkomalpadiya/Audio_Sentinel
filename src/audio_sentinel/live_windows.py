"""A10.2 deterministic rolling windows for authorized live PCM streams."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from threading import RLock
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.config import (
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_WINDOW_SECONDS,
    AudioSettings,
    WindowSeconds,
)
from audio_sentinel.live_streaming import (
    AuthorizedStreamSession,
    StreamAudioChunk,
    StreamChunkReceipt,
    StreamCloseReceipt,
)


DEFAULT_MAX_BUFFER_BYTES = 67_108_864
DEFAULT_MAX_ACTIVE_STREAMS = 32
DEFAULT_MAX_WINDOWS_PER_HANDOFF = 4_096


class RollingWindowError(RuntimeError):
    """Stable A10.2 failure code with a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RollingWindowSettings(BaseModel):
    """Bounded live window grid compatible with offline window geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    target_sample_rate_hz: int = Field(
        default=DEFAULT_SAMPLE_RATE_HZ,
        ge=8_000,
        le=192_000,
        strict=True,
    )
    window_seconds: tuple[WindowSeconds, ...] = DEFAULT_WINDOW_SECONDS
    window_overlap_ratio: float = Field(default=0.5, ge=0, lt=1)
    close_tail_policy: Literal["pad", "drop"] = "pad"
    max_buffer_bytes: int = Field(
        default=DEFAULT_MAX_BUFFER_BYTES,
        gt=0,
        strict=True,
    )
    max_active_streams: int = Field(
        default=DEFAULT_MAX_ACTIVE_STREAMS,
        gt=0,
        strict=True,
    )
    max_windows_per_handoff: int = Field(
        default=DEFAULT_MAX_WINDOWS_PER_HANDOFF,
        gt=0,
        strict=True,
    )

    @field_validator("window_seconds")
    @classmethod
    def validate_window_seconds(
        cls, values: tuple[float, ...]
    ) -> tuple[float, ...]:
        if not values:
            raise ValueError("window_seconds must include at least one duration")
        if tuple(sorted(values)) != values or len(set(values)) != len(values):
            raise ValueError(
                "window_seconds must be unique and ordered from shortest to longest"
            )
        return values

    @model_validator(mode="after")
    def validate_window_grid(self) -> "RollingWindowSettings":
        largest = 0
        for seconds in self.window_seconds:
            length, hop = self.window_sample_counts(seconds)
            if not 1 <= hop <= length:
                raise ValueError("window length and hop must span at least one sample")
            largest = max(largest, length)
        if largest * np.dtype(np.float32).itemsize > self.max_buffer_bytes:
            raise ValueError("max_buffer_bytes cannot hold the largest live window")
        return self

    def window_sample_counts(self, seconds: float) -> tuple[int, int]:
        """Match ``AudioSettings.window_sample_counts`` exactly."""

        length = round(seconds * self.target_sample_rate_hz)
        return length, round(length * (1 - self.window_overlap_ratio))

    @classmethod
    def from_audio_settings(
        cls,
        settings: AudioSettings,
        *,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
        max_active_streams: int = DEFAULT_MAX_ACTIVE_STREAMS,
        max_windows_per_handoff: int = DEFAULT_MAX_WINDOWS_PER_HANDOFF,
    ) -> "RollingWindowSettings":
        """Copy the offline sample grid without claiming clip-global transforms."""

        if not isinstance(settings, AudioSettings):
            raise TypeError("settings must be AudioSettings")
        if not settings.convert_to_mono:
            raise ValueError("live rolling windows require mono pipeline output")
        return cls(
            target_sample_rate_hz=settings.target_sample_rate_hz,
            window_seconds=settings.window_seconds,
            window_overlap_ratio=settings.window_overlap_ratio,
            close_tail_policy=settings.tail_policy,
            max_buffer_bytes=max_buffer_bytes,
            max_active_streams=max_active_streams,
            max_windows_per_handoff=max_windows_per_handoff,
        )


class LiveWindowRecord(BaseModel):
    """Audio-free metadata for one emitted live analysis window."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )

    window_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    session_id: str = Field(min_length=3, max_length=128)
    group_index: int = Field(ge=0, strict=True)
    window_index: int = Field(ge=0, strict=True)
    window_seconds: float = Field(gt=0)
    sample_rate_hz: int = Field(gt=0, strict=True)
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(ge=0, strict=True)
    padding_samples: int = Field(ge=0, strict=True)
    emitted_after_sequence_number: int = Field(ge=0, strict=True)
    audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_audio_persisted: Literal[False] = False

    @model_validator(mode="after")
    def validate_span(self) -> "LiveWindowRecord":
        length = round(self.window_seconds * self.sample_rate_hz)
        if self.end_sample < self.start_sample:
            raise ValueError("end_sample cannot precede start_sample")
        if self.end_sample - self.start_sample + self.padding_samples != length:
            raise ValueError("real span plus padding must equal the window length")
        return self


@dataclass(frozen=True)
class LiveAudioWindow:
    """One owned, read-only float32 mono window plus audio-free provenance."""

    record: LiveWindowRecord
    samples: NDArray[np.float32] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        samples = self.samples
        expected = round(self.record.window_seconds * self.record.sample_rate_hz)
        if (
            not isinstance(samples, np.ndarray)
            or samples.dtype != np.dtype(np.float32)
            or samples.shape != (expected,)
            or not samples.flags.c_contiguous
            or samples.flags.writeable
            or not np.isfinite(samples).all()
            or np.any(samples < -1)
            or np.any(samples > 1)
        ):
            raise ValueError(
                "samples must be a read-only contiguous finite float32 mono window"
            )
        digest = hashlib.sha256(samples.astype("<f4", copy=False).tobytes()).hexdigest()
        if digest != self.record.audio_sha256:
            raise ValueError("audio_sha256 must match the canonical float32 samples")


class LiveWindowGroupSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    group_index: int = Field(ge=0, strict=True)
    window_seconds: float = Field(gt=0)
    window_samples: int = Field(gt=0, strict=True)
    hop_samples: int = Field(gt=0, strict=True)
    emitted_windows: int = Field(ge=0, strict=True)
    padded_windows: int = Field(ge=0, strict=True)
    uncovered_tail_samples: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_counts(self) -> "LiveWindowGroupSummary":
        if self.padded_windows > self.emitted_windows:
            raise ValueError("padded_windows cannot exceed emitted_windows")
        return self


class LiveWindowCloseSummary(BaseModel):
    """Audio-free final accounting for one rolling live stream."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    session_id: str = Field(min_length=3, max_length=128)
    total_input_samples: int = Field(ge=0, strict=True)
    total_emitted_windows: int = Field(ge=0, strict=True)
    tail_policy: Literal["pad", "drop"]
    groups: tuple[LiveWindowGroupSummary, ...]
    raw_audio_persisted: Literal[False] = False

    @model_validator(mode="after")
    def validate_totals(self) -> "LiveWindowCloseSummary":
        if self.total_emitted_windows != sum(
            item.emitted_windows for item in self.groups
        ):
            raise ValueError("total_emitted_windows must match the group inventory")
        return self


@runtime_checkable
class LiveWindowConsumer(Protocol):
    """Synchronous downstream boundary for rolling windows and final accounting."""

    def accept_window(
        self,
        session: AuthorizedStreamSession,
        window: LiveAudioWindow,
    ) -> None: ...

    def close_live_windows(
        self,
        session: AuthorizedStreamSession,
        receipt: StreamCloseReceipt,
        summary: LiveWindowCloseSummary,
    ) -> None: ...


@dataclass
class _GroupState:
    group_index: int
    seconds: float
    length: int
    hop: int
    next_start: int = 0
    emitted: int = 0
    padded: int = 0
    last_real_end: int = 0


@dataclass
class _PendingChunk:
    chunk: StreamAudioChunk
    receipt: StreamChunkReceipt


@dataclass
class _PendingClose:
    receipt: StreamCloseReceipt
    expected_counts: tuple[int, ...]


@dataclass
class _StreamState:
    session: AuthorizedStreamSession
    namespace: str
    buffer_start: int
    total_samples: int
    buffer: NDArray[np.float32]
    groups: list[_GroupState]
    last_sequence_number: int = -1
    pending_chunk: _PendingChunk | None = None
    pending_close: _PendingClose | None = None


def _session_namespace(session: AuthorizedStreamSession) -> str:
    payload = json.dumps(
        session.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_pcm(
    session: AuthorizedStreamSession,
    chunk: StreamAudioChunk,
) -> NDArray[np.float32]:
    audio_format = session.audio_format
    try:
        frames = np.frombuffer(chunk.payload, dtype="<i2").reshape(
            chunk.sample_count,
            audio_format.channels,
        )
        samples = frames.astype(np.float32)
        if audio_format.channels > 1:
            samples = samples.mean(axis=1, dtype=np.float32)
        else:
            samples = samples[:, 0]
        samples *= np.float32(1.0 / 32768.0)
        return np.ascontiguousarray(samples, dtype=np.float32)
    except (MemoryError, TypeError, ValueError) as error:
        raise RollingWindowError(
            "decode_failed", "The accepted PCM chunk could not be decoded."
        ) from error


class RollingLiveWindowSink:
    """Convert accepted PCM chunks into deterministic overlapping live windows.

    Calls are synchronous. Successfully delivered windows are checkpointed before
    the next callback, so retrying the same failed chunk or close resumes at the
    first undelivered window instead of emitting earlier windows again.
    """

    def __init__(
        self,
        consumer: LiveWindowConsumer,
        *,
        settings: RollingWindowSettings | None = None,
    ) -> None:
        if not isinstance(consumer, LiveWindowConsumer):
            raise TypeError("consumer must implement LiveWindowConsumer")
        self._consumer = consumer
        self._settings = RollingWindowSettings.model_validate(
            (settings or RollingWindowSettings()).model_dump()
        )
        self._streams: dict[str, _StreamState] = {}
        self._lock = RLock()

    @property
    def settings(self) -> RollingWindowSettings:
        return self._settings

    @property
    def active_stream_count(self) -> int:
        with self._lock:
            return len(self._streams)

    def _new_state(self, session: AuthorizedStreamSession) -> _StreamState:
        if len(self._streams) >= self._settings.max_active_streams:
            raise RollingWindowError(
                "stream_capacity_reached",
                "The rolling-window stream capacity has been reached.",
            )
        if session.audio_format.sample_rate_hz != self._settings.target_sample_rate_hz:
            raise RollingWindowError(
                "sample_rate_mismatch",
                "Live PCM sample rate must match the rolling-window sample grid.",
            )
        groups = [
            _GroupState(index, seconds, *self._settings.window_sample_counts(seconds))
            for index, seconds in enumerate(self._settings.window_seconds)
        ]
        empty = np.empty(0, dtype=np.float32)
        state = _StreamState(
            session=session,
            namespace=_session_namespace(session),
            buffer_start=0,
            total_samples=0,
            buffer=empty,
            groups=groups,
        )
        self._streams[session.session_id] = state
        return state

    def _state(self, session: AuthorizedStreamSession) -> _StreamState:
        state = self._streams.get(session.session_id)
        if state is None:
            return self._new_state(session)
        if state.session != session:
            raise RollingWindowError(
                "session_mismatch", "The live session differs from rolling-window state."
            )
        return state

    @staticmethod
    def _validate_handoff(
        session: AuthorizedStreamSession,
        chunk: StreamAudioChunk,
        receipt: StreamChunkReceipt,
    ) -> None:
        if not isinstance(session, AuthorizedStreamSession):
            raise RollingWindowError(
                "invalid_session", "The rolling-window session is invalid."
            )
        if not isinstance(chunk, StreamAudioChunk) or not isinstance(
            receipt, StreamChunkReceipt
        ):
            raise RollingWindowError(
                "invalid_chunk", "The rolling-window chunk handoff is invalid."
            )
        digest = hashlib.sha256(chunk.payload).hexdigest()
        expected_bytes = (
            chunk.sample_count
            * session.audio_format.channels
            * session.audio_format.sample_width_bytes
        )
        if (
            chunk.session_id != session.session_id
            or receipt.session_id != session.session_id
            or chunk.sample_count > session.audio_format.max_chunk_samples
            or len(chunk.payload) != expected_bytes
            or receipt.sequence_number != chunk.sequence_number
            or receipt.start_sample != chunk.start_sample
            or receipt.end_sample != chunk.start_sample + chunk.sample_count
            or receipt.sample_count != chunk.sample_count
            or receipt.captured_at != chunk.captured_at
            or receipt.payload_sha256 != digest
            or receipt.raw_audio_persisted
        ):
            raise RollingWindowError(
                "handoff_mismatch",
                "Chunk receipt does not match the accepted live PCM payload.",
            )

    def _ready_count(self, state: _StreamState, total_samples: int) -> int:
        count = 0
        for group in state.groups:
            if group.next_start + group.length <= total_samples:
                count += 1 + (
                    total_samples - group.next_start - group.length
                ) // group.hop
        return count

    def _window(self, state: _StreamState, group: _GroupState) -> LiveAudioWindow:
        start = group.next_start
        end = min(start + group.length, state.total_samples)
        padding = group.length - (end - start)
        relative_start = start - state.buffer_start
        relative_end = end - state.buffer_start
        if relative_start < 0 or relative_end > len(state.buffer):
            raise RollingWindowError(
                "buffer_inconsistent",
                "Rolling-window state no longer contains the required samples.",
            )
        try:
            samples = np.zeros(group.length, dtype=np.float32)
            samples[: end - start] = state.buffer[relative_start:relative_end]
        except MemoryError as error:
            raise RollingWindowError(
                "insufficient_memory", "A live audio window could not be allocated."
            ) from error
        samples.setflags(write=False)
        digest = hashlib.sha256(
            samples.astype("<f4", copy=False).tobytes()
        ).hexdigest()
        record = LiveWindowRecord(
            window_id=(
                f"{state.namespace}-g{group.group_index:04d}-s{start:012d}"
            ),
            session_id=state.session.session_id,
            group_index=group.group_index,
            window_index=group.emitted,
            window_seconds=group.seconds,
            sample_rate_hz=self._settings.target_sample_rate_hz,
            start_sample=start,
            end_sample=end,
            padding_samples=padding,
            emitted_after_sequence_number=state.last_sequence_number,
            audio_sha256=digest,
        )
        return LiveAudioWindow(record=record, samples=samples)

    @staticmethod
    def _next_complete_group(state: _StreamState) -> _GroupState | None:
        ready = [
            group
            for group in state.groups
            if group.next_start + group.length <= state.total_samples
        ]
        if not ready:
            return None
        return min(
            ready,
            key=lambda item: (
                item.next_start + item.length,
                item.group_index,
                item.next_start,
            ),
        )

    def _deliver_group(self, state: _StreamState, group: _GroupState) -> None:
        window = self._window(state, group)
        try:
            self._consumer.accept_window(state.session, window)
        except Exception as error:
            if isinstance(error, RollingWindowError):
                raise
            raise RollingWindowError(
                "consumer_failed", "The live window consumer rejected a window."
            ) from error
        group.emitted += 1
        if window.record.padding_samples:
            group.padded += 1
        group.last_real_end = window.record.end_sample
        group.next_start += group.hop
        self._trim(state)

    @staticmethod
    def _trim(state: _StreamState) -> None:
        retain_from = min(group.next_start for group in state.groups)
        if retain_from <= state.buffer_start:
            return
        drop = min(retain_from - state.buffer_start, len(state.buffer))
        state.buffer = np.array(
            state.buffer[drop:], dtype=np.float32, order="C", copy=True
        )
        state.buffer_start += drop

    def accept_chunk(
        self,
        session: AuthorizedStreamSession,
        chunk: StreamAudioChunk,
        receipt: StreamChunkReceipt,
    ) -> None:
        self._validate_handoff(session, chunk, receipt)
        with self._lock:
            created = session.session_id not in self._streams
            if created and (chunk.sequence_number != 0 or chunk.start_sample != 0):
                raise RollingWindowError(
                    "sample_position_mismatch",
                    "The first live PCM chunk must begin at sequence and sample zero.",
                )
            state = self._state(session)
            try:
                if state.pending_close is not None:
                    raise RollingWindowError(
                        "close_pending", "The stream already has a pending close handoff."
                    )
                pending = state.pending_chunk
                if pending is not None:
                    if pending.chunk != chunk or pending.receipt != receipt:
                        raise RollingWindowError(
                            "chunk_pending",
                            "A different chunk cannot replace a failed rolling handoff.",
                        )
                else:
                    if chunk.start_sample != state.total_samples:
                        raise RollingWindowError(
                            "sample_position_mismatch",
                            "Live PCM does not continue from the rolling-window position.",
                        )
                    ready = self._ready_count(
                        state, state.total_samples + chunk.sample_count
                    )
                    if ready > self._settings.max_windows_per_handoff:
                        raise RollingWindowError(
                            "too_many_windows",
                            "One PCM handoff would emit too many live windows.",
                        )
                    decoded = _decode_pcm(session, chunk)
                    combined_bytes = (len(state.buffer) + len(decoded)) * 4
                    if combined_bytes > self._settings.max_buffer_bytes:
                        raise RollingWindowError(
                            "buffer_limit_exceeded",
                            "Rolling live audio exceeded the configured memory limit.",
                        )
                    try:
                        state.buffer = np.concatenate((state.buffer, decoded))
                    except MemoryError as error:
                        raise RollingWindowError(
                            "insufficient_memory",
                            "The live PCM chunk could not be buffered.",
                        ) from error
                    state.total_samples += chunk.sample_count
                    state.last_sequence_number = chunk.sequence_number
                    state.pending_chunk = _PendingChunk(chunk=chunk, receipt=receipt)

                while (group := self._next_complete_group(state)) is not None:
                    self._deliver_group(state, group)
                state.pending_chunk = None
            except Exception:
                if created and state.total_samples == 0 and state.pending_chunk is None:
                    self._streams.pop(session.session_id, None)
                raise

    @staticmethod
    def _expected_count(
        total_samples: int,
        group: _GroupState,
        tail_policy: Literal["pad", "drop"],
    ) -> int:
        if total_samples == 0:
            return 0
        remainder = total_samples - group.length
        if tail_policy == "pad":
            return 1 + max(0, (remainder + group.hop - 1) // group.hop)
        return 0 if remainder < 0 else 1 + remainder // group.hop

    @staticmethod
    def _next_close_group(
        state: _StreamState,
        expected: tuple[int, ...],
    ) -> _GroupState | None:
        pending = [
            group
            for group, count in zip(state.groups, expected, strict=True)
            if group.emitted < count
        ]
        if not pending:
            return None
        return min(
            pending,
            key=lambda item: (
                min(item.next_start + item.length, state.total_samples),
                item.group_index,
                item.next_start,
            ),
        )

    def _summary(self, state: _StreamState) -> LiveWindowCloseSummary:
        groups = tuple(
            LiveWindowGroupSummary(
                group_index=group.group_index,
                window_seconds=group.seconds,
                window_samples=group.length,
                hop_samples=group.hop,
                emitted_windows=group.emitted,
                padded_windows=group.padded,
                uncovered_tail_samples=max(
                    0, state.total_samples - group.last_real_end
                ),
            )
            for group in state.groups
        )
        return LiveWindowCloseSummary(
            session_id=state.session.session_id,
            total_input_samples=state.total_samples,
            total_emitted_windows=sum(item.emitted_windows for item in groups),
            tail_policy=self._settings.close_tail_policy,
            groups=groups,
        )

    def close_stream(
        self,
        session: AuthorizedStreamSession,
        receipt: StreamCloseReceipt,
    ) -> None:
        if not isinstance(session, AuthorizedStreamSession) or not isinstance(
            receipt, StreamCloseReceipt
        ):
            raise RollingWindowError(
                "invalid_close", "The rolling-window close handoff is invalid."
            )
        if receipt.session_id != session.session_id:
            raise RollingWindowError(
                "close_mismatch", "The close receipt does not match the live session."
            )
        with self._lock:
            created = session.session_id not in self._streams
            if created and (
                receipt.next_start_sample != 0
                or receipt.next_sequence_number != 0
            ):
                raise RollingWindowError(
                    "sample_position_mismatch",
                    "An empty live stream must close at sequence and sample zero.",
                )
            state = self._state(session)
            if state.pending_chunk is not None:
                raise RollingWindowError(
                    "chunk_pending", "The stream still has a pending PCM handoff."
                )
            if receipt.next_start_sample != state.total_samples:
                raise RollingWindowError(
                    "sample_position_mismatch",
                    "The close position differs from rolling-window input.",
                )
            if receipt.next_sequence_number != state.last_sequence_number + 1:
                raise RollingWindowError(
                    "sequence_mismatch",
                    "The close sequence differs from rolling-window input.",
                )
            pending = state.pending_close
            if pending is None:
                expected = tuple(
                    self._expected_count(
                        state.total_samples,
                        group,
                        self._settings.close_tail_policy,
                    )
                    for group in state.groups
                )
                remaining = sum(
                    count - group.emitted
                    for group, count in zip(state.groups, expected, strict=True)
                )
                if remaining > self._settings.max_windows_per_handoff:
                    raise RollingWindowError(
                        "too_many_windows",
                        "Closing the stream would emit too many live windows.",
                    )
                pending = _PendingClose(receipt=receipt, expected_counts=expected)
                state.pending_close = pending
            elif pending.receipt != receipt:
                raise RollingWindowError(
                    "close_pending",
                    "A different close cannot replace a failed rolling handoff.",
                )

            while (
                group := self._next_close_group(state, pending.expected_counts)
            ) is not None:
                self._deliver_group(state, group)
            summary = self._summary(state)
            try:
                self._consumer.close_live_windows(session, receipt, summary)
            except Exception as error:
                if isinstance(error, RollingWindowError):
                    raise
                raise RollingWindowError(
                    "consumer_failed",
                    "The live window consumer rejected final stream accounting.",
                ) from error
            del self._streams[session.session_id]
