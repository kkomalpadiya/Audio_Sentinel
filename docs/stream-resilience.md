# Live stream buffering and reconnect behavior

## Scope

B10.2 adds a bounded capture-side PCM queue and explicit connection lifecycle
around the B10.1 local client. `BufferedStreamingClient` accepts a factory that
creates authenticated `LocalStreamingClient` connections, but the controller is
defined against the smaller `StreamingClientConnection` protocol for testing and
future transport adapters.

The implementation remains local and in-process. It does not open a microphone,
create a socket, discover a device, persist PCM, run a model, deliver an alert,
or authorize external action. A real network transport would need an additional
protocol with equivalent authorization and idempotency guarantees.

## Bounded PCM queue

`StreamBufferSettings` defines:

- the maximum queued chunk count;
- the maximum total queued bytes;
- `reject` or `block` behavior when either limit is reached;
- the maximum time a blocking producer may wait for capacity.

The byte limit must hold one maximum authorized stream chunk. Every enqueued
payload is validated against the authorized PCM geometry before allocation. The
queue owns a `bytearray` copy so delivered or explicitly discarded bytes can be
overwritten before their entry is released.

`enqueue_pcm(...)` requires an active authorized session. Capture timestamps must
be timezone-aware, no later than the controller clock, inside the current session
authorization interval, and nondecreasing within that connection. The returned
`BufferedChunkTicket` contains only an opaque queue identity, size, sample count,
timestamp, and SHA-256. It contains no PCM.

## Backpressure

No policy silently drops or replaces audio.

- `reject` raises `backpressure` immediately and leaves the queue unchanged.
- `block` waits on bounded queue capacity. It returns when a flush or explicit
  discard frees space, or raises `backpressure_timeout` at the configured limit.
- A session that expires while a producer waits changes to `disconnected` and
  rejects the new payload instead of buffering it under expired authorization.

Only one flush may run at a time. Producers may continue enqueueing while the
oldest entry is being delivered, subject to the same limits.

## Ordered delivery and retry

`flush(...)` sends the queue head through the active B10.1 client. The entry is
removed only after the returned receipt matches the active session, sample count,
and payload SHA-256. Successful removal overwrites the queue's mutable PCM copy
and wakes blocked producers.

If B10.1 reports `sink_failed`, the queue head remains unchanged and B10.2 raises
`delivery_deferred`. A later flush retries that exact payload on the same session
and sample position. This is the normal response when the synchronous A10.2
consumer cannot accept more work.

An inconsistent receipt changes the connection to `disconnected` and retains the
entry. It cannot be resent automatically because the remote position would be
ambiguous. The caller must resolve and explicitly discard the entry before a new
session can open.

## Reconnect boundary

Every reconnect closes the old session when it still exists, performs a fresh
device challenge and authentication through the client factory, and opens a new
session at sequence and sample position zero. `StreamReconnectReceipt` records
the old and new session IDs and explicitly records that zero chunks and zero bytes
crossed the boundary.

Queued PCM never moves to the new session. Its capture timestamp belongs to the
old authorization and may precede the new session. Therefore:

1. A clean reconnect requires an empty queue.
2. If the old session disappears or expires with queued PCM, `flush` raises
   `reconnect_required` and preserves that PCM.
3. The caller must either restore the old session and deliver the queue, or call
   `discard_buffer(...)` with an explicit reason.
4. Only then may `reconnect(...)` open fresh authorization for future capture.

Reconnect can receive a refreshed `ConsentRecord`. Scope and audio format remain
fixed for the controller. A close rejected by downstream backpressure raises
`reconnect_deferred`; B10.2 does not abandon the old session or create a parallel
one.

## Explicit discard and close

`discard_buffer(...)` requires a reason such as connection loss, authorization
expiry, service shutdown, or operator request. It overwrites every queued PCM
copy and returns audio-free ticket/count/byte accounting. It does not pretend the
discarded samples were analyzed.

`close(...)` refuses to proceed while PCM is queued or delivery is in flight.
The caller must flush or explicitly discard first. Downstream close backpressure
returns `close_deferred` and leaves the connection available for retry. A
successful close moves the controller to its terminal `closed` state.

## Status and privacy

`status()` returns only connection state, the current session ID, queue counts,
the oldest capture time, and cumulative delivered, discarded, backpressure,
deferred, and reconnect counters. It never returns PCM. All public receipts state
that raw audio was not persisted.

The queue is process memory, not durable storage. A process crash loses queued
audio. This is deliberate: adding durable retry would require a separately
approved encrypted-retention design tied to consent and deletion policy.

## Minimal integration

```python
from audio_sentinel.stream_resilience import (
    BackpressureMode,
    BufferedStreamingClient,
    StreamBufferSettings,
)
from audio_sentinel.streaming_ingestion import LocalStreamingClient


def client_factory():
    return LocalStreamingClient(
        service,
        enrollment_id=enrollment_id,
        device_id=device_id,
        credential_provider=credential_provider,
    )


buffered = BufferedStreamingClient(
    client_factory,
    consent=current_consent,
    requested_scope=requested_scope,
    audio_format=authorized_format,
    settings=StreamBufferSettings(
        max_buffered_chunks=128,
        max_buffered_bytes=16_777_216,
        backpressure_mode=BackpressureMode.REJECT,
    ),
)

buffered.connect()
buffered.enqueue_pcm(captured_pcm, captured_at=captured_at)
buffered.flush()
buffered.close()
```

Credential, consent, and capture providers must be trusted local integrations.
Do not hardcode production credentials or treat queue acceptance as evidence of
an incident.

## Stable failure groups

- `backpressure`, `backpressure_timeout`, `chunk_exceeds_buffer`: capacity.
- `delivery_deferred`, `close_deferred`, `reconnect_deferred`: retryable
  downstream refusal.
- `reconnect_required`, `buffer_not_empty`: explicit old-session resolution is
  required before new authorization.
- `connection_failed`, `connection_mismatch`, `not_connected`,
  `connection_state_invalid`: connection lifecycle.
- `receipt_mismatch`, `buffer_inconsistent`, `delivery_failed`: ambiguous or
  invalid delivery; PCM remains queued for explicit handling.
- `capture_not_authorized`, `capture_out_of_order`, `authorization_expired`,
  `invalid_time`: capture authorization and time.
- `invalid_payload`, `chunk_too_large`, `insufficient_memory`: PCM validation and
  allocation.

Failure messages contain no PCM, credentials, challenges, or proofs.

## Verification

Run the focused suite:

```powershell
python -m pytest tests\test_stream_resilience.py -q
```

Run all live-stream layers:

```powershell
python -m pytest tests\test_live_streaming.py tests\test_streaming_ingestion.py tests\test_live_windows.py tests\test_stream_resilience.py -q
```

## Next task

A10.3 will measure latency, reliability, retry/backpressure behavior, and timing
through the complete authorized live path. It must not convert performance data
into alert or external-action authority.
