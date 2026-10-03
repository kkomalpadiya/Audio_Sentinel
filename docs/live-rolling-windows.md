# Rolling live audio windows

## Scope

A10.2 connects the B10.1 synchronous ingestion boundary to deterministic,
in-memory analysis windows. `RollingLiveWindowSink` implements
`StreamingAudioSink`, so it can be supplied directly to
`LocalStreamingIngestionService`. It accepts only chunks already authorized and
validated by that service.

This task does not run an acoustic or speech model. It does not persist PCM or
window samples, make an incident decision, send a notification, open a device,
or expose a network endpoint. Buffer queues, reconnect behavior, asynchronous
workers, and backpressure policy remain B10.2.

## Window grid

`RollingWindowSettings` defines:

- the target sample rate;
- one or more ordered window durations;
- one overlap ratio shared by those durations;
- `pad` or `drop` behavior when a stream closes;
- memory, active-stream, and per-handoff window limits.

`RollingWindowSettings.from_audio_settings(...)` copies the target sample rate,
window durations, overlap, and tail policy from `AudioSettings`. The resulting
length and hop calculations use the same rounding rule as the offline
segmentation pipeline. The method requires mono pipeline output because the live
adapter always produces mono analysis windows.

This compatibility covers the sample grid only. Clip-global loudness
normalization and noise estimation cannot be reproduced safely from incomplete
live input and are not applied by A10.2.

## Chunk adaptation algorithm

For each active session the sink performs these steps under one lock:

1. Verify that the session, chunk, and B10.1 receipt refer to the same payload,
   sequence number, sample span, capture time, and SHA-256.
2. Require the stream sample rate to match the configured analysis grid.
3. Decode signed 16-bit little-endian PCM to float32 in `[-1, 1]`.
4. Downmix authorized multichannel frames by taking their float32 mean.
5. Append the mono samples to the bounded per-session rolling buffer.
6. Select every complete window in deterministic readiness order: earliest end
   sample, then duration group, then start sample.
7. Give each window synchronously to the configured `LiveWindowConsumer`.
8. After successful delivery, advance that duration's cursor and release samples
   that no duration can need again.

Each `LiveAudioWindow` owns a contiguous, read-only float32 array. Its
`LiveWindowRecord` carries the session ID, deterministic window ID, duration,
absolute real-audio span, padding, sample rate, source sequence position, and a
SHA-256 of the canonical float32 samples. The record states that raw audio was
not persisted.

Chunk boundaries do not affect the window grid. A window may contain samples
from several irregular chunks, and a large chunk may complete several windows.

## Close behavior

Complete windows are emitted as soon as enough samples exist. Incomplete tails
are considered only when the verified B10.1 close receipt arrives.

- `pad` emits the same final zero-padded grid that offline segmentation would
  create for a nonempty signal.
- `drop` emits no incomplete tail and records the uncovered trailing sample
  count for each duration.
- An empty stream emits no synthetic silence under either policy.

After all tail windows are accepted, the consumer receives one
`LiveWindowCloseSummary`. It records total input samples, total emitted windows,
the tail policy, and per-duration length, hop, emitted, padded, and uncovered-tail
counts. The summary contains no PCM.

## Retry semantics

B10.1 does not advance its accepted chunk position when a sink raises. A10.2
therefore checkpoints each successfully delivered window and keeps the exact
failed chunk or close as pending state. Retrying that same handoff resumes at the
first undelivered window. A different chunk or close cannot replace pending work.

This prevents the adapter from replaying windows whose consumer callback already
returned successfully. A consumer must still follow the usual synchronous
contract: it must not perform an external side effect and then raise as though
the callback failed. Durable or asynchronous delivery, queue acknowledgement,
and overload handling belong to B10.2.

## Minimal integration

```python
from audio_sentinel.live_windows import (
    LiveWindowConsumer,
    RollingLiveWindowSink,
    RollingWindowSettings,
)
from audio_sentinel.streaming_ingestion import LocalStreamingIngestionService


class AnalysisConsumer:
    def accept_window(self, session, window):
        # Run or enqueue approved local analysis here. Do not retain samples
        # unless current consent and a later approved design permit it.
        analyze(window.samples, window.record)

    def close_live_windows(self, session, receipt, summary):
        record_audio_free_summary(summary)


window_sink = RollingLiveWindowSink(
    AnalysisConsumer(),
    settings=RollingWindowSettings(
        target_sample_rate_hz=16_000,
        window_seconds=(1.0, 5.0, 10.0),
        window_overlap_ratio=0.5,
        close_tail_policy="pad",
    ),
)

service = LocalStreamingIngestionService(
    registry,
    authenticator,
    window_sink,
)
```

The example placeholders must be supplied by trusted local application code.
The adapter does not grant model, storage, notification, or external-action
authority.

## Stable failure groups

- `invalid_session`, `invalid_chunk`, `invalid_close`: wrong object types.
- `handoff_mismatch`, `close_mismatch`, `session_mismatch`: inconsistent
  verified metadata.
- `sample_rate_mismatch`, `sample_position_mismatch`, `sequence_mismatch`:
  incompatible or noncontiguous stream geometry.
- `stream_capacity_reached`, `buffer_limit_exceeded`, `too_many_windows`:
  configured resource limits.
- `decode_failed`, `insufficient_memory`, `buffer_inconsistent`: safe local
  conversion or state failures.
- `chunk_pending`, `close_pending`: a different handoff attempted to replace
  retryable work.
- `consumer_failed`: the synchronous downstream consumer rejected a window or
  final summary.

Messages are safe for local diagnostics and do not include PCM, credentials, or
authentication proofs.

## Verification

Run the focused live-window tests:

```powershell
python -m pytest tests\test_live_windows.py -q
```

Run the complete live boundary set:

```powershell
python -m pytest tests\test_live_streaming.py tests\test_streaming_ingestion.py tests\test_live_windows.py -q
```

## Next task

B10.2 will define bounded queueing, reconnect identity and position checks,
backpressure behavior, and shutdown semantics around the synchronous handoff.
It must preserve the A10.1 authorization boundary and the A10.2 window grid.
