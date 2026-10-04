# Live latency, reliability, and alert-timing measurement

## Scope

A10.3 adds a reproducible, audio-free measurement contract around the authorized
in-process live path built in A10.1 through B10.2. The benchmark opens a genuinely
authenticated local stream, queues synthetic PCM through the bounded resilience
controller, forces one downstream deferral, verifies exact retry, triggers one
full-buffer rejection, emits rolling windows, and records deterministic decision
and local-alert timing probes.

This is a regression measurement, not deployment approval. It uses accelerated
synthetic PCM in one Python process. It does not measure a microphone, device
driver, network transport, production model inference, real incident accuracy,
notification delivery, or external action.

## Measurement boundaries

All durations use `time.perf_counter_ns`, so system-clock adjustments cannot make
an elapsed time negative. The saved report records milliseconds to six decimal
places and summarizes each metric with count, minimum, median, mean, p95, and
maximum. Median and p95 use linear interpolation over the sorted observations.

| Metric | Start | End |
| --- | --- | --- |
| `connection_open` | Immediately before the buffered client connects | Fresh device challenge, HMAC authentication, authorization, and local session open complete |
| `buffer_enqueue` | Immediately before a synthetic PCM chunk enters B10.2 | The owned in-memory chunk and audio-free ticket exist |
| `capture_to_acceptance` | Immediately before the successful capture/enqueue attempt | The flush call returns after verified FIFO delivery; a retried chunk includes its deferred interval |
| `capture_to_window` | Start of the chunk that makes the window available | The rolling-window consumer receives the complete window |
| `capture_to_decision` | Same triggering-chunk start | The deterministic decision timing probe completes |
| `capture_to_local_alert_probe` | Same triggering-chunk start | The local alert timing marker is created for the one designated alert candidate |

The alert probe measures orchestration timing only. It deliberately does not claim
that a model detected an incident or that a real `LocalAlertDocument` was created.
The report permanently states `alert_effectiveness_measured=false`,
`notification_delivery=not_sent`, and `alert_delivery_authorized=false`.

## Reliability accounting

Every denominator is explicit:

- chunk delivery rate = accepted chunks / enqueued chunks;
- delivery-attempt success rate = accepted chunks / all delivery attempts,
  including the forced deferral;
- window processing rate = processed windows / emitted windows;
- alert-probe completion rate = completed alert probes / designated candidates;
- loss rate = explicitly discarded chunks / enqueued chunks.

Enqueued chunks must reconcile exactly to accepted, discarded, and still-queued
chunks. Delivery attempts must reconcile to accepted, deferred, and failed
attempts. Emitted windows must reconcile to processed windows and processing
failures. Duplicate acceptances and sequence gaps remain separate counters rather
than disappearing inside a percentage.

## Recorded benchmark

The checked-in run used CPython 3.13.5 on Windows/AMD64. It processed twelve
10 ms mono PCM chunks at 16 kHz through a four-chunk reject-mode buffer and a
40 ms rolling window with 50% overlap. The run lasted 41.7179 ms in accelerated
in-process execution.

| Latency | Samples | Median (ms) | p95 (ms) | Maximum (ms) |
| --- | ---: | ---: | ---: | ---: |
| Connection open | 1 | 4.394300 | 4.394300 | 4.394300 |
| Buffer enqueue | 12 | 0.013550 | 0.049610 | 0.074800 |
| Capture to acceptance | 12 | 5.004600 | 6.343420 | 6.389400 |
| Capture to window | 5 | 4.886900 | 5.920120 | 6.172900 |
| Capture to decision probe | 5 | 4.905500 | 5.952780 | 6.208200 |
| Capture to local-alert probe | 1 | 6.221300 | 6.221300 | 6.221300 |

The reliability result was:

- 12 of 12 enqueued chunks accepted;
- 13 delivery attempts, including one forced deferred attempt;
- one expected backpressure rejection outside the enqueue denominator;
- zero discarded or unresolved chunks;
- zero duplicate acceptances and zero sequence gaps;
- five of five emitted windows processed;
- one of one deterministic alert timing probes completed; and
- zero raw-audio persistence and zero notification delivery.

The resulting chunk delivery, window processing, and alert-probe completion rates
are 100%. Delivery-attempt success is 92.3077% because the forced deferral remains
visible in its denominator. Observed loss is 0%.

These values describe one small local regression run. A p95 calculated from one or
five samples is only a mechanically complete summary, not a statistically stable
latency estimate. A deployment review must collect longer runs on intended
hardware, with real capture scheduling and the actual approved model path.

## Report integrity and privacy

`LivePerformanceReport` validates its canonical SHA-256 identity, recomputes every
latency distribution and rate from the underlying observations and counters, and
rejects mismatched inventories. The report contains no PCM, payload hashes,
credentials, proofs, consent IDs, device IDs, session IDs, transcripts, local
paths, recipients, or transport settings. Writes are bounded and atomic;
replacement requires the explicit `--replace` option.

The portable contract is checked in at
`docs/schemas/v1/live-performance-report.schema.json`. The measured report is
`outputs/a10_3_measurement/live-performance-report.json`.

## Reproduce locally

Run the measurement:

```powershell
python scripts\measure_live_performance.py --replace
```

Run the focused checks:

```powershell
python -m pytest tests\test_live_performance.py -q
```

The exact timings and report identity will change between machines and runs. The
scenario, metric definitions, safety flags, and reliability accounting remain the
same.

## Final review outcome

A10.4 used these limits in the
[final privacy, security, and deployment review](final-privacy-security-deployment-review.md).
The accelerated benchmark remains conditional regression evidence and does not
support production approval. The final decision is `not_approved_for_production`.
