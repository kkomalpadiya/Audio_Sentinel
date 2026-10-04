# Phase 10 completion report: Live Device Integration and Final Review

## 1. Executive summary

Phase 10 turned the earlier offline Audio Sentinel pipeline into a carefully
bounded **local live-streaming path**. It did not turn the project into a finished
production monitoring product. The phase added the rules, services, rolling audio
windows, buffering, retry behavior, measurements, and final review needed to show
exactly what the local implementation can and cannot do.

In common terms, Phase 10 built the following chain:

1. An approved device is enrolled for a limited time, processing scope, and audio
   format.
2. The device proves that it knows its secret without sending the secret itself.
3. A short-lived local streaming session is opened only when enrollment and
   consent are still valid.
4. Every piece of PCM audio must arrive in order, with no missing or repeated
   sample range.
5. Accepted chunks are converted into the same overlapping analysis-window grid
   used by the offline pipeline.
6. A bounded memory queue handles temporary downstream delay without silently
   losing, duplicating, or moving audio into a new authorization session.
7. A deterministic in-process benchmark measures timing and reconciles every
   accepted, deferred, rejected, discarded, and still-queued item.
8. A final evidence-backed review records the implemented controls and the work
   that still blocks production deployment.

The final Phase 10 decision is:

> **Not approved for production.** Permitted use remains controlled local research
> and offline evaluation with authorized recordings and human review.

That decision is not a failure of the phase. It is one of its main outputs. The
project now distinguishes a tested local software path from the additional legal,
security, hardware, operational, and field-validation work required for a real
deployment.

## 2. Phase 10 goals and boundaries

### 2.1 What Phase 10 set out to solve

The Phase 1–9 pipeline already knew how to evaluate recorded audio, create
privacy-minimized reports, record local alert candidates, evaluate a labeled
collection, and apply limited retention controls. It did not define how live audio
could enter the system safely.

Live audio introduces new problems that a file-based pipeline does not have:

- a device must be recognized and authorized;
- permission can expire or be withdrawn while audio is flowing;
- audio arrives in small pieces rather than one complete file;
- missing, duplicated, or reordered pieces can corrupt time alignment;
- the analysis path can be slower than capture;
- reconnecting can accidentally mix old and new authorization;
- retry can duplicate windows or decisions;
- unbounded queues can exhaust memory;
- performance numbers can be misleading when their denominators are hidden; and
- a technically functioning stream can still be legally, securely, or
  operationally unsuitable for production.

Phase 10 addressed these problems through six tasks.

### 2.2 What Phase 10 deliberately did not add

Phase 10 did not add:

- microphone or device-driver capture;
- a Bluetooth, WebSocket, TCP, gRPC, mobile, or internet transport;
- durable storage of queued live PCM;
- a background operating-system service;
- production live model inference orchestration;
- reviewer accounts, assignment, or disposition handling;
- SMS, email, push, emergency-service, or other notification delivery;
- automatic external action;
- production credential storage or key rotation;
- representative field validation; or
- production monitoring, incident response, disaster recovery, or signed release
  operations.

These exclusions are important because they prevent an in-process demonstration
from being described as a complete live safety product.

## 3. Phase 10 task summary

| Task | Main result | Plain-language purpose |
| --- | --- | --- |
| A10.1 | Authorized-device streaming contracts | Defines who may stream, under which consent, in which format, and how chunks must line up. |
| B10.1 | Local streaming ingestion service | Implements enrollment, revocation, one-time authentication, session management, and synchronous audio handoff. |
| A10.2 | Rolling live-window adapter | Converts arbitrary PCM chunks into the exact overlapping windows expected by analysis. |
| B10.2 | Bounded buffering and reconnect behavior | Handles slow consumers, retry, overload, disconnect, discard, and fresh-session reconnect without silent loss. |
| A10.3 | Live-path performance measurement | Measures latency and reconciles reliability counters through a deterministic local benchmark. |
| A10.4 | Final privacy, security, and deployment review | Records verified controls, conditional controls, release blockers, and the non-production decision. |

The six tasks added 124 focused tests. At Phase 10 completion, the complete project
suite contained 1,989 passing tests, plus the generated-audio preparation smoke
test. The only reported warnings were two existing Pydantic deprecation warnings
from the API OpenAPI test.

## 4. Complete Phase 10 flow in common terms

The complete local flow can be pictured as:

```text
approved enrollment
        |
        v
one-time challenge -> HMAC proof -> single-use authentication receipt
        |
        v
consent- and format-bound stream session
        |
        v
bounded FIFO queue -> ordered chunk validation -> synchronous sink handoff
        |
        v
PCM decode/downmix -> rolling overlapping windows -> local analysis consumer
        |
        v
audio-free close accounting, measurements, and review artifacts
```

At every arrow, the system checks a different question:

- **Enrollment:** Is this exact device currently approved?
- **Authentication:** Can it prove possession of the enrolled credential?
- **Consent:** Is the requested processing still permitted now?
- **Format:** Is the audio geometry one of the approved formats?
- **Order:** Is this the exact next sequence number and sample position?
- **Capacity:** Can the bounded queue and rolling buffer safely hold the data?
- **Delivery:** Did the downstream consumer accept the exact item?
- **Reconnect:** Has old-session audio been delivered or explicitly discarded?
- **Measurement:** Do all attempts and outcomes reconcile?
- **Review:** Does the evidence support production use? In Phase 10, it does not.

## 5. A10.1 — Authorized-device enrollment and streaming contracts

### 5.1 Problem being solved

Before A10.1, “live input” was only an idea. There was no exact contract describing
which device could open a stream, how it proved its identity, which consent scope
applied, what PCM format was allowed, how chunks were ordered, or what a successful
close meant.

Implementing a transport first would have embedded these policy decisions inside
network code. A10.1 therefore defined the rules before implementing the service.

### 5.2 Enrollment contract

`DeviceEnrollment` records:

- an opaque enrollment ID and device ID;
- active or revoked status;
- enrollment and expiry times;
- the maximum approved processing scope;
- one or more exact PCM formats;
- the SHA-256 fingerprint of the device credential; and
- revocation time and reference when revoked.

Timestamps must include a timezone. An enrollment cannot be used before it starts,
after it expires, or after revocation. A requested processing scope cannot be
broader than the enrolled maximum.

The credential itself is not stored in the enrollment. Only its fingerprint is
recorded. This limits what is exposed if the enrollment document is read.

### 5.3 Session authorization algorithm

The authorization algorithm is a fail-closed sequence:

1. Validate the enrollment, session request, authentication receipt, and consent
   record through strict Pydantic contracts.
2. Require the enrollment to be active at the stream-open time.
3. Require the authentication receipt to be unexpired and bound to the same
   enrollment, device, credential fingerprint, and challenge.
4. Require consent to be granted, active, device-authorized, and broad enough for
   the requested processing scope.
5. Require the requested scope not to exceed the enrollment scope.
6. Require the requested PCM format to match an enrolled format exactly.
7. Derive an authorized session containing the verified enrollment digest,
   consent snapshot, scope, format, start time, and expiry boundary.

The session expiry cannot outlive the authorization sources that created it.

### 5.4 Chunk continuity algorithm

Every chunk carries:

- a session ID;
- sequence number;
- starting sample position;
- sample count;
- capture time; and
- PCM bytes.

For each accepted chunk, the service expects two counters:

```text
expected sequence number = previous accepted sequence + 1
expected start sample    = previous accepted end sample
```

The first chunk starts at sequence `0` and sample `0`. A chunk is rejected when
its sequence is missing, repeated, or out of order, or when its starting sample
does not exactly match the previously accepted end sample.

The byte-length check is:

```text
expected bytes = sample count × channels × bytes per sample
```

The format currently uses signed 16-bit little-endian PCM, so bytes per sample is
two. The declared sample count and payload length must agree exactly.

### 5.5 Close algorithm

A close request includes the next expected sequence and sample positions. Closure
is accepted only when those positions match the service's accepted state. This
prevents a caller from closing a stream while pretending that missing audio had
already been processed.

The close receipt contains audio-free final positions and a reason. It does not
create alert or notification authority.

### 5.6 Why strict contracts were chosen

Strict immutable contracts were chosen over dictionaries or transport-specific
messages because they:

- reject unexpected fields and ambiguous coercions;
- make every boundary testable without a network;
- generate portable JSON Schemas;
- preserve timezone and resource-limit rules;
- keep later transport choices open; and
- prevent a network framework from silently becoming the policy authority.

The main alternative was to start with a WebSocket or mobile protocol and validate
messages inside its handlers. That was rejected because authentication, consent,
ordering, and close semantics would become difficult to reuse and audit.

## 6. B10.1 — Local streaming ingestion service

### 6.1 Enrollment registry algorithm

`FileDeviceEnrollmentRegistry` stores enrollment records below the managed
processed-data root. It uses opaque validated identifiers and exact directory
inventories to prevent path traversal or unexpected files.

Registration follows this pattern:

1. Validate the complete enrollment.
2. Resolve the managed parent and the exact enrollment directory.
3. Serialize canonical validated JSON.
4. Write through a private staging location using exclusive creation.
5. Flush and synchronize the file.
6. Publish the record without overwriting an unrelated existing record.
7. Reload it through the public contract.

An exact repeat can be reused safely. A conflicting record is rejected rather
than overwritten.

Revocation is append-only at the document level. The original active enrollment
remains intact, while a separate `revoked.json` records the status, time, and
reference. On load, the registry verifies that the revocation has the same base
enrollment fields and changes only the allowed revocation fields.

#### Why append-only revocation was selected

Overwriting the original enrollment would destroy the evidence of what was
initially approved. A database event log could also preserve history, but it would
introduce a database dependency and migration surface before the local contract
was stable. Two small verified documents provide a simple audit trail suitable for
the current local prototype.

### 6.2 One-time HMAC-SHA256 authentication

The local authenticator uses challenge-response authentication:

1. Load the active enrollment.
2. Generate unpredictable challenge bytes with `secrets.token_bytes`.
3. Hash the challenge and bind it to the enrollment and device.
4. Give the challenge to the local client with a short expiry.
5. The client calculates:

   ```text
   proof = HMAC-SHA256(device secret, challenge bytes)
   ```

6. The server resolves the expected secret through an external provider, verifies
   its SHA-256 fingerprint against the enrollment, independently calculates the
   HMAC, and compares values using `hmac.compare_digest`.
7. The challenge is removed on the first authentication attempt.
8. A short-lived authentication receipt is created.
9. That receipt is claimed once when the stream opens and is then removed.

The resolved secret is copied into a mutable byte array and overwritten after the
calculation. Secrets, proofs, and challenges are not persisted to the registry.

#### Why HMAC was selected

HMAC-SHA256 was chosen because it is available in Python's standard library,
well understood, efficient, and appropriate for proving possession of a shared
secret in this local prototype. It avoids transmitting or storing the clear secret
inside session records.

Alternatives were considered:

- **Sending the secret directly:** rejected because capture, logs, or debugging
  could expose it and replay would be easy.
- **A reusable bearer token:** rejected because theft would permit replay until
  expiry.
- **Password hashing:** useful for stored human passwords, but not the right
  challenge-response primitive for device possession.
- **Public-key signatures or mutual TLS:** stronger choices for a real distributed
  deployment, but they require certificate issuance, trust stores, rotation, and
  transport integration that Phase 10 intentionally did not claim to provide.

HMAC is therefore a local implementation choice, not a conclusion that shared
secrets are sufficient for production.

### 6.3 Synchronous ingestion algorithm

`LocalStreamingIngestionService` keeps bounded active-session state under a
reentrant lock. For each chunk it:

1. verifies that the supplied session exactly matches active state;
2. reloads the enrollment;
3. rejects and removes the session if enrollment is missing, revoked, expired, or
   different from the hash bound into the session;
4. validates sequence, sample position, timestamp, format, sample count, and
   payload length;
5. calls the required downstream sink synchronously;
6. advances sequence and sample counters only after the sink succeeds; and
7. returns an audio-free acceptance receipt.

If the sink fails, the expected position is not advanced. The same exact chunk can
be retried. This creates an **at-least-once attempt with exactly-once state
advancement** at this local boundary.

### 6.4 Why synchronous handoff was chosen

Synchronous handoff gives one unambiguous answer: either the sink accepted the
chunk and the stream position advanced, or it did not. This made the ordering and
retry contract testable before introducing queues or workers.

An asynchronous fire-and-forget design was rejected because it would need durable
acknowledgements, queue ownership, crash recovery, and a definition of when a
chunk counts as accepted. B10.2 later adds a bounded capture-side queue without
weakening the B10.1 acknowledgement boundary.

### 6.5 Local client

`LocalStreamingClient` performs the expected caller sequence:

- request challenge;
- calculate HMAC proof;
- construct the session request;
- open the session;
- derive sample counts from PCM geometry;
- send the next sequence/sample position; and
- close at the final accepted position.

It accepts already captured PCM bytes. It does not open a microphone or transport.

## 7. A10.2 — Rolling live windows

### 7.1 Why rolling adaptation was needed

Models and feature stages analyze fixed-length windows, but device chunks can be
any approved duration and rarely align with window boundaries. An adapter must
join chunks while preserving the same sample geometry used offline.

### 7.2 Window-grid algorithm

For each configured window duration:

```text
window samples = round(window seconds × target sample rate)
hop samples    = round(window samples × (1 - overlap ratio))
```

For a 40 ms window at 16 kHz with 50% overlap:

```text
window samples = round(0.040 × 16000) = 640
hop samples    = round(640 × 0.5)     = 320
```

This is deliberately the same formula used by the offline `AudioSettings` path.
Matching the sample grid prevents live and offline evaluation from describing the
same recording with different window boundaries.

### 7.3 Chunk adaptation algorithm

For each verified chunk, the adapter:

1. validates the session, chunk, and acceptance-receipt relationship;
2. requires the stream sample rate to match the analysis sample rate;
3. interprets signed 16-bit little-endian PCM;
4. converts samples to finite `float32` values in the `[-1, 1]` range;
5. downmixes multiple authorized channels to mono by deterministic arithmetic
   mean;
6. appends the decoded samples to the session's bounded memory buffer;
7. emits every complete configured window whose sample range is now available;
8. advances the cursor for each window duration by its hop size; and
9. releases prefix samples once no duration can need them again.

Multiple window durations are emitted in earliest-readiness order. Stable group
and window indices remove dependence on dictionary iteration or chunk boundaries.

Each emitted window owns a contiguous, read-only NumPy array. Its audio-free
record includes absolute sample span, duration group, padding count, triggering
sequence, and SHA-256 of canonical little-endian float32 samples.

### 7.4 Why deterministic mean downmix was chosen

Arithmetic mean is transparent, deterministic, inexpensive, and consistent across
runs. Selecting only the left or first channel could discard useful sound.
Taking the maximum channel could inflate amplitude and distort model inputs.
Energy-preserving or learned spatial mixing can be useful for microphone arrays,
but requires calibration and device-specific assumptions absent from the current
contract.

### 7.5 Tail handling

The normal streaming path emits only complete windows. At verified close, the
configured offline-compatible policy applies:

- `pad`: emit a final partial window with zero padding; or
- `drop`: record the uncovered tail without fabricating a window.

An empty stream does not create artificial silence. Close produces audio-free
per-duration counts for emitted windows, padded windows, and uncovered samples.

### 7.6 Retry checkpoint algorithm

One input chunk may make several windows ready. A downstream consumer can accept
the first windows and fail on a later one. Rebuilding the whole handoff would
duplicate the windows already accepted.

The adapter therefore advances each duration cursor immediately after that window
is accepted. If a later delivery fails, it stores the pending handoff identity and
resumes at the first undelivered item when the exact same chunk is retried. A
different chunk or close request cannot replace pending work.

This checkpoint design was selected over replaying the entire chunk because model
or alert stages may not be idempotent. It was selected over rolling back accepted
windows because the downstream side may already have committed effects that the
adapter cannot undo.

### 7.7 Why a bounded rolling buffer was selected

An ever-growing array would be easy to implement but would make a long stream a
memory-exhaustion risk. Reprocessing every chunk independently would miss windows
that cross chunk boundaries. Writing temporary WAV files would create unnecessary
raw-audio retention.

The selected rolling buffer retains only samples that a future window can still
need, enforces a byte limit, and remains process-memory only.

## 8. B10.2 — Buffering, backpressure, and reconnect

### 8.1 Bounded FIFO algorithm

`BufferedStreamingClient` uses a `deque` as a first-in, first-out PCM queue. Each
entry owns a mutable byte copy and an audio-free ticket containing sample count,
size, capture/enqueue times, enqueue index, and payload SHA-256.

Enqueue is allowed only when:

- the controller is connected;
- authorization has not expired;
- the PCM is non-empty and aligned to the approved frame geometry;
- capture time is timezone-aware and nondecreasing; and
- both chunk-count and byte-count limits remain satisfied.

The queue is bounded in two dimensions because many tiny chunks can exhaust object
overhead while a few large chunks can exhaust bytes.

### 8.2 Backpressure algorithms

Two explicit policies are supported.

#### Reject mode

When full, enqueue fails immediately with a stable `backpressure` result. Capture
code can stop, reduce rate, or record an external loss decision. The queue never
pretends it accepted data it could not store.

#### Block mode

The producer waits on a condition variable until flush or explicit discard frees
capacity, authorization changes, or the configured timeout expires. The timeout is
bounded to avoid an indefinitely stuck capture thread.

#### Why both modes exist

Interactive or real-time capture may prefer immediate rejection so the caller
keeps control of timing. A controlled producer may prefer bounded waiting to absorb
a short downstream stall. An unbounded blocking mode was rejected because it can
deadlock shutdown or hide a failed consumer.

### 8.3 Ordered flush algorithm

Flush always sends the oldest entry first:

1. mark one flush as active;
2. read the queue head without removing it;
3. send it through the authenticated B10.1 client;
4. verify that the receipt matches session, sample count, and payload hash;
5. only then remove the queue head;
6. overwrite the mutable PCM copy;
7. update counters; and
8. notify waiting producers that capacity is available.

Only one flush may run at a time. Producers may still enqueue within the remaining
limits.

If the sink returns `sink_failed`, the result becomes `delivery_deferred`. The
exact queue head remains in place for retry at the same stream position.

### 8.4 Why remove-after-receipt was selected

Removing before delivery risks silent data loss. Removing after merely calling the
client risks treating an ambiguous response as success. The selected algorithm
removes only after a matching verified receipt, so queue state and accepted stream
position remain aligned.

An inconsistent receipt is not automatically retried because the service may have
advanced even though the controller cannot prove it. The connection becomes
disconnected, the entry stays queued, and a human or higher-level policy must
resolve it explicitly.

### 8.5 Reconnect algorithm

Reconnect is a security boundary, not just a socket retry:

1. require the old queue to be empty;
2. close the old session if it still exists;
3. create a fresh client through the supplied factory;
4. perform a new challenge and authentication;
5. apply current consent; and
6. open a new session at sequence `0`, sample `0`.

Old-session PCM is never copied into the new session. Capture time, consent, and
authorization all belong to the session under which data was obtained.

If the old session disappears while data remains queued, the controller retains
the data and requires either restored delivery or explicit discard before
reconnect. This avoids silently relabeling old audio as newly authorized.

### 8.6 Explicit discard algorithm

Discard requires a typed reason such as operator request, authorization expiry,
connection loss, or service shutdown. Every mutable PCM copy is overwritten, and
the returned receipt lists only ticket IDs, counts, bytes, reason, and time.

Discard is visible loss accounting. It does not claim that the samples were
analyzed or securely erased from every possible memory copy.

### 8.7 Why memory-only buffering was chosen

Durable retry could survive process crashes, but it would write raw live audio and
would require encryption, key management, retention, legal hold, secure deletion,
and recovery semantics. Those controls do not yet exist. A bounded in-memory queue
was therefore the safer scope for Phase 10.

The tradeoff is explicit: a process crash loses queued audio. The project records
this as an availability limitation rather than hiding it.

## 9. A10.3 — Latency, reliability, and alert timing

### 9.1 Measurement design

The benchmark runs through the real A10.1–B10.2 local implementation using
synthetic PCM. It performs:

- fresh HMAC authentication and session opening;
- bounded queue enqueue and flush;
- one forced downstream deferral followed by exact retry;
- one intentional full-buffer rejection;
- rolling-window emission;
- deterministic decision timing; and
- one local-alert timing probe with no notification delivery.

### 9.2 Clock algorithm

Durations use `time.perf_counter_ns`:

```text
elapsed milliseconds = (end nanoseconds - start nanoseconds) / 1,000,000
```

This monotonic high-resolution clock was chosen over `datetime.now()` because
wall-clock adjustments can jump backward or forward. UTC datetimes remain useful
for document timestamps, but monotonic time is the correct tool for elapsed
duration inside one process.

### 9.3 Percentile algorithm

For sorted observations and quantile `q`:

```text
position = (number of values - 1) × q
```

When the position falls between two indexes, the report uses linear interpolation.
The report saves count, minimum, median, mean, p95, and maximum for each metric.

Linear interpolation was selected because it is deterministic and behaves
sensibly for the small fixed benchmark. A production service should use a much
larger sample set and may use a streaming histogram, HDR histogram, or telemetry
backend. Those tools were not justified for a tiny checked-in regression receipt.

### 9.4 Reliability accounting algorithm

Every denominator is explicit:

```text
chunk delivery rate = accepted chunks / enqueued chunks
attempt success rate = accepted chunks / all delivery attempts
window rate         = processed windows / emitted windows
alert-probe rate    = completed probes / designated candidates
loss rate           = discarded chunks / enqueued chunks
```

The counters must reconcile:

```text
delivery attempts = accepted + deferred + failed
enqueued chunks   = accepted + discarded + still queued
emitted windows   = processed + processing failures
```

Backpressure rejections are outside the enqueue denominator because the queue
explicitly did not accept them. Deferred attempts remain inside the attempt
denominator. Duplicate acceptances and sequence gaps have separate counters.

This approach was chosen over a single “success percentage” because one number can
hide retries, rejected input, unresolved data, or a denominator of zero.

### 9.5 Recorded results

The checked-in run used CPython 3.13.5 on Windows/AMD64. It processed twelve 10 ms
mono PCM chunks at 16 kHz, a four-chunk reject-mode queue, and a 40 ms window with
50% overlap.

| Metric | Samples | Median | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Connection open | 1 | 4.394300 ms | 4.394300 ms | 4.394300 ms |
| Buffer enqueue | 12 | 0.013550 ms | 0.049610 ms | 0.074800 ms |
| Capture to acceptance | 12 | 5.004600 ms | 6.343420 ms | 6.389400 ms |
| Capture to window | 5 | 4.886900 ms | 5.920120 ms | 6.172900 ms |
| Capture to decision probe | 5 | 4.905500 ms | 5.952780 ms | 6.208200 ms |
| Capture to local-alert probe | 1 | 6.221300 ms | 6.221300 ms | 6.221300 ms |

Reliability accounting recorded:

- 12 of 12 enqueued chunks accepted;
- 13 attempts because one delivery was deliberately deferred;
- one expected backpressure rejection outside the enqueue denominator;
- zero discarded or unresolved chunks;
- zero duplicate acceptances and zero sequence gaps;
- five of five emitted windows processed; and
- one of one local-alert timing probes completed.

Chunk delivery, window processing, and alert-probe completion were 100%. Attempt
success was 92.3077% because the forced deferral remained visible. Observed loss
was 0%.

### 9.6 What the measurements do not mean

The run was accelerated and in-process. It did not measure:

- real capture scheduling;
- device-driver delay;
- network delay or loss;
- production model inference;
- concurrent devices;
- long-duration memory behavior;
- operating-system service recovery;
- detection correctness;
- alert effectiveness; or
- notification delivery.

A p95 from one or five samples is mechanically calculable but not statistically
stable. The report intentionally labels itself regression evidence rather than
deployment approval.

## 10. A10.4 — Final privacy, security, and deployment review

### 10.1 Review algorithm

The final review uses a fixed inventory rather than letting the caller choose only
favorable checks:

1. Resolve 16 defined evidence files under the repository root.
2. Reject missing, empty, oversized, or escaping paths.
3. Record normalized relative path, purpose, byte count, and SHA-256 for each file.
4. Create fixed findings across privacy, security, and deployment.
5. Require every finding to reference declared evidence.
6. Require every release blocker to be high or critical and to have a concrete
   required action.
7. Recompute domain and disposition counts.
8. Canonicalize the complete document and derive its review ID with SHA-256.
9. Save it using bounded atomic persistence.

`verify_review_evidence` later rehashes the files and reports which evidence has
changed since the review.

### 10.2 Why a fixed evidence-backed review was chosen

A free-form checklist is easy to edit after the fact, omit uncomfortable items
from, or detach from the exact code and test state. The selected structure makes
the review portable, reproducible, and tamper-evident at the content level.

A database governance tool or signed attestation system would be more appropriate
for a production organization, but it would require identities, signing keys,
approval roles, and external infrastructure. Phase 10 records that absence rather
than simulating it.

### 10.3 Review results

The review contains 17 findings:

| Result | Count |
| --- | ---: |
| Verified controls | 5 |
| Conditional controls | 2 |
| Release blockers | 10 |

Verified controls include consent/scope enforcement, bounded memory-only live PCM,
privacy-minimized final outputs, loopback-only API exposure, and authenticated
revocable device sessions.

Conditional controls include hash/atomic-write integrity and the A10.3 local
performance evidence. Both are useful, but neither proves production safety.

The release blockers cover:

- incomplete end-to-end data lifecycle enforcement;
- missing legal, privacy, ethics, and organizational approval;
- missing production credential provisioning, storage, rotation, and recovery;
- missing host identity, ACL, encryption, isolation, and hardening controls;
- no independent threat model, penetration test, SBOM, vulnerability gate, or
  signed release evidence;
- no production microphone, transport, supervised service, or live model path;
- no representative hardware, concurrency, soak, recovery, or tail-latency data;
- no authenticated reviewer, disposition, escalation, or notification workflow;
- no production monitoring, incident response, recovery, rollback, and support
  package; and
- insufficient representative detection, fairness, calibration, and human-factors
  validation.

### 10.4 Final decision

The review permanently records:

```text
production_approved = false
notification_delivery_authorized = false
external_action_authorized = false
```

Passing tests cannot automatically turn these fields true. New implementation,
representative evidence, independent review, and organizational approval are
required.

## 11. Algorithms selected and alternatives rejected

| Problem | Selected algorithm | Main rejected alternatives | Why selected |
| --- | --- | --- | --- |
| Device proof | One-time HMAC-SHA256 challenge | Plain secret, reusable bearer token, premature PKI | Prevents clear-secret transmission and basic replay using standard-library primitives. |
| Enrollment history | Immutable base plus append-only revocation document | Overwrite original, add database immediately | Preserves what was approved with low local complexity. |
| Stream order | Exact sequence and absolute sample counters | Arrival order only, timestamps only | Detects duplicates, gaps, and reordering without clock assumptions. |
| Handoff | Synchronous receipt before state advance | Fire-and-forget async | Provides an unambiguous acceptance boundary and exact retry point. |
| Windowing | Rolling sample buffer with offline-equivalent grid | Per-chunk windows, ever-growing buffer, temporary files | Preserves cross-chunk windows, bounds memory, and avoids raw-audio persistence. |
| Multichannel audio | Deterministic arithmetic mean | First channel, maximum, learned spatial mix | Transparent, reproducible, and device-neutral for the current scope. |
| Partial close | Explicit pad or drop policy | Always pad, silently ignore tail | Matches offline policy and keeps tail treatment visible. |
| Downstream retry | Per-window checkpoints | Replay whole chunk, rollback accepted work | Prevents duplicate downstream work and respects already accepted effects. |
| Capture buffering | Bounded FIFO by count and bytes | Unbounded queue, disk spool | Preserves order and limits memory without introducing raw-audio retention. |
| Overload | Explicit reject or bounded block | Silent drop, unlimited block | Keeps loss/flow control visible and prevents deadlock. |
| Reconnect | Fresh authorization at zero with empty old queue | Carry queue into new session | Prevents audio from crossing consent and authentication boundaries. |
| Elapsed time | `perf_counter_ns` | Wall clock | Monotonic and high resolution. |
| Tail summary | Linear-interpolated percentiles plus raw counts | One average, opaque telemetry | Deterministic and inspectable for the small benchmark. |
| Reliability | Reconciled counters with explicit denominators | Single success percentage | Cannot hide deferrals, rejects, losses, or unresolved work. |
| Document identity | Canonical JSON plus SHA-256 | Filename identity, random UUID only | Changes when semantic content changes and is easy to verify offline. |
| Persistence | Stage, flush, synchronize, publish, reload | Direct overwrite | Avoids partial documents and detects invalid saved output. |
| Final review | Fixed hashed evidence inventory | Free-form checklist | Reduces selective omission and detects evidence drift. |

## 12. Technology stack and why it was chosen

### 12.1 Python 3.13

Phase 10 stayed in Python so the live path could reuse the existing contracts,
consent rules, audio settings, tests, and local pipeline. Python also provides the
needed cryptographic building blocks, filesystem operations, threading primitives,
high-resolution clocks, and numerical integration without a new service language.

The tradeoff is that Python is not automatically suitable for hard real-time audio.
Phase 10 measures a local in-process path, not a deterministic real-time operating
system. A production capture adapter may require platform-native components, but
that decision should follow representative measurement.

### 12.2 Pydantic v2 and JSON Schema

Pydantic supplies strict immutable models, field limits, cross-field validation,
timezone checks, enum values, and JSON serialization. Generated JSON Schemas make
the contracts portable to future clients.

Plain dictionaries were rejected because they permit missing or unexpected fields
to travel farther into the system. Handwritten validators would duplicate common
logic and make schema parity harder to prove.

### 12.3 Python standard-library security tools

Phase 10 uses:

- `secrets` for unpredictable challenge and identity material;
- `hmac` with SHA-256 for possession proof;
- `hmac.compare_digest` for constant-time-style equality checks;
- `hashlib.sha256` for fingerprints, payload identity, evidence identity, and
  canonical document identity;
- `pathlib` for explicit path handling;
- `os.fsync`, exclusive creation, hard-link publication, and `os.replace` for
  persistence; and
- UTC-aware `datetime` for authorization boundaries.

These standard tools avoid adding a cryptographic package merely to implement the
local HMAC boundary. They do not replace a production key-management system.

### 12.4 Threading primitives

`RLock`, `Lock`, and `Condition` protect session state, challenge inventories,
queue state, flush exclusivity, and bounded producer waiting.

Threads were chosen because the service is local and synchronous, and the main
problem is protecting small shared state structures. An async framework would not
remove the need for ordering and locking, and it would complicate integration with
synchronous model code. Multiple processes or distributed workers remain outside
the current consistency boundary.

### 12.5 `deque` for FIFO buffering

`collections.deque` provides efficient append and removal at opposite ends. A
Python list would require shifting remaining items when removing the oldest chunk.
A broker such as Kafka, RabbitMQ, or Redis would add network, durability, identity,
operations, and retention questions that are much larger than the Phase 10 local
queue.

### 12.6 NumPy for PCM and windows

NumPy performs exact-width PCM decoding, vectorized float conversion, deterministic
downmixing, slicing, padding, finite/range validation, and read-only window arrays.
It was already part of the audio pipeline and is more reliable and efficient than
sample-by-sample Python loops.

### 12.7 FastAPI boundary

The existing FastAPI application remains a loopback-only offline evaluation API.
Phase 10 does **not** expose the new live ingestion service through FastAPI. This
separation was deliberate: a convenient web framework does not by itself provide
device identity, transport security, production authorization, or safe internet
exposure.

### 12.8 Pytest

Pytest supports fixtures, parameterized boundary matrices, temporary files,
subprocess import checks, concurrency cases, schema parity, tamper tests, and
integration through the real local layers.

Mock-heavy tests alone were avoided where behavior crossed components. The live
integration tests use the real registry, authenticator, service, queue, and rolling
window adapter, while substituting only the external capture/model boundaries.

### 12.9 JSON artifacts and Markdown documentation

JSON provides portable machine-readable receipts. Markdown explains the same
behavior to humans and keeps decisions reviewable in Git. A binary custom format
would make inspection and diffs harder. The XLSX task tracker is project management
evidence, not a runtime dependency.

## 13. Fixes and hardening introduced in Phase 10

This section uses “fix” in the engineering sense: each change closes a concrete
failure mode in a naive live-stream design. Not every item was a user-visible bug
found after release; many were prevented before the live path could be misused.

### 13.1 Replay and stale-authentication fixes

**Problem:** A captured challenge response or authentication receipt could be
reused to open another session.

**Fix:** Challenges are consumed on the first authentication attempt, receipts are
short-lived, and each receipt can be claimed once. Challenge, enrollment, device,
credential fingerprint, and session request are cross-checked.

### 13.2 Enrollment-change and midstream-revocation fixes

**Problem:** A device could authenticate while approved and continue indefinitely
after revocation or record replacement.

**Fix:** Enrollment is reloaded before open and before every chunk. The session
contains the enrollment digest. Missing, revoked, expired, or changed enrollment
terminates acceptance.

### 13.3 Missing, duplicate, and reordered audio fixes

**Problem:** Arrival order alone cannot reveal a missing chunk, duplicate retry, or
sample gap.

**Fix:** Every chunk must match both the next sequence number and next absolute
sample. State advances only after verified sink acceptance. Concurrent duplicates
therefore produce one acceptance and one deterministic rejection.

### 13.4 Partial sink-failure fix

**Problem:** Advancing stream state before downstream success silently loses audio;
retrying after partial success can duplicate work.

**Fix:** B10.1 advances only after sink success. A10.2 advances each window cursor
after that exact window succeeds and checkpoints the first undelivered item.

### 13.5 Chunk-boundary window fix

**Problem:** Creating windows independently inside each device chunk misses windows
that begin in one chunk and end in another.

**Fix:** A rolling sample buffer and absolute cursors create windows on the stream
sample grid, independent of chunk boundaries.

### 13.6 Memory-growth fix

**Problem:** Long streams and slow consumers can make buffers grow without limit.

**Fix:** The rolling adapter releases samples no future window needs. The capture
queue enforces both count and byte limits. Active sessions, pending challenges,
and per-handoff window counts are also bounded.

### 13.7 Silent overload fix

**Problem:** A full buffer might drop audio without recording the loss or freeze a
producer forever.

**Fix:** Reject mode returns an explicit backpressure failure. Block mode has a
condition variable, authorization recheck, and bounded timeout. Counters record
backpressure events.

### 13.8 Remove-before-delivery data-loss fix

**Problem:** Dequeuing before downstream acknowledgement loses the only queued copy
when delivery fails.

**Fix:** The FIFO head remains until a matching receipt proves acceptance. Deferred
delivery retries the exact same head.

### 13.9 Authorization-crossing reconnect fix

**Problem:** Automatically carrying old queued PCM into a freshly authenticated
session would relabel data under new consent and identity state.

**Fix:** Reconnect requires an empty queue and starts at sequence/sample zero.
Unresolved old PCM must be delivered under the old session or explicitly zeroed
and discarded with a reason.

### 13.10 Ambiguous-receipt fix

**Problem:** Automatically retrying after an inconsistent receipt can duplicate a
chunk if the service actually advanced.

**Fix:** Receipt mismatch moves the controller to disconnected state and retains
the entry for explicit resolution rather than guessing.

### 13.11 Close-tail and empty-stream fixes

**Problem:** A live stream can end with a partial window, while padding an empty
stream would fabricate evidence.

**Fix:** Verified close applies an explicit pad/drop policy to real remaining
samples. Empty input creates no synthetic-silence window. Audio-free close
accounting records what happened.

### 13.12 Wall-clock timing fix

**Problem:** System time correction can produce invalid elapsed durations.

**Fix:** Performance intervals use monotonic `perf_counter_ns`; UTC timestamps are
used only for document and authorization time.

### 13.13 Hidden-denominator measurement fix

**Problem:** Reporting “100% success” can hide a failed first attempt,
backpressure, discarded data, or unresolved queue items.

**Fix:** Raw observations and reconciled counters remain in the report. Delivery
rate, attempt rate, loss, window rate, and alert-probe rate have separate explicit
denominators.

### 13.14 Tampered-summary fix

**Problem:** A saved report could retain plausible headline values after its raw
observations or counters were changed.

**Fix:** Load-time validation recomputes distributions, rates, inventories, and
canonical identity. Mismatches are rejected.

### 13.15 Path and unexpected-file fixes

**Problem:** User-controlled IDs or paths could escape managed storage, and extra
files could hide inside a trusted bundle.

**Fix:** Identifiers are opaque and pattern-limited, resolved paths must remain
beneath the managed root, document sizes are bounded, and exact inventories are
checked.

### 13.16 Secret and sensitive-output fixes

**Problem:** Credentials, proofs, PCM, transcript text, recipients, or absolute
paths could leak through persistence, logs, or status responses.

**Fix:** Credential providers remain external, temporary mutable secret copies are
overwritten, live PCM remains memory-only, failures use safe codes, and public
receipts/reports are audio-free and privacy-minimized.

### 13.17 Deployment-overclaim fix

**Problem:** A fast synthetic benchmark and large passing test suite could be
mistaken for field readiness.

**Fix:** The measurement artifact permanently states what was not measured. The
final review requires explicit blockers, hashes its evidence, and fixes the release
decision at `not_approved_for_production` for the reviewed build.

## 14. Security and privacy principles

Phase 10 consistently applies these principles:

1. **Least authority:** device authentication authorizes local audio processing,
   not alert delivery or external action.
2. **Current consent:** authorization is checked at open and live boundaries, not
   assumed from an old session start.
3. **Data minimization:** live PCM stays in bounded process memory; public records
   retain only the metadata needed for verification.
4. **Fail closed:** missing, expired, inconsistent, oversized, reordered, or
   tampered input is rejected rather than guessed into a valid state.
5. **Explicit loss:** backpressure, deferred delivery, discard, and unresolved
   data have named outcomes and counters.
6. **No boundary crossing:** old-session audio cannot enter a new authorization.
7. **Integrity with limits:** SHA-256 detects content changes but is not described
   as an author signature or trusted chain of custody.
8. **Human authority retained:** a local alert candidate is not a confirmed event,
   notification, dispatch, or permission to act.

## 15. Testing and verification

### 15.1 A10.1 contract tests

The 25 focused tests cover enrollment geometry, active/revoked state, consent
scope, challenge binding, authentication expiry, sample and sequence continuity,
payload size, close position, immutable contracts, schemas, protocols, safe errors,
and import without network/model runtimes.

### 15.2 B10.1 ingestion tests

The 20 focused tests cover exact registration reuse, conflicting records,
append-only revocation, path/inventory tampering, one-time challenges and receipts,
incorrect HMAC, forged receipts, capacity, full client/service intake, concurrent
duplicates, sink retry, midstream revocation, expiry, absence of audio files, and
dependency-free import.

### 15.3 A10.2 rolling-window tests

The 20 focused tests cover offline-grid parity, irregular chunk boundaries,
padding and dropping tails, multiple durations, stereo downmix, immutable arrays,
sample-rate errors, receipt mismatches, resumable chunk/close callbacks, empty
streams, capacity, memory limits, protocol conformance, and real B10.1 integration.

### 15.4 B10.2 resilience tests

The 23 focused tests cover FIFO delivery, partial flush, reject/block/timeout
backpressure, exact deferred retry, disconnect retention, clean reconnect, refreshed
consent, invalid PCM and time, authorization expiry, ambiguous receipts, explicit
close, concurrent flush exclusion, factory failure, real B10.1/A10.2 integration,
absence of audio files, and network/model-free import.

### 15.5 A10.3 measurement tests

The 22 focused tests cover percentile math, explicit and zero denominators, counter
reconciliation, observation identities, canonical report integrity, tamper
rejection, bounded atomic save/load, replacement rules, the real local benchmark,
schema parity, script output, privacy exclusions, and import isolation.

### 15.6 A10.4 review tests

The 14 focused tests cover the non-production decision, domain and count
reconciliation, mandatory blocker actions, evidence hashing, evidence drift,
portable privacy exclusions, canonical tamper rejection, invalid paths and times,
bounded atomic persistence, schema parity, script output, and network/model-free
import.

### 15.7 Full-suite progression

| Completed task | Full suite |
| --- | ---: |
| A10.1 | 1,890 passed |
| B10.1 | 1,910 passed |
| A10.2 | 1,930 passed |
| B10.2 | 1,953 passed |
| A10.3 | 1,975 passed |
| A10.4 | 1,989 passed |

The Phase 10 completion run also passed the generated 440 Hz preparation smoke
test, confirming that the new live modules did not break the original offline
audio-preparation path.

## 16. Main Phase 10 artifacts

### Contracts and ingestion

- `src/audio_sentinel/live_streaming.py`
- `src/audio_sentinel/streaming_ingestion.py`
- `docs/live-device-streaming-interface.md`
- `docs/streaming-ingestion.md`
- `docs/schemas/v1/device-enrollment.schema.json`
- `docs/schemas/v1/device-authentication-receipt.schema.json`
- `docs/schemas/v1/authorized-stream-session.schema.json`
- the session, chunk, receipt, and close schemas in `docs/schemas/v1/`

### Windows and resilience

- `src/audio_sentinel/live_windows.py`
- `src/audio_sentinel/stream_resilience.py`
- `docs/live-rolling-windows.md`
- `docs/stream-resilience.md`

### Measurement

- `src/audio_sentinel/live_performance.py`
- `scripts/measure_live_performance.py`
- `docs/live-performance-measurement.md`
- `docs/schemas/v1/live-performance-report.schema.json`
- `outputs/a10_3_measurement/live-performance-report.json`

### Final review

- `src/audio_sentinel/deployment_review.py`
- `scripts/run_final_deployment_review.py`
- `docs/final-privacy-security-deployment-review.md`
- `docs/schemas/v1/final-deployment-review.schema.json`
- `outputs/a10_4_review/final-deployment-review.json`

### Tests

- `tests/test_live_streaming.py`
- `tests/test_streaming_ingestion.py`
- `tests/test_live_windows.py`
- `tests/test_stream_resilience.py`
- `tests/test_live_performance.py`
- `tests/test_deployment_review.py`

## 17. Reproduction commands

Run all Phase 10 focused suites:

```powershell
python -m pytest `
  tests\test_live_streaming.py `
  tests\test_streaming_ingestion.py `
  tests\test_live_windows.py `
  tests\test_stream_resilience.py `
  tests\test_live_performance.py `
  tests\test_deployment_review.py -q
```

Regenerate the synthetic live-path measurement:

```powershell
python scripts\measure_live_performance.py --replace
```

Regenerate the final review after an intentional evidence change:

```powershell
python scripts\run_final_deployment_review.py --replace
```

Run complete project verification:

```powershell
.\scripts\verify_project.ps1
```

The exact timing values and content-addressed report IDs change when their inputs
or runtime measurements change. The contracts, safety flags, reconciliation rules,
and non-production boundary remain the intended behavior.

## 18. Manual work required before any production deployment

No manual step is required merely to read this report or run the local contract
tests. Production deployment, however, requires work outside the current codebase:

1. Obtain legal, privacy, ethics, and organizational approval for the exact use.
2. Define the people, places, devices, notices, consent process, and prohibited
   uses.
3. Implement complete inventory, retention, backup, legal-hold, deletion, and
   secure-disposal controls for every data class.
4. Select a hardware capture path and authenticated encrypted transport.
5. Integrate a managed keystore or secret manager with provisioning, rotation,
   recovery, and compromise response.
6. Define operator identity, least-privilege service accounts, filesystem access,
   encryption, process isolation, and endpoint hardening.
7. Implement the approved live model path and measure it on target hardware.
8. Perform representative field, subgroup, calibration, failure, and human-factors
   evaluation with independently reviewed labels.
9. Build authenticated reviewer, disposition, escalation, and notification
   workflows that preserve human authority.
10. Add observability, service objectives, alerting, incident response, recovery,
    rollback, signed releases, dependency gates, SBOM, ownership, and support.
11. Conduct an independent threat model, security assessment, and penetration test.
12. Repeat the final review against the completed evidence; do not manually edit
    the existing non-production decision.

## 19. Final conclusion

Phase 10 completed the planned engineering bridge between offline evaluation and a
local live-streaming research path. It established strong internal invariants:
only an enrolled and freshly authenticated device may open a session; consent and
format remain bounded; chunks cannot silently skip or repeat; rolling windows
match offline sample geometry; retry does not duplicate accepted work; queues
cannot grow without limit; reconnect cannot move audio across authorization; and
measurements cannot hide their denominators.

The phase also made the limits equally explicit. There is no production capture or
transport, no production key management, no independently validated field model,
no operational reviewer and notification system, and no production security or
operations package. The final review therefore correctly says “not approved for
production.”

In simple terms: Phase 10 proves that the project's local live-processing rules are
coherent, bounded, testable, and measurable. It does not prove that the system is
ready to watch real people, detect real emergencies, or contact anyone. That
difference is the central technical and safety result of the phase.
