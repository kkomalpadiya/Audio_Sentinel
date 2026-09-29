# Phase 8 completion report: Offline Evaluator and Alerts

## 1. Executive summary

Phase 8 turned the analytical work from Phases 1 through 7 into a complete local
application workflow.

Before Phase 8, the project already knew how to:

- prepare an authorized recording;
- detect acoustic events;
- find speech and create accepted transcripts when consent allowed it;
- analyze accepted language;
- calculate a bounded risk score; and
- compare the evidence branches before choosing `no_action`, `log`, `review`, or
  `alert`.

Those capabilities were still separate building blocks. Phase 8 connected them,
created durable reports, recorded local alert-handling decisions, and exposed the
workflow through both a command line and a loopback-only HTTP API.

In common terms, Phase 8 answers:

> "How can one authorized local recording travel through the whole system, leave
> behind trustworthy records, and be operated safely without uploading data or
> automatically contacting anyone?"

The completed phase provides:

- one end-to-end offline evaluator for a recorded clip;
- one portable, versioned, self-checking final JSON report;
- one local audit record for every completed outcome;
- an optional pending local alert only for a verified `alert` outcome;
- a local command-line interface for evaluation and report inspection;
- a loopback-only HTTP API using the same application service as the CLI;
- atomic, tamper-aware, idempotent local persistence;
- privacy-minimized success and error responses;
- CLI-to-report integration coverage; and
- HTTP-to-evaluator integration coverage, including real failure recovery.

The strongest safety rule remains unchanged:

> A Phase 8 alert is a pending local record for human review. It is not a sent
> notification and does not authorize delivery.

At Phase 8 completion:

- all seven Phase 8 tasks are complete;
- 61 of the project's 72 tracked tasks are complete;
- all 87 dedicated Phase 8 tests pass;
- all 1,813 project tests pass;
- the generated audio-preparation smoke test passes;
- the command line and API use the same evaluation service;
- no Phase 8 request accepts an audio upload, URL, recipient, or delivery setting;
- no Phase 8 record contains transcript text, raw audio, tensors, or absolute local
  paths; and
- the combined local runtime was verified with TensorFlow 2.21.0, ONNX Runtime
  1.23.2, and Faster-Whisper 1.2.1.

Phase 8 does not claim that the system is ready for unattended real-world alerting.
It creates a controlled, reproducible, offline MVP boundary for the evaluation and
release work that follows in Phase 9.

## 2. What Phase 8 does in common terms

Imagine the earlier phases as a group of specialists:

- the preparation specialist standardizes the recording;
- the acoustic specialist identifies important sounds;
- the speech specialist finds and transcribes speech when permission exists;
- the language specialist examines accepted text for configured concerns and safe
  context;
- the risk specialist creates a bounded triage score; and
- the verification specialist decides whether the independent branches agree.

Phase 8 acts like the case coordinator and records clerk.

The coordinator:

1. checks that the request describes an authorized local recording;
2. calls the specialists in the required order;
3. skips specialists that consent does not permit;
4. preserves the exact result from each stage;
5. produces one final decision;
6. writes a self-checking local report;
7. writes a local handling receipt;
8. creates a pending local alert only when the verified outcome is `alert`; and
9. returns safe identifiers and relative paths to the operator.

The coordinator never emails, texts, calls, posts, or otherwise sends the result.
There is no recipient field and no notification transport in the Phase 8 workflow.

## 3. Phase 8 scope and task summary

| Task | Common description | Main result |
| --- | --- | --- |
| A8.1 | Connect the complete analysis pipeline | One-call offline evaluator |
| B8.1 | Save the completed evaluation safely | Versioned final JSON report |
| A8.2 | Record local handling and create a local alert when allowed | Audit record plus optional pending alert |
| B8.2 | Give an operator a local command interface | `evaluate` and `inspect-report` commands |
| A8.3 | Give a local program a validated HTTP interface | Loopback-only evaluation API |
| B8.3 | Test the command and report as one workflow | CLI/report integration suite |
| A8.4 | Test the API and evaluator as one workflow | HTTP/evaluator integration suite |

The work was committed task by task:

| Task | Commit | Commit message |
| --- | --- | --- |
| A8.1 | `ec54482` | `feat: add offline clip evaluator orchestration` |
| B8.1 | `abc1faf` | `feat: add versioned final report serialization` |
| A8.2 | `134a620` | `feat: add local alert audit records` |
| B8.2 | `b6cc5db` | `feat: add evaluation and report inspection CLI` |
| A8.3 | `356105e` | `feat: add loopback evaluation API` |
| B8.3 | `c519c05` | `test: add CLI report integration coverage` |
| A8.4 | `19302d3` | `test: add API evaluator integration coverage` |

The phase began at 54 completed tracker tasks and ended at 61 completed tasks.
The next task is B9.1, which builds repeatable evaluation manifests and metric
calculations.

## 4. Complete Phase 8 flow

```text
authorized local WAV or FLAC file under data/raw
                     |
                     v
        CLI command or loopback HTTP request
                     |
                     v
     strict request and consent-field validation
                     |
                     v
          shared evaluation application service
                     |
                     +--> load verified local YAMNet
                     |
                     +--> if speech is authorized:
                     |       load verified local Silero VAD
                     |       load verified local Faster-Whisper
                     |
                     v
             offline clip evaluator
                     |
                     +--> prepare mono 16 kHz audio
                     +--> run acoustic inference
                     +--> apply explicit acoustic thresholds
                     +--> aggregate overlapping detections
                     +--> persist and recheck acoustic evidence
                     +--> run permitted speech and language branches
                     +--> integrate risk inputs
                     +--> calculate bounded risk score
                     +--> compare evidence branches
                     +--> choose one consensus outcome
                     |
                     v
          build versioned final report in memory
                     |
                     +--> revalidate every embedded document
                     +--> bind source and evidence hashes
                     +--> recompute identities and summary
                     v
      stage, flush, verify, and atomically publish report
                     |
                     v
          reload the saved report and verify it again
                     |
                     v
             build local handling audit
                     |
                     +--> non-alert outcome: audit only
                     |
                     +--> alert outcome:
                     |       audit + pending local-only alert
                     v
      stage, flush, verify, and atomically publish audit bundle
                     |
                     v
       return privacy-minimized IDs and relative paths
                     |
                     v
       notification_delivery = "not_sent"
       alert_delivery_authorized = false
```

Every arrow represents a typed boundary. A later stage does not simply assume that
an earlier object is correct. Important identities, hashes, branch states, and
references are validated again before a durable record is published.

## 5. A8.1: One-call offline evaluator

### 5.1 What was built

`audio_sentinel.evaluator.OfflineClipEvaluator` connects the completed analysis
stages for one recorded clip.

The evaluator receives:

- shared project settings;
- a documented source-dataset name;
- already loaded and verified model objects;
- an explicit acoustic aggregation threshold policy;
- the existing versioned language, risk, and agreement rules, or explicitly
  trusted replacements; and
- one `InputAudio` record containing the clip identity, local path, and consent.

It returns one immutable `OfflineEvaluationResult` containing the validated output
from every completed stage.

### 5.2 Preflight checks

Before the evaluator writes prepared output, it checks that:

- the request is an `InputAudio` record;
- the effective preparation recipe produces mono audio;
- the prepared sample rate is 16 kHz, matching the selected models; and
- both VAD and transcription models are available when the consent scope includes
  speech.

These checks happen early so a known incompatible request does not leave a partial
evaluation behind.

### 5.3 Consent-directed execution

The evaluator has two permitted branches:

| Scope | Acoustic processing | Speech processing | Language processing |
| --- | --- | --- | --- |
| `acoustic_only` | Runs | Does not load or run | Does not run |
| `acoustic_and_speech` | Runs | Runs | Runs on accepted text |

For acoustic-only consent, the speech and language inputs are recorded as
`not_permitted`. They are not recorded as zeros. That difference matters:

- zero means the branch ran and found no countable result;
- `not_permitted` means the system was not allowed to run the branch.

The system never processes speech and discards it afterward. The prohibited work
does not happen at all.

### 5.4 Stage order

The evaluator runs the stages in this order:

1. validate the clip and effective audio recipe;
2. prepare and persist the audio windows;
3. run YAMNet over the prepared waveform windows;
4. apply the explicit per-label thresholds;
5. aggregate overlapping event candidates;
6. save and revalidate the acoustic evidence;
7. run VAD, speech-segment extraction, transcription, and language analysis when
   consent permits them;
8. convert the evidence into the compact Phase 6 risk-input contract;
9. calculate the bounded risk score;
10. evaluate evidence agreement and conflict; and
11. ask the final verification service for exactly one outcome.

### 5.5 Result boundary

The full result retains the stage objects needed by report serialization. A smaller
`to_summary()` view exposes only:

- opaque identities;
- event, segment, and finding counts;
- risk score and severity;
- outcome and review state;
- alert-candidate state; and
- `notification_sent=false`.

It excludes transcript text, matched phrases, audio samples, tensors, and local
paths.

### 5.6 Failure behavior

Evaluator-owned preflight problems receive stable codes such as:

- `invalid_clip`;
- `incompatible_audio_settings`;
- `speech_models_required`; and
- `invalid_prepared_output`.

Once a real stage starts, that stage's typed exception is allowed to keep its own
code. This makes a failure easier to diagnose without exposing private data or
hiding the stage where the problem occurred.

### 5.7 Verification

The 12 focused evaluator tests cover:

- a complete speech-authorized alert-candidate path;
- acoustic-only branch exclusion;
- no detected speech;
- missing speech models;
- incomplete model pairs;
- incompatible audio recipes;
- invalid acoustic model output;
- deterministic fixed-time retry and artifact reuse;
- privacy-minimized summaries;
- immutable results and policies; and
- stable invalid-input handling.

## 6. B8.1: Versioned final evaluation report

### 6.1 Why a separate final report was needed

The in-memory evaluator result is useful to the Python process, but it is not a
portable long-term record. It contains implementation objects and references to
stage results that need to be checked before being trusted later.

B8.1 creates one durable JSON document that another process can load and verify
without retaining the original in-memory evaluator object.

### 6.2 Report contents

The `FinalReportDocument` contains:

- schema and format version `1.0`;
- a semantic `report_id`;
- a timezone-aware UTC creation time;
- privacy-minimized source identity and provenance hashes;
- exactly three ordered evidence receipts: acoustic, speech, and language;
- a small derived summary;
- the complete trusted risk assessment;
- the complete evidence-agreement evaluation; and
- the complete final consensus decision.

The report does not contain:

- raw or prepared audio;
- transcript text;
- matched phrases;
- tensor values;
- an absolute or relative local artifact path;
- a recipient;
- notification transport settings; or
- delivery authorization.

### 6.3 Cross-document alignment algorithm

The report validator checks that the embedded records describe the same evaluation.
It verifies that:

1. the source identity matches the risk-input source;
2. each evidence receipt has the same status as its risk branch;
3. any evidence identity and SHA-256 match the reference used for risk scoring;
4. the agreement references the exact embedded assessment identity and hash;
5. the decision references the same assessment;
6. the decision branch states equal the agreement branch states;
7. the decision ID matches a fresh canonical calculation;
8. the summary equals a fresh derivation from the embedded documents; and
9. the report ID equals a fresh canonical calculation.

Changing one nested score or outcome is therefore not enough to make a valid
modified report. Every dependent identity and reference would also have to be
rebuilt through the trusted rules.

### 6.4 Semantic report identity

The report ID has this conceptual form:

```text
report_id = "report-" + SHA256(canonical semantic report content)
```

Canonical content means:

- keys are sorted;
- JSON uses stable separators;
- text is UTF-8;
- non-finite values are prohibited; and
- the report's own ID and report-creation timestamp are excluded from its semantic
  hash.

The timestamps already embedded in the risk assessment and decision remain part of
the semantic content. Saving the same already-built evaluation again can reuse the
same report. Running a new evaluation later normally creates a new assessment and
therefore a new report.

### 6.5 Persistence algorithm

Reports are stored at:

```text
data/processed/final-reports/<report_id>/report.json
```

Saving follows a staged commit algorithm:

1. build and validate the report in memory;
2. reload the current acoustic evidence and confirm it has not changed;
3. serialize the report within a configured byte limit;
4. create a private temporary directory under the final-report parent;
5. create `report.json` exclusively so an existing file is not replaced;
6. write the bytes, flush the stream, and call `fsync`;
7. read the staged file back and validate it;
8. atomically rename the complete staging directory to the semantic destination;
9. if the destination already exists, load it and require semantic equality; and
10. load the published report again and require exact equality.

Temporary staging data is removed after either success or failure.

### 6.6 Load-time checks

Loading accepts only a relative path within `data/processed`. It rejects:

- absolute paths;
- traversal such as `..`;
- linked or escaping paths;
- oversized files;
- malformed JSON;
- a report stored in the wrong identity directory;
- missing or extra bundle files; and
- any contract, hash, identity, or cross-document mismatch.

### 6.7 Evidence status behavior

The ordered evidence receipts distinguish:

| Status | Meaning |
| --- | --- |
| `present` | A completed artifact is pinned by identity and SHA-256. |
| `no_accepted_text` | Language analysis completed, but no transcript was accepted for scoring. |
| `not_permitted` | Consent excluded the branch; no artifact reference may exist. |
| `missing` | Expected evidence was unavailable. |
| `not_applicable` | The branch did not apply to this case. |

This avoids turning a consent restriction or missing artifact into a misleading
zero.

### 6.8 Verification

The 14 focused report tests cover both consent scopes, no accepted text, semantic
identity, atomic persistence, reuse, output limits, unsafe paths, changed evidence,
tampering, unexpected files, wrong identity directories, inconsistent evaluator
handoffs, immutability, privacy, and checked-in JSON Schema parity.

## 7. A8.2: Local alert and audit records

### 7.1 Why the audit layer is separate

The final report answers, "What did the evaluation conclude?"

The alert audit answers, "How was that conclusion handled locally?"

Keeping these records separate prevents report serialization from silently becoming
an alert-delivery system.

### 7.2 Outcome handling

Every verified final report receives one audit record.

| Final outcome | Audit action | Local alert document |
| --- | --- | --- |
| `no_action` | `no_alert_created` | No |
| `log` | `no_alert_created` | No |
| `review` | `no_alert_created` | No |
| `alert` | `local_alert_created` | Yes |

The audit does not upgrade a `review`, `log`, or `no_action` outcome into an alert.
It consumes the trusted decision exactly as recorded.

### 7.3 Local alert contract

When the outcome is `alert`, the local alert contains:

- a semantic `alert_id`;
- a hash-pinned report reference;
- clip, assessment, and decision identities;
- critical score and severity;
- consensus reason codes;
- `review_required=true`;
- `review_status=pending`;
- `local_only=true`;
- `notification_delivery=not_sent`; and
- `alert_delivery_authorized=false`.

The alert has no recipient and no transport configuration.

### 7.4 Audit contract

The audit records:

- the referenced final report and its hash;
- the consensus outcome;
- whether a local alert was created;
- the reason for that action;
- whether review is required;
- the optional local-alert identity and semantic hash; and
- the permanent no-delivery state.

### 7.5 Audit identity algorithm

Alert and audit identities use canonical SHA-256 content hashing.

The local alert hash excludes only its own ID and local creation timestamp. The
audit hash similarly excludes only its own ID and local creation timestamp.

That design makes a retry of the same already-built local handling action reusable,
while a meaningful change to the referenced report, outcome, reasons, review state,
or alert content creates a different identity.

### 7.6 Persistence algorithm

Bundles are stored at:

```text
data/processed/alert-audit/<audit_id>/audit.json
data/processed/alert-audit/<audit_id>/alert.json  # alert outcome only
```

The service:

1. reloads the saved final report through the full report verifier;
2. builds the expected audit and optional alert;
3. checks configured byte limits;
4. writes the documents to a private staging directory;
5. flushes and `fsync`s each file;
6. parses and validates the staged documents;
7. rebuilds the expected bundle from the report and compares it;
8. atomically publishes the directory;
9. reuses only a semantically identical existing bundle; and
10. reloads and revalidates the published report, audit, and optional alert.

### 7.7 Verification

The 15 focused audit tests cover alert and non-alert outcomes, pending review,
privacy, semantic identity, timestamp ordering, atomic persistence, reuse, exact
bundle inventory, size limits, path containment, tampering, wrong identity paths,
missing final reports, strict immutability, and checked-in schema parity.

## 8. B8.2: Local command-line interface

### 8.1 What was built

The installed `audio-sentinel` command has two Phase 8 operations:

```text
audio-sentinel evaluate
audio-sentinel inspect-report
```

### 8.2 `evaluate`

The command requires the operator to state:

- the path relative to `data/raw`;
- a clip ID;
- a consent ID;
- the authorized scope;
- a timezone-aware consent grant time;
- optional consent expiry;
- explicit device authorization;
- the documented source-dataset name; and
- an acoustic threshold from 0 through 1.

The threshold is required rather than hidden because the earlier acoustic
evaluation did not establish a production-calibrated operating threshold. An
explicit value is visible, reproducible, and auditable.

The command loads YAMNet for every evaluation. It loads Silero VAD and
Faster-Whisper only when the authorized scope includes speech.

The command then calls the shared Phase 8 service to evaluate the recording, save
the final report, and create the local audit bundle.

### 8.3 `inspect-report`

Report inspection is not a raw file dump. The command reloads the report through
all B8.1 path, size, inventory, schema, hash, identity, and alignment checks before
printing the privacy-minimized JSON view.

### 8.4 Output and errors

Success is machine-readable JSON on standard output. Expected failures are small
JSON objects on standard error with a stable code and safe explanation. Unexpected
exception details are replaced with a generic `unexpected_error` message.

The optional compact mode writes one-line JSON for scripts. Normal mode writes
indented JSON for people.

### 8.5 Why the command is a thin adapter

The CLI does not own another copy of the evaluation algorithm. It validates and
normalizes command input, then delegates to the shared application service.

This prevents the command and API from drifting into different scoring, consent,
report, or alert behavior.

### 8.6 Verification

The nine focused CLI tests cover help, command discovery, both processing scopes,
speech-model gating, report and audit persistence, alert creation, report
inspection, compact JSON, explicit authorization, timezone handling, unsafe paths,
privacy, and unexpected-error redaction.

## 9. A8.3: Shared application service and local API

### 9.1 Shared request contract

`EvaluationRequest` is a strict, immutable Pydantic model shared by the API and the
CLI service boundary.

It permits only:

- a portable relative raw-audio path using forward slashes;
- bounded opaque clip and consent identifiers;
- `acoustic_only` or `acoustic_and_speech`;
- the strict Boolean value `true` for device authorization;
- timezone-aware grant and optional expiry times;
- an expiry later than the grant;
- a bounded source-dataset label; and
- a real finite acoustic threshold from 0 through 1.

Unknown fields are rejected. The contract does not accept transcript text, an
upload, a URL, recipient information, a webhook, or notification authority.

### 9.2 Shared service algorithm

The application service:

1. verifies that settings and request objects passed the expected contracts;
2. creates application-owned directories;
3. converts request fields into one granted consent record;
4. loads the verified local models allowed by that scope;
5. builds the offline evaluator and explicit threshold policy;
6. evaluates one `InputAudio` record;
7. saves the final report;
8. converts its path to a portable path relative to `data/processed`;
9. saves the local audit and optional alert; and
10. returns one privacy-minimized `EvaluationResponse`.

### 9.3 Response contract

The response includes:

- clip identity and processing scope;
- outcome, score, severity, and review state;
- report ID and relative path;
- audit ID and relative path;
- optional alert ID and relative path; and
- fixed no-delivery fields.

The response validator recomputes the expected paths from the returned identities.
It also rejects an alert reference unless the outcome is `alert`, review is
required, and `alert_candidate=true`.

### 9.4 Loopback-only API

The FastAPI route is:

```text
POST /api/v1/evaluations
```

It accepts only loopback client addresses such as `127.0.0.1` and `::1`.
Non-loopback callers receive `403 local_access_required` before evaluation starts.

The API is intended to be served with Uvicorn bound to `127.0.0.1`, not
`0.0.0.0`.

### 9.5 Single-evaluation lock

Only one request can own the in-process evaluator at a time. A second concurrent
request receives `409 evaluation_busy`.

The lock is always released in a `finally` block, including when model loading,
audio processing, validation, or persistence fails.

This protects the prototype from concurrent requests multiplying the memory and
CPU cost of large local model runtimes.

### 9.6 HTTP status mapping

Typed domain codes are mapped into meaningful HTTP groups:

| HTTP status | Common meaning |
| --- | --- |
| 400 | The valid request could not be processed as requested. |
| 403 | Access, consent, device authorization, or scope did not permit processing. |
| 404 | The local recording was not found. |
| 409 | A run is active or immutable local state conflicts. |
| 422 | The request body failed the strict contract. |
| 503 | A pinned model or runtime is unavailable or invalid. |
| 500 | An unexpected internal failure occurred and details were redacted. |

Validation responses do not echo rejected private values.

### 9.7 Verification

The 23 focused API cases cover valid local use, every request constraint, unknown
fields, path rules, scopes, strict Booleans and numbers, time rules, loopback
enforcement, concurrency, typed status mapping, error redaction, lock release,
immutable requests, and OpenAPI documentation.

## 10. B8.3: CLI and final-report integration tests

### 10.1 Purpose

Focused tests prove individual contracts. B8.3 proves that the real command,
application service, evaluator stages, persistence services, and later inspection
work together.

The suite replaces only the three heavy model execution engines with deterministic
objects that satisfy the same metadata and tensor contracts.

### 10.2 Eight covered workflows

1. Acoustic-only CLI evaluation followed by public report and audit reload.
2. Speech-authorized evaluation producing one pending local alert.
3. Two later assessments of the same recording without overwriting the first.
4. Report inspection from a fresh Python process.
5. Rejection of a report whose nested content was changed.
6. Rejection of an unexpected file added to a report bundle.
7. Rejection of relative traversal before an outside file can be read.
8. Verification that returned paths are relative, portable, and bound to IDs.

### 10.3 Why two command runs do not reuse one report

A new command invocation creates a new time-stamped risk assessment and decision.
Those embedded stage timestamps are meaningful semantic content, so a later run
receives a different report and audit identity.

The original bundle remains byte-for-byte unchanged. Reuse still applies when the
same already-built typed result is saved again.

This distinction avoids pretending that a new assessment is the old assessment.

### 10.4 Privacy checks

The tests place a known private phrase in the synthetic transcription path and
search the CLI output and Phase 8 documents for it. They also search for the
absolute temporary project root. Neither appears.

## 11. A8.4: API and evaluator integration tests

### 11.1 Purpose

A8.4 keeps the HTTP route and the real evaluator active. It checks that request
fields reach the expected pipeline policies and that a genuine stage failure is
contained and recoverable.

### 11.2 Six covered workflows

1. Acoustic-only HTTP evaluation that never loads VAD or transcription.
2. Speech-authorized HTTP evaluation producing one verified pending local alert.
3. No detected speech, no transcription call, zero risk, and no action.
4. A high HTTP threshold suppressing a lower acoustic score in real aggregation.
5. A missing recording returning a safe 404 without final artifacts.
6. Invalid acoustic output returning a safe error, releasing the lock, and allowing
   a corrected retry of the same clip.

### 11.3 Why the failure test uses a real stage contract

It would be easy to replace the entire application service with a function that
simply raises an exception. That proves route error handling, but not what happens
after preparation has begun.

The A8.4 stage-failure test supplies a model with an invalid output tensor shape.
The real inference contract rejects it. The test then verifies that:

- no incomplete final report exists;
- no incomplete alert-audit bundle exists;
- the response contains no private path;
- the API lock is released; and
- replacing the bad model allows a successful retry.

## 12. Algorithms used and why they were chosen

### 12.1 Sequential staged orchestration

#### Selected approach

The evaluator uses a deterministic sequence of typed stages. Each stage consumes a
validated result from the previous stage.

#### Why it was selected

- The work has real dependencies: language cannot run before accepted transcripts,
  and consensus cannot run before risk scoring.
- Stage order makes provenance and failure location clear.
- One-record offline processing does not need distributed scheduling.
- It is easier to reproduce and test than asynchronous fan-out.

#### Alternatives considered

| Alternative | Why it was not selected for Phase 8 |
| --- | --- |
| Microservices and message queues | They add deployment, network, retry, ordering, and privacy complexity to a local MVP. |
| General DAG/workflow engine | Useful for large batch pipelines, but unnecessary for one fixed sequence. |
| Run every branch in parallel | Consent may prohibit speech, and later stages depend on earlier outputs. |

The sequential design is intentionally simple. Phase 9 can add repeatable batch
evaluation around it without changing the one-clip truth boundary.

### 12.2 Consent-directed branching

#### Selected approach

The consent scope decides whether the speech and language branch exists for the
evaluation.

#### Why it was selected

Privacy is stronger when prohibited processing never occurs. It also saves model
loading time, memory, and CPU.

#### Rejected alternative

Running speech processing for every recording and discarding the result afterward
would still perform unpermitted processing. Redaction after the fact is not a
substitute for consent enforcement.

### 12.3 Explicit threshold policy

#### Selected approach

The caller must supply the acoustic threshold, and the evaluator converts it into
an explicit aggregation policy.

#### Why it was selected

The project has not yet calibrated one production threshold against operational
false-positive and false-negative costs. Requiring the value prevents an arbitrary
default from looking authoritative.

#### Alternatives

| Alternative | Reason deferred or rejected |
| --- | --- |
| Hidden fixed threshold | Easy to use but easy to mistake for a validated production setting. |
| Adaptive threshold | Requires calibration data and a clearly evaluated adaptation rule. |
| Learned end-to-end threshold | Requires a representative labeled dataset and operational loss function. |

### 12.4 Deterministic risk and consensus rules

Phase 8 reuses the deterministic algorithms from Phases 5 through 7:

- versioned token and phrase language rules with bounded negation/context handling;
- an additive, independently capped risk score from 0 through 100;
- explicit missing-data and uncertainty handling;
- branch-specific support, neutral, conflict, and unavailable states; and
- conservative alert gates requiring a critical score, critical severity,
  acoustic support, language support, no conflict, no missing blocking evidence,
  and no unresolved uncertainty.

#### Why rules were retained

- No representative labeled incident corpus exists for training an end-to-end
  classifier.
- Every score contribution and decision reason remains inspectable.
- Results are deterministic and repeatable offline.
- Threshold boundaries can be tested exactly.
- A model confidence is not mistaken for operational certainty.

#### Why majority vote was not used

The branches do not have equal meanings. Speech presence alone is not threat
support, and a contradiction should not disappear because two other branches
vote differently. The agreement engine therefore uses branch-specific rules and
lets conflicts force review.

### 12.5 Canonical JSON and SHA-256 semantic identity

#### Selected approach

Reports, alerts, audits, assessments, and decisions derive identities from stable
canonical JSON content and SHA-256.

#### Why it was selected

- The same semantic object receives the same identity.
- A meaningful content change receives a different identity.
- IDs also act as integrity expectations for bundle directories.
- SHA-256 is available in Python's standard library and has a negligible collision
  risk for this use.
- The scheme requires no central database or ID service.

#### Alternatives

| Alternative | Trade-off |
| --- | --- |
| Random UUID | Unique, but does not reveal whether two documents have the same semantic content. |
| Database sequence | Requires a database and ties identity to insertion order instead of content. |
| MD5 or SHA-1 | Faster but obsolete for integrity-sensitive use. |
| HMAC or digital signature | Adds proof tied to a secret or private key, but Phase 8 has no key-management design. |

SHA-256 detects change. It does not prove authorship. A signed audit chain remains
future deployment work.

### 12.6 Atomic staging, flush, rename, and reload

#### Selected approach

Every durable bundle is built in a private staging directory, flushed, verified,
and atomically renamed into place.

#### Why it was selected

Writing directly to the final filename can leave a valid-looking but incomplete
file after a crash, disk error, or process interruption. Publishing only a complete
directory gives readers an all-or-nothing view on the same filesystem.

#### Why `fsync` and reload both exist

- `flush` moves Python's buffered bytes toward the operating system.
- `fsync` asks the operating system to flush the file's content to storage.
- readback catches serialization or file-content problems;
- model validation catches structural and semantic problems; and
- the final reload checks the public loader, not only a private write path.

#### Alternatives

| Alternative | Reason not selected |
| --- | --- |
| Direct overwrite | Can expose partially written or conflicting state. |
| Temporary file only | A report or audit is a bundle whose inventory must move together. |
| Database transaction | Strong option for query-heavy multi-user systems, but adds a database to a portable local artifact workflow. |

### 12.7 Idempotent compare-and-reuse

#### Selected approach

If a semantic destination already exists, Phase 8 loads and compares it. An exact
match is reused; any difference is an `output_conflict` and is never overwritten.

#### Why it was selected

- Safe retries are important after uncertain process outcomes.
- Immutable evidence should not change in place.
- Reuse avoids duplicate bundles for the same typed result.
- A conflict remains visible rather than being silently replaced.

### 12.8 Exact bundle inventory

#### Selected approach

A final-report directory must contain exactly `report.json`. An audit directory
must contain exactly the expected `audit.json` and, only for an alert outcome,
`alert.json`.

#### Why it was selected

Ignoring extra files can hide stale data, an abandoned write, or deliberately
inserted content. Exact inventory makes the bundle meaning unambiguous.

### 12.9 Path containment and portable relative paths

#### Selected approach

Public requests and responses use forward-slash relative paths within approved
project directories. Resolved paths are checked against their managed roots, and
linked/escaping paths are rejected.

#### Why it was selected

- Relative paths do not reveal a user's machine layout.
- They remain portable when the project directory moves.
- Containment blocks traversal into unrelated local files.
- The returned path can be recomputed from the artifact identity.

### 12.10 Strict contracts and fail-closed validation

#### Selected approach

Pydantic models reject unknown fields, invalid enums, naive timestamps, non-finite
numbers, inconsistent references, and invalid branch combinations.

#### Why it was selected

A loose dictionary can carry a typo or unrecognized field without warning. A
security- and privacy-sensitive boundary should reject uncertainty rather than
guess what the caller meant.

### 12.11 Shared service for CLI and API

#### Selected approach

Both operator interfaces call `evaluation_service.run_evaluation()`.

#### Why it was selected

Duplicated orchestration would eventually produce two definitions of consent,
model loading, scoring, persistence, or alert handling. One shared service keeps
the interfaces thin and behaviorally consistent.

### 12.12 Loopback access and a single in-process lock

#### Selected approach

The API accepts loopback clients only and serializes model evaluation with one
non-blocking lock.

#### Why it was selected

- The feature is a local integration boundary, not a public service.
- Large model runtimes can consume substantial memory.
- A single process with one active evaluation is easy to reason about and test.
- A busy response is safer than unbounded resource contention.

#### Alternatives

| Alternative | Reason deferred |
| --- | --- |
| Public network binding | Requires authentication, TLS, authorization, rate limiting, deployment hardening, and a broader threat model. |
| Thread pool with many simultaneous evaluations | Can multiply memory use and make model-runtime safety less predictable. |
| Durable job queue | Useful for batch service operation, but beyond the local MVP and not needed for one-record control. |

### 12.13 Deterministic model doubles in integration tests

#### Selected approach

Integration tests keep the production contracts and orchestration but substitute
small deterministic model objects for TensorFlow, ONNX Runtime, and
Faster-Whisper execution.

#### Why it was selected

- Tests run offline and do not download models.
- Results do not depend on CPU kernels, hardware, or probabilistic model changes.
- The suite is fast enough to run frequently.
- Failure shapes can be created deliberately.
- The code after the model boundary remains real.

#### Why real-model tests are not the only test strategy

Running every test with all real models would be slow, resource-heavy, and harder
to reproduce. It would also make precise outcome fixtures fragile. Real artifacts
and runtimes still have their own loader, metadata, checksum, and setup validation.
The combined runtime was also manually verified on the development machine.

### 12.14 Local alert record instead of automatic notification

#### Selected approach

An `alert` outcome creates a pending local document and an audit receipt.

#### Why it was selected

The system has not yet completed operational calibration, recipient authorization,
delivery policy, retention policy, or production security review. Automatically
contacting someone would turn an experimental classifier into an unattended safety
action. A pending local record preserves the result for review without creating
that risk.

### 12.15 Analysis algorithms that Phase 8 orchestrates

Phase 8 did not replace the validated algorithms from earlier phases. It calls them
through their public contracts. Their main choices are summarized here because they
determine the final Phase 8 result.

#### Audio preparation

The preparation pipeline converts allowed input into mono 16 kHz audio, applies the
configured RMS loudness normalization, and creates deterministic overlapping
windows.

Why this approach was used:

- 16 kHz mono matches YAMNet, Silero VAD, and Faster-Whisper;
- one shared prepared waveform prevents each model from inventing a different
  resampling path;
- RMS normalization is simple, bounded, and reproducible; and
- overlapping windows reduce boundary misses while recorded contribution maps make
  duplicates traceable.

The system does not normalize by uncontrolled peak amplification, and it does not
let zero padding become evidence.

#### Acoustic inference and aggregation

YAMNet emits scores for short model patches across 521 classes. A versioned mapping
translates selected classes into the smaller Audio Sentinel label set. When several
YAMNet classes support one project label, the algorithm keeps the highest relevant
score rather than adding them.

After explicit thresholding, candidate patches for the same label are joined when
they overlap, touch, or fall within the configured gap. The merged interval keeps
the maximum score and every contributing window/patch reference.

Why this approach was used:

- maximum score prevents related model classes from manufacturing confidence by
  addition;
- interval union prevents overlapping preparation windows from double-counting one
  real sound;
- label-specific processing avoids merging unlike events; and
- explicit thresholds make the operating policy visible.

A more complex temporal neural detector could learn event boundaries directly, but
would require suitable labeled temporal data and a new validation program.

#### Voice activity detection and speech interval merging

Silero VAD scores 512 real samples at a time, or 32 milliseconds at 16 kHz, with 64
samples of explicit preceding context. Qualifying frames are converted to absolute
sample positions and merged across overlapping windows. The final interval retains
the highest support score rather than summing duplicate support.

Sample indices are the source of truth; seconds are derived for display. This was
chosen over repeated floating-point timestamp arithmetic because integer sample
positions avoid accumulated rounding drift.

#### Transcription

Each final speech interval is reconstructed sample-exactly and sent to the pinned
English Faster-Whisper model as an independent call. Decoding is fixed to greedy
one-beam, temperature zero, no previous-text conditioning, no prompt/hotwords, and
no hidden internal VAD.

Why this approach was used:

- fixed decoding reduces hidden variability;
- independent calls prevent one segment from influencing another;
- the explicit VAD pipeline remains the single owner of speech boundaries; and
- local inference protects audio and transcript privacy.

Beam search or prompted decoding could improve some transcripts, but would add
state, compute, and prompt-governance questions before evaluation justifies them.

#### Language matching

Accepted transcripts are tokenized and matched against versioned keyword, phrase,
and explicit-negation rules. Rules are indexed by first token, overlapping matches
are resolved deterministically, and negation or quoted/reported context applies only
within bounded local windows.

This was chosen over global substring matching because local token boundaries and
bounded context reduce obvious false positives. It was chosen over embedding or
large-language-model classification because the project lacks a calibrated labeled
corpus and needs exact, offline explanations.

#### Risk scoring

The risk algorithm is an additive weighted policy with independent acoustic,
speech, and language caps plus a total cap of 100. Missing data and uncertainty
remain explicit, and a score at or above the configured review boundary requires
human review.

Independent caps stop one branch with many repeated findings from overwhelming the
entire score. The score is a triage policy value, not a probability of an incident.

#### Evidence agreement and final decision

The agreement algorithm classifies branches as supporting, neutral, conflicting,
or unavailable. It applies explicit cross-branch contradiction rules and lets a
conflict override apparent support. The final decision checks alert eligibility
first, then review, log, and no action.

The order matters because a valid alert is also review-required. Checking review
first would make alert unreachable. Conflicts and blocking uncertainty force review
instead of being averaged away.

## 13. Technology stack and why it was chosen

### 13.1 Python 3.11+

Python remains the primary implementation language.

Why it fits:

- strong scientific-audio and machine-learning ecosystems;
- direct support from all selected model runtimes;
- mature validation, API, CLI, and testing libraries;
- readable orchestration code; and
- modern typing, enums, dataclasses, and path APIs.

Using another language for Phase 8 would have added cross-language serialization
and duplicated contracts around an already Python-based analysis pipeline.

### 13.2 Pydantic 2

Pydantic defines strict immutable request, response, report, alert, audit, policy,
and nested evidence contracts.

Why it was chosen over plain dictionaries:

- runtime type and value validation;
- unknown-field rejection;
- field and cross-field validators;
- JSON parsing and serialization;
- frozen records;
- generated JSON Schema; and
- natural integration with FastAPI.

Dataclasses remain useful for trusted in-process composite results, while Pydantic
owns externally serialized and untrusted-input boundaries.

### 13.3 FastAPI

FastAPI provides the local HTTP boundary.

Why it was chosen:

- direct use of the same Pydantic request and response models;
- automatic OpenAPI documentation;
- explicit response status contracts;
- small routing layer; and
- a lightweight test client.

Flask could provide routing but would require more manual request/schema wiring.
Django would add a much larger web and database framework than this single local
endpoint needs.

### 13.4 Uvicorn

Uvicorn runs the FastAPI application locally.

It was chosen because it is a standard lightweight ASGI server for FastAPI. The
documented command binds it to `127.0.0.1`, keeping the endpoint local.

### 13.5 Typer

Typer provides the `audio-sentinel` console application.

Why it was chosen:

- typed command options;
- generated help text;
- clean subcommands;
- predictable exit handling; and
- low boilerplate around Python functions.

The standard-library `argparse` could also work, but Typer better matches the
project's type-driven contracts and keeps command declarations concise.

### 13.6 JSON and JSON Schema

JSON is the durable document format for reports, audits, alerts, and public command
or API messages.

Why it was chosen:

- human-readable;
- portable across operating systems and languages;
- easy to hash canonically;
- directly supported by Pydantic and web clients;
- easy to inspect during review; and
- compatible with checked-in JSON Schema.

SQLite or another database would improve querying and multi-user transactions, but
Phase 8 prioritizes immutable portable bundles. A binary format such as Protobuf is
compact, but less convenient for manual inspection and would add another schema and
toolchain.

### 13.7 SHA-256 through `hashlib`

Python's standard `hashlib` supplies semantic and artifact hashes.

It requires no external dependency and is appropriate for content identity and
change detection. The report clearly states that these hashes are not digital
signatures.

### 13.8 Standard-library filesystem primitives

Phase 8 uses `pathlib`, `tempfile`, `os`, `errno`, and atomic same-filesystem rename
behavior.

Why this stack was chosen:

- cross-platform path handling;
- no persistence framework dependency;
- explicit control over staging and `fsync`;
- straightforward containment checks; and
- artifacts remain normal local files.

### 13.9 TensorFlow 2.21.0 and YAMNet

YAMNet is the selected pretrained acoustic classifier and runs through TensorFlow.

Why it fits the project:

- broad pretrained acoustic vocabulary;
- no project-specific training requirement for the prototype;
- window-level scores that can feed explicit aggregation rules;
- local execution; and
- pinned model and class-map metadata.

Training a new acoustic neural network would require a larger representative
labeled corpus, compute, calibration, and model-governance work that the project
does not yet have.

### 13.10 ONNX Runtime 1.23.2 and Silero VAD

Silero VAD runs locally through ONNX Runtime.

Why it was preferred over a simple loudness threshold:

- it estimates speech presence rather than general sound energy;
- it is lightweight enough for local CPU use;
- the ONNX artifact has an explicit tensor contract; and
- it avoids cloud processing.

A raw energy gate would be simpler but would confuse many loud non-speech sounds
with speech and miss quieter speech.

### 13.11 Faster-Whisper 1.2.1 and CTranslate2

Faster-Whisper provides offline transcription through the CTranslate2 runtime.

Why it was chosen:

- local processing protects audio and transcript privacy;
- efficient CPU inference compared with the original reference runtime;
- explicit segment confidence information;
- compatible with the selected small English model; and
- no network request during evaluation.

A cloud speech API could be easier to scale but would violate the offline boundary
and introduce provider, credential, network, retention, and privacy concerns.

### 13.12 NumPy, SciPy, librosa, and SoundFile

These libraries support numeric audio arrays, signal processing, resampling,
feature work, and WAV/FLAC I/O inherited from earlier phases.

They were retained because Phase 8 orchestrates the existing tested preparation and
inference stack rather than rewriting audio processing inside the evaluator.

### 13.13 `threading.Lock`

The API's one-run guard uses the Python standard library.

It is sufficient for the single-process local server described by Phase 8. A
multi-process production deployment would need a process-external coordination
mechanism or job queue.

### 13.14 pytest 8, FastAPI TestClient, and HTTPX

pytest provides focused and integration testing. TestClient/HTTPX exercise the
actual HTTP contract without starting an external server.

Why this combination was chosen:

- parameterized boundary cases;
- temporary isolated project roots;
- precise exception and output assertions;
- real request serialization and validation;
- deterministic fixtures; and
- fast local execution.

### 13.15 Git and checked-in schemas

Production code, policies, tests, documentation, and JSON Schemas are versioned
together. A contract change can therefore be reviewed beside its implementation and
tests.

### 13.16 Technologies deliberately not added

Phase 8 does not require:

- a cloud platform;
- a hosted database;
- a message broker;
- a remote model API;
- a notification provider;
- a recipient directory;
- authentication for a public service;
- Docker or Kubernetes orchestration;
- a distributed tracing system; or
- a trained end-to-end incident classifier.

Those tools may become appropriate for a later production architecture, but adding
them now would expand the privacy, security, reliability, and operational scope
without solving the current local MVP requirement.

## 14. Data products created by Phase 8

### 14.1 Final report

```text
data/processed/final-reports/<report_id>/report.json
```

Purpose: durable, self-checking evaluation record.

### 14.2 Alert audit

```text
data/processed/alert-audit/<audit_id>/audit.json
```

Purpose: record how the final outcome was handled locally.

### 14.3 Optional local alert

```text
data/processed/alert-audit/<audit_id>/alert.json
```

Purpose: pending local review record for an actual consensus `alert` outcome.

### 14.4 Returned references

CLI and API responses use paths such as:

```text
final-reports/report-<sha256>/report.json
alert-audit/audit-<sha256>/audit.json
alert-audit/audit-<sha256>/alert.json
```

They never return an absolute `C:\...` or `/home/...` path.

## 15. Worked examples

### 15.1 Acoustic-only concerning sound

Suppose YAMNet produces an explosion score above the operator's explicit threshold,
but consent permits only acoustic processing.

The system:

1. runs preparation and acoustic analysis;
2. skips VAD, transcription, and language analysis;
3. records speech and language as `not_permitted`;
4. calculates risk from the allowed evidence;
5. prevents an acoustic-only case from becoming an automatic alert candidate;
6. records a review outcome;
7. saves a final report and audit; and
8. creates no local alert.

### 15.2 Aligned acoustic and language evidence

Suppose the acoustic branch detects a high-confidence explosion and accepted speech
contains an active threat phrase under the versioned language rules.

If the critical score and every consensus gate pass, the system:

1. records an `alert` outcome;
2. saves the report;
3. creates an audit with `local_alert_created`;
4. creates a pending local alert; and
5. records that delivery was not sent or authorized.

### 15.3 Speech scope but no detected speech

Suppose speech processing is permitted but VAD accepts no speech segment.

The system:

1. runs VAD;
2. does not call transcription because no accepted segment exists;
3. records language as `no_accepted_text`;
4. preserves the completed branch artifact;
5. can reach a zero-risk `no_action` result when no acoustic concern exists; and
6. writes an audit without an alert.

### 15.4 High acoustic threshold

Suppose the model score is `0.90` and the operator's experimental threshold is
`0.95`.

The event does not pass the aggregation threshold. In the tested no-other-evidence
case, the result is zero risk and `no_action`.

This example shows that the HTTP field reaches the real aggregation policy rather
than being accepted and ignored.

### 15.5 Missing recording

If the requested relative file does not exist, the API returns a safe
`file_not_found` response. It does not create a final report or audit bundle, does
not expose the resolved machine path, and releases the evaluator lock.

### 15.6 Invalid model output and retry

If the acoustic model returns an invalid tensor shape, the inference contract
rejects it with `invalid_output`. No final report or audit is published. After the
model is corrected, the same clip identity can be evaluated successfully.

### 15.7 Tampered saved report

If someone changes a nested score in `report.json`, report inspection recomputes the
contract and identities. The document is rejected as invalid instead of being
printed as if it were trusted.

## 16. Privacy, safety, and integrity controls

### 16.1 Local by design

Evaluation uses local files and local model artifacts. Phase 8 adds no cloud model
call, external upload, telemetry service, or remote notification.

### 16.2 Data minimization

Public summaries and Phase 8 records retain identities, counts, hashes, configured
outcomes, and reason codes. They omit sensitive media and text.

### 16.3 Consent before prohibited work

Speech models are not loaded for acoustic-only consent. This is stronger than
processing first and redacting later.

### 16.4 Human review remains required

An alert record starts pending and local-only. Phase 8 contains no transition that
turns it into an approved or delivered message.

### 16.5 Immutable records

Semantic destinations are never overwritten with different content. Conflicts are
errors.

### 16.6 Tamper evidence

Canonical hashes, identity directories, cross-document references, and public
reload validation make accidental or unauthorized changes detectable.

### 16.7 Bounded resources

Audio, report, alert, audit, and rule artifacts have configured size or count
limits. The API allows one active evaluation at a time.

### 16.8 Safe errors

Stable codes help operators and tests identify the failure class. Unknown details,
submitted invalid values, transcripts, and absolute paths are not echoed.

### 16.9 Exact provenance

Reports bind source hashes, evidence hashes, assessment identity, agreement
reference, and decision identity into one validated record.

### 16.10 No silent fallback

The workflow does not silently replace missing speech models, invalid tensors,
changed evidence, or conflicting persisted content with a plausible-looking result.

## 17. Testing and verification summary

### 17.1 Dedicated Phase 8 tests

| Test area | Collected cases |
| --- | ---: |
| Offline evaluator | 12 |
| Final report | 14 |
| Alert and audit | 15 |
| CLI boundary | 9 |
| API boundary | 23 |
| CLI/report integration | 8 |
| API/evaluator integration | 6 |
| **Total** | **87** |

The Phase 8 regression run used during completion also included two project-health
checks, for 89 passing tests in that command.

### 17.2 Full repository verification

At A8.4 completion:

- 1,813 project tests passed;
- two non-failing Pydantic deprecation warnings were reported by FastAPI OpenAPI
  generation;
- the generated 440 Hz stereo preparation smoke input was converted to 16 kHz mono;
- duration, five-window inventory, target loudness, source preservation, reuse, and
  temporary-file cleanup all passed; and
- the repository diff and tracker checks contained no unintended formula changes.

### 17.3 What deterministic tests prove

The suites prove contract, orchestration, persistence, branch, integrity, privacy,
and failure behavior for controlled inputs.

They do not prove real-world detection accuracy. That measurement belongs to the
Phase 9 evaluation manifest and metric work.

## 18. Main files produced or changed

### 18.1 Production modules

- `src/audio_sentinel/evaluator.py`: one-call offline evaluator.
- `src/audio_sentinel/final_report.py`: final report contracts, validation, and
  persistence.
- `src/audio_sentinel/alert_audit.py`: local alert and audit contracts and
  persistence.
- `src/audio_sentinel/cli.py`: `evaluate` and `inspect-report` commands.
- `src/audio_sentinel/evaluation_service.py`: shared CLI/API application service.
- `src/audio_sentinel/api.py`: loopback-only HTTP route.
- `src/audio_sentinel/main.py`: API registration, validation handler, and updated
  project status.

### 18.2 Tests

- `tests/test_evaluator.py`
- `tests/test_final_report.py`
- `tests/test_alert_audit.py`
- `tests/test_cli.py`
- `tests/test_api.py`
- `tests/test_cli_report_integration.py`
- `tests/test_api_evaluator_integration.py`

### 18.3 Documentation

- `docs/offline-evaluator.md`
- `docs/final-report.md`
- `docs/local-alert-audit.md`
- `docs/cli.md`
- `docs/local-api.md`
- `docs/cli-report-integration-tests.md`
- `docs/api-evaluator-integration-tests.md`

### 18.4 Schemas

- `docs/schemas/v1/final-report.schema.json`
- `docs/schemas/v1/alert-audit-record.schema.json`
- `docs/schemas/v1/local-alert.schema.json`

### 18.5 Tracker artifacts

Each Phase 8 task produced a reviewable tracker copy. The final Phase 8 tracker is:

```text
outputs/a8_4_tracker_update/Audio_Sentinel_Master_Task_List.xlsx
```

It records 61 complete, 0 in progress, and 11 not started tasks, with B9.1 next.

## 19. Reproduction commands and manual steps

### 19.1 Run the complete repository verification

```powershell
.\scripts\verify_project.ps1
```

### 19.2 Run the dedicated Phase 8 tests

```powershell
python -m pytest `
  tests/test_evaluator.py `
  tests/test_final_report.py `
  tests/test_alert_audit.py `
  tests/test_cli.py `
  tests/test_api.py `
  tests/test_cli_report_integration.py `
  tests/test_api_evaluator_integration.py -q
```

### 19.3 Start the local API

For acoustic-only evaluation:

```powershell
.\.venv\yamnet\Scripts\python.exe -m uvicorn audio_sentinel.main:app `
  --app-dir src --host 127.0.0.1 --port 8000
```

Do not bind this prototype to `0.0.0.0`.

### 19.4 Combined speech runtime

An `acoustic_and_speech` API request needs TensorFlow, ONNX Runtime, and
Faster-Whisper in the same Python process.

The manual setup step was:

```powershell
.\.venv\yamnet\Scripts\python.exe -m pip install -e ".[speech]"
```

The combined environment was verified with:

- TensorFlow 2.21.0;
- ONNX Runtime 1.23.2;
- Faster-Whisper 1.2.1; and
- a working installed `audio-sentinel` command.

No further manual step is required to use the committed deterministic test suites.
Real evaluation still requires the pinned local model artifacts and a consent-covered
recording under `data/raw`.

## 20. Important limitations

### 20.1 No empirical production calibration yet

The acoustic threshold is explicit because Phase 8 does not claim that one value is
operationally optimal. Risk weights and consensus gates are deterministic policy,
not probability estimates.

### 20.2 No real-world performance claim

Passing controlled integration scenarios does not establish sensitivity,
specificity, precision, recall, or acceptable false-alert rates on a representative
population.

### 20.3 Local API is not a production remote service

The loopback check and in-process lock are appropriate for the local prototype.
They are not replacements for production authentication, authorization, TLS,
rate-limiting, distributed coordination, or audit infrastructure.

### 20.4 SHA-256 is not a signature

Hashes detect content changes but do not identify the author. A production signed
chain would require key management and operational policy.

### 20.5 No alert delivery workflow

The system does not select recipients, approve alerts, send messages, retry
delivery, or record delivery receipts.

### 20.6 Retention and deletion remain open

Phase 8 creates durable local artifacts. B9.2 is responsible for retention,
deletion settings, and local audit-log tests.

### 20.7 Language coverage is bounded

The configured English rule set cannot recognize every paraphrase, language,
long-range context, joke, quotation, or ambiguous real-world situation.

### 20.8 Pretrained model limitations remain

YAMNet, Silero VAD, and Faster-Whisper can fail on noise, overlap, accents, unusual
devices, domain-specific sounds, short events, or distribution shift.

### 20.9 Single-process coordination

`threading.Lock` coordinates requests inside one process. A multi-worker server
would need an external lock or job queue to enforce global serialization.

### 20.10 Filesystem access control is external

The application validates paths and bundle content, but operating-system users who
can modify project files remain part of the deployment threat model.

## 21. What Phase 8 hands to Phase 9

Phase 9 receives a stable executable boundary:

- one request contract;
- one complete evaluation service;
- one portable final report format;
- one local audit and alert format;
- one CLI;
- one loopback API;
- stable error codes;
- deterministic integration fixtures; and
- verified local persistence behavior.

B9.1 can now build an evaluation manifest that repeatedly calls this same boundary
across a labeled collection and calculates metrics without inventing another
evaluation path.

The important distinction for Phase 9 is:

> Phase 8 proves that the workflow behaves consistently. Phase 9 measures how well
> its configured models, thresholds, and rules perform on evaluation data.

## 22. Short glossary

**Acoustic aggregation:** Combining window-level model scores into event candidates
using explicit thresholds and overlap rules.

**Alert candidate:** A verified critical result eligible to become a pending local
alert record. It is not a delivered message.

**Atomic publish:** Making a complete staged directory visible through one rename
rather than writing a final bundle piece by piece.

**Canonical JSON:** A stable JSON representation with deterministic key order and
formatting, suitable for repeatable hashing.

**Content-addressed identity:** An ID derived from the semantic content instead of a
random value or insertion number.

**Evidence receipt:** A privacy-minimized reference to one evidence branch,
including status, identity, and hash when an artifact exists.

**Fail closed:** Rejecting uncertain, invalid, or inconsistent state instead of
guessing or silently continuing.

**Idempotent retry:** Repeating the same operation produces or reuses the same
semantic result without overwriting different content.

**Loopback:** The local computer's network interface, usually `127.0.0.1` or `::1`.

**Model double:** A small deterministic test implementation that satisfies the same
contract as a large model runtime.

**Semantic hash:** A hash of the fields that define an artifact's meaning, excluding
only self-identity and the local creation timestamp where documented.

**VAD:** Voice activity detection, used to identify time regions likely to contain
speech.

## 23. Conclusion

Phase 8 completed the transition from separate analysis components to a usable
local evaluation application.

An authorized recording can now move through preparation, acoustic detection,
consent-aware speech and language processing, risk scoring, evidence agreement,
and final verification through one shared service. The result becomes a portable,
self-checking final report and a local handling audit. A verified alert outcome can
create a pending local alert, but nothing is sent and no delivery is authorized.

The main design choices favor determinism, consent enforcement, explicit policy,
content-addressed identity, atomic immutable persistence, safe local operation, and
human review. Those choices are less ambitious than an autonomous cloud alerting
platform, but they are better matched to the evidence and maturity of the current
prototype.

Phase 8 therefore ends with a trustworthy execution boundary rather than a claim of
real-world accuracy. Phase 9 can use that boundary to measure performance,
investigate errors, define retention, document limitations, and prepare the final
offline MVP demonstration.
