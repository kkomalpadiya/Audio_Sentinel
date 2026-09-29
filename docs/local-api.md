# Local evaluation API

## What A8.3 adds

A8.3 exposes one HTTP operation for the existing offline workflow:

```text
POST /api/v1/evaluations
```

The endpoint accepts metadata and a path relative to `data/raw`; it does not accept
audio uploads, microphone streams, URLs, recipient data, or notification settings.
It delegates to the same shared application service as the B8.2 CLI, so consent,
model loading, evaluation, final-report persistence, and local audit behavior cannot
drift between the two operator boundaries.

## Local-access boundary

The endpoint accepts only loopback clients such as `127.0.0.1` and `::1`. Start the
server bound to loopback as an additional deployment boundary:

```powershell
python -m uvicorn audio_sentinel.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Do not bind this prototype to `0.0.0.0`. A non-loopback request receives
`403 local_access_required` before evaluation begins.

Only one evaluation can use the in-process model runner at a time. A concurrent
request receives `409 evaluation_busy`; it does not cancel or release the active
run. This keeps the prototype's CPU and memory use bounded.

## Request

Example PowerShell request:

```powershell
$body = @{
  audio_path = "examples/authorized-recording.wav"
  clip_id = "demo-clip-001"
  consent_id = "consent-demo-001"
  processing_scope = "acoustic_only"
  device_authorized = $true
  granted_at = "2026-09-01T09:00:00+05:30"
  source_dataset = "local-authorized-demo"
  acoustic_threshold = 0.5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/evaluations" `
  -ContentType "application/json" `
  -Body $body
```

The request contract rejects unknown fields and enforces:

- a portable forward-slash path contained by `data/raw`, without an absolute path,
  drive prefix, empty component, or traversal;
- bounded opaque clip and consent identifiers;
- exactly `acoustic_only` or `acoustic_and_speech` scope;
- a strict Boolean `device_authorized=true` value;
- timezone-aware grant and optional expiry times, with expiry after grant;
- a bounded nonblank source-dataset label without control characters; and
- a numeric acoustic threshold from 0 through 1.

Malformed requests receive a small `422 invalid_request` response. Validation
details and submitted values are deliberately not echoed.

## Successful response

A successful evaluation returns HTTP `201` with JSON containing:

- the clip ID and authorized scope;
- risk score, severity, outcome, review state, and alert-candidate state;
- final-report and audit IDs plus paths relative to `data/processed`;
- an optional local-alert ID and relative path only for an alert outcome; and
- permanent `notification_delivery=not_sent` and
  `alert_delivery_authorized=false` values.

The response contract cross-checks every returned path against its artifact ID and
rejects an alert reference unless the outcome is an alert that requires review. It
contains no transcript text, raw audio, tensors, absolute paths, recipients, or
transport details.

## Safe failures

Typed pipeline errors retain their stable code and safe explanation. HTTP status
groups are:

| Status | Meaning |
| --- | --- |
| `400` | Valid JSON could not be processed as requested. |
| `403` | Local access or active consent did not authorize the operation. |
| `404` | The referenced recording was not found. |
| `409` | Another run is active or immutable persisted state conflicts. |
| `422` | Request JSON failed the strict contract. |
| `503` | A pinned model artifact or runtime is unavailable or invalid. |
| `500` | An unexpected failure occurred; internal details are redacted. |

Every failure releases the API's own evaluation lock. Existing stage persistence
remains governed by the idempotent and tamper-aware services from earlier phases.

## Runtime setup

The existing setup scripts correctly place YAMNet in `.venv/yamnet` and the speech
models in `.venv/speech`. For acoustic-only API use, run Uvicorn with the YAMNet
environment:

```powershell
.\.venv\yamnet\Scripts\python.exe -m uvicorn audio_sentinel.main:app `
  --app-dir src --host 127.0.0.1 --port 8000
```

An `acoustic_and_speech` request needs TensorFlow, ONNX Runtime, and Faster-Whisper
in the same Python process. On this development machine, add the speech runtime to
the existing YAMNet environment once:

```powershell
.\.venv\yamnet\Scripts\python.exe -m pip install -e ".[speech]"
```

This installs runtime packages only; evaluation still uses the already verified
local model artifacts and performs no model download.

## Verification

Run the focused boundary tests with:

```powershell
python -m pytest tests/test_api.py -q
```

The 23 cases cover successful local use, all request constraints, unknown fields,
privacy-safe validation, loopback enforcement, concurrency, typed status mapping,
unexpected-error redaction, lock release, immutable request records, and OpenAPI
scope/error documentation. The existing CLI and demo suites also verify that the
shared service and global validation handler do not regress those boundaries.

## Next task

B8.3 will expand CLI and final-report integration testing.

In plain language: the API is a small local front door, not a remote alert service.
It accepts only enough information to evaluate an already authorized recording,
then returns references to self-checking local records. It never receives or sends
the sensitive media itself.

