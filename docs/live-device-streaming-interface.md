# Authorized-device enrollment and streaming interface

## Scope

A10.1 defines the stable authorization and data boundary for future live-device
ingestion. It does not open a microphone, connect to a device, start a server,
store an enrollment, verify a cryptographic proof, persist raw audio, create
rolling analysis windows, or run the detection pipeline. B10.1 will implement the
first ingestion service against these contracts.

The interface lives in `src/audio_sentinel/live_streaming.py`. It is dependency
light and can be imported without a model runtime or network connection.

## Enrollment contract

`DeviceEnrollment` is an immutable, privacy-minimized authorization record for one
physical capture device. It records:

- opaque enrollment and device IDs;
- an opaque reference to the external device-approval record;
- a SHA-256 fingerprint of the enrolled credential, never the credential or
  secret itself;
- one or both supported processing scopes in canonical order;
- an exact bounded inventory of permitted audio formats;
- enrollment and expiry times; and
- either active status or a complete revocation time and reference.

An active enrollment cannot carry revocation fields. A revoked enrollment must
carry both. Every timestamp must include a timezone, expiry must follow enrollment,
`none` is not an authorized processing scope, duplicate audio formats are rejected,
and format order is canonical so a semantically stable record has a stable digest.

`device_enrollment_sha256()` hashes canonical JSON for the complete enrollment.
The digest lets a session identify the exact authorization record it used without
copying external approval material into later receipts.

## Authentication boundary

Enrollment and authentication are separate requirements. Knowing a device ID is
not proof that a caller controls the enrolled device.

`DeviceAuthenticationReceipt` represents the short-lived result of a future
trusted authenticator. It binds the enrollment ID, device ID, enrolled credential
fingerprint, and one session challenge digest to an authentication time and
expiry. It contains no credential proof or secret.

`DeviceAuthenticator` is the runtime-checkable implementation boundary. Its
future implementation receives credential proof only as a transient byte string
and must return a matching receipt without persisting that proof. A10.1 does not
choose a key scheme, provision credentials, or claim that constructing a receipt
is authentication; B10.1 must supply and test the trusted verifier.

## Session authorization

`StreamSessionRequest` requires an explicit session ID, the enrollment/device
pair, a fresh challenge digest, requested processing scope, exact audio format,
the existing consent record, and a timezone-aware request time.

`authorize_stream_session()` fails closed unless all of these checks pass:

1. the enrollment is active and inside its enrollment/expiry interval;
2. the request time is not later than the authorization time;
3. request and authentication identities match the enrollment;
4. authentication uses the enrolled credential fingerprint and the request's
   session challenge;
5. authentication occurred after the request and is still active;
6. the enrollment permits the requested processing scope and exact audio format;
7. consent is granted, device-authorized, timezone-aware, and active;
8. the requested scope does not exceed consent; and
9. a positive common authorization interval remains.

The authorized session expires at the earliest of the enrollment,
authentication, and consent expiries. Speech-capable consent may be narrowed to
`acoustic_only`; neither enrollment nor a request can expand consent. The session
also carries the consent's raw-audio-retention flag so later code cannot silently
assume permission to persist live audio.

Every authorization result permanently states
`notification_delivery=not_sent`, `alert_delivery_authorized=false`, and
`external_action_authorized=false`. Permission to capture and process a stream is
not permission to notify a recipient, dispatch a service, or claim that an
incident occurred.

## Audio format and chunk rules

Version 1 accepts signed 16-bit little-endian PCM in memory. Each
`StreamAudioFormat` fixes:

- sample rate from 8 kHz through 192 kHz;
- one through eight interleaved channels;
- a two-byte sample width; and
- a 10 ms through 1,000 ms maximum chunk duration that produces an integral
  sample count.

The resulting payload must stay at or below 1 MiB. A `StreamAudioChunk` carries
the session ID, zero-based sequence number, absolute zero-based start sample,
sample count, capture time, and bytes. It contains no file path, URL, recipient,
transport, transcript, model output, or alert authority.

`validate_stream_chunk()` requires the caller to provide the next expected
sequence and start sample. It rejects:

- a different session;
- a missing, repeated, or out-of-order sequence;
- any sample gap or overlap;
- capture time before authorization or after acceptance;
- expiry of the session;
- a chunk longer than the authorized duration; and
- a byte count inconsistent with sample count, channel count, and sample width.

On success it returns `StreamChunkReceipt`, which contains the exact sample span
and payload SHA-256 but never the audio bytes. A10.1 receipts always record
`raw_audio_persisted=false`; future persistence requires a separate implementation
that also enforces the session's retention permission.

## Stream closure

`StreamCloseRequest` records a close reason, next expected sequence, next expected
sample, and request time. `validate_stream_close()` requires those positions to
match the service's accepted state and returns an audio-free close receipt. Close
reasons distinguish normal client closure, disconnect, consent end,
authorization expiry, protocol error, and service shutdown.

A close receipt also preserves the no-notification and no-external-action
boundary. Closing a stream is not an evaluation result.

## Implementation protocols

Three runtime-checkable protocols define the handoff to later tasks:

- `DeviceEnrollmentRegistry` registers, retrieves, and revokes immutable
  enrollment records.
- `DeviceAuthenticator` verifies transient device proof and returns a short-lived
  receipt.
- `StreamingAudioIngestor` opens an authorized session, accepts ordered chunks,
  and closes the session.

These protocols do not select a database, key store, transport, concurrency model,
buffer, retry rule, or backpressure policy. B10.1 owns ingestion; A10.2 owns
rolling pipeline windows; B10.2 owns buffering, reconnect, and backpressure.

## Portable schemas

`live_streaming_schema_documents()` and
`write_live_streaming_schemas(output_directory)` expose checked-in v1 JSON
Schemas for enrollment, authentication, session authorization, audio chunks,
audio-free chunk receipts, and stream closure. The chunk schema can represent
bytes as base64 for contract interchange, but the schema does not authorize a
remote endpoint, message transport, logging, or persistence.

## Privacy and security limits

- Enrollment IDs, device IDs, authorization references, consent IDs, verifier
  IDs, and authentication IDs must remain opaque. Do not place names, locations,
  room labels, account names, or network addresses in them.
- The credential fingerprint is not a secret and cannot authenticate a device by
  itself. The future authenticator must verify possession using an approved local
  credential mechanism.
- A valid enrollment does not replace recording consent. A session needs both
  current enrollment and current consent every time it opens.
- Session challenge binding reduces receipt reuse but does not by itself provide
  replay protection. The B10.1 implementation must generate unpredictable
  challenges, enforce one-time use, and protect verifier state.
- No A10.1 function performs network access, device discovery, credential
  provisioning, secret storage, audio persistence, model inference, or alert
  delivery.

## Verification

Run the focused contract suite:

```powershell
python -m pytest tests/test_live_streaming.py -q
```

Run Python compilation and the full suite:

```powershell
python -m compileall -q src tests scripts
python -m pytest -q
```

## Next task

B10.1 will implement the streaming audio-ingestion client/service against this
interface. It must provide real enrollment storage and authentication behavior,
one-time challenge handling, session state, and ordered in-memory chunk intake
without broadening consent, persistence, notification, or external-action
authority.
