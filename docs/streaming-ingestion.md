# Local streaming audio ingestion

## Scope

B10.1 implements the first local client/service for the A10.1 live-device
contracts. It provides:

- append-only enrollment and revocation storage under managed processed data;
- one-time HMAC-SHA256 device authentication;
- authenticated session opening with receipt replay protection;
- exact ordered PCM chunk intake;
- synchronous audio handoff to a required sink; and
- explicit stream closure with audio-free receipts.

The implementation lives in `src/audio_sentinel/streaming_ingestion.py`. It is an
in-process boundary, not an HTTP, WebSocket, Bluetooth, microphone, or operating-
system audio service. A10.2 will connect accepted chunks to rolling analysis
windows. B10.2 will add buffering, reconnect, and backpressure behavior.

## Enrollment registry

`FileDeviceEnrollmentRegistry` implements the A10.1
`DeviceEnrollmentRegistry` protocol. It stores records under:

```text
data/processed/live-device-enrollments/<enrollment_id>/
```

An active enrollment is written once as `enrollment.json`. Revocation appends
`revoked.json`; it never rewrites the initial authorization. Registration is
idempotent only when the complete existing record matches. A reused identity with
different content fails with `enrollment_conflict`, and a second different
revocation fails with `revocation_conflict`.

The registry:

- derives paths only from validated opaque enrollment IDs;
- remains under the configured, unlinked processed-data directory;
- rejects links, noncanonical paths, unexpected files, empty files, oversized
  documents, malformed JSON, identity/path mismatches, and inconsistent
  revocations;
- writes a new enrollment through a private staging directory, flushes it, reloads
  it through the contract, and then renames the complete directory into place;
- writes revocation with exclusive creation and verifies it by reloading the
  complete two-file record; and
- stores the credential SHA-256 fingerprint but never the credential itself.

This is local application storage, not a hardware-backed device inventory or
organizational approval system. The opaque `authorization_reference` still points
to an externally governed approval record.

## One-time device authentication

`LocalDeviceAuthenticator` implements the A10.1 `DeviceAuthenticator` protocol
using HMAC-SHA256 challenge-response:

1. The service retrieves an active enrollment.
2. The authenticator creates 32 unpredictable bytes with `secrets.token_bytes()`.
3. It retains that challenge only in process for a short, bounded interval.
4. The client obtains its device credential through a caller-supplied provider and
   sends `HMAC-SHA256(credential, challenge)`.
5. The authenticator independently obtains the credential through a separate
   caller-supplied resolver.
6. It checks the credential's SHA-256 against the enrollment fingerprint and
   compares the HMAC in constant time.
7. It consumes the challenge on the first authentication attempt, including a
   failed attempt.
8. It issues a short-lived authentication receipt bound to the enrollment, device,
   credential fingerprint, and challenge.

The service then claims that exact receipt once before opening a stream. A forged,
expired, changed, or replayed receipt fails with `authentication_invalid`.
Challenges and unused receipts have fixed capacity and expiry limits so a caller
cannot create an unbounded in-memory authentication inventory.

The credential provider and resolver are integration points for approved local
secret storage. The module does not read environment variables, key files,
operating-system credential stores, or hardware keys by itself. It does not log or
persist secrets, challenges, HMAC proofs, or authentication receipts. Temporary
mutable secret copies are overwritten after HMAC calculation, within Python's
runtime limitations.

## Ingestion service

`LocalStreamingIngestionService` implements the A10.1
`StreamingAudioIngestor` protocol. A normal session uses this sequence:

1. `issue_challenge()` reloads the enrollment and rejects a missing, mismatched,
   revoked, or expired device.
2. `open_authenticated_stream()` authenticates the challenge response.
3. `open_stream()` claims the receipt once, reloads the enrollment to reject a
   change between authentication and opening, and calls the A10.1 consent/scope/
   format authorization function.
4. The service creates bounded in-memory state containing only the authorized
   session and its next expected sequence and sample positions.
5. `accept_chunk()` reloads the enrollment, verifies it remains active and exactly
   hash-bound to the session, validates the next chunk, and synchronously hands it
   to the configured sink.
6. Only after the sink returns successfully does the service advance its expected
   sequence and sample positions.
7. `finish_stream()` validates the final position, hands the close receipt to the
   sink, and removes the session state.

The service serializes session-state changes with a reentrant lock. Concurrent
attempts to submit the same sequence produce one acceptance and one deterministic
`sequence_mismatch`; a sink failure leaves the same position available for a safe
retry. The number of active sessions is hard bounded.

Midstream revocation or enrollment replacement removes the active session before
any later audio reaches the sink. Session expiry, consent expiry already bound into
the session, payload geometry, capture time, gaps, overlaps, duplicates, and
out-of-order chunks continue to fail through the A10.1 checks.

## Audio sink boundary

Every service requires a `StreamingAudioSink` implementation with two synchronous
methods:

- `accept_chunk(session, chunk, receipt)` receives one verified chunk and its
  audio-free receipt.
- `close_stream(session, receipt)` receives the verified end position.

The ingestion service itself keeps no audio queue, aggregate byte buffer, or audio
file. It creates no recording file. The sink must finish handling or copying the
chunk before returning and must raise an exception if handoff fails. B10.1 then
leaves the sequence position unchanged for a retry.

The sink is trusted application code. It must not persist raw audio unless
`session.raw_audio_persistence_allowed` is true and a later approved persistence
design enforces that permission. B10.1 supplies no persistence sink.

## Local client

`LocalStreamingClient` constructs the protocol correctly for one device and one
active session. It:

- requests a one-time challenge;
- obtains the credential through a provider only while calculating the HMAC;
- creates the session request with current consent, scope, exact format, and a
  fresh opaque session ID;
- calculates sample counts from interleaved PCM byte geometry;
- advances its sequence/sample position only after service acceptance; and
- builds the final close request from the last accepted position.

The client rejects a second simultaneous open, sending before open, empty or
unaligned PCM, and closing without an active session. It does not capture audio;
the caller supplies already captured PCM bytes.

## Minimal integration example

The following is an interface sketch. Real code must obtain both credential copies
from approved local secret storage and must provide genuine current consent:

```python
from audio_sentinel.config import load_settings
from audio_sentinel.streaming_ingestion import (
    FileDeviceEnrollmentRegistry,
    LocalDeviceAuthenticator,
    LocalStreamingClient,
    LocalStreamingIngestionService,
)

settings = load_settings()
settings.ensure_directories()
registry = FileDeviceEnrollmentRegistry(settings.paths)

# registry.register(enrollment) happens only after external device approval.
# credential_resolver and credential_provider must not hardcode a real secret.
authenticator = LocalDeviceAuthenticator(credential_resolver)
service = LocalStreamingIngestionService(registry, authenticator, rolling_sink)
client = LocalStreamingClient(
    service,
    enrollment_id="opaque-enrollment-id",
    device_id="opaque-device-id",
    credential_provider=credential_provider,
)

session = client.open(
    consent=current_consent,
    requested_scope=requested_scope,
    audio_format=enrolled_format,
)
client.send_pcm(pcm_s16le_bytes, captured_at=capture_time)
client.close()
```

Do not copy this sketch with placeholder credentials or consent into a real run.
The project does not yet include a production credential source or rolling sink.

## Failure behavior

Public failures use stable safe codes. Important groups include:

- `invalid_registry_path`, `invalid_registry_record`, `enrollment_conflict`, and
  `revocation_conflict` for storage integrity;
- `challenge_invalid`, `challenge_mismatch`, `authentication_failed`, and
  `authentication_invalid` for authentication;
- `enrollment_changed`, `enrollment_inactive`, `session_conflict`, and
  `session_not_active` for lifecycle changes;
- A10.1 protocol codes such as `authorization_expired`, `sequence_mismatch`,
  `sample_gap`, and `payload_size_mismatch`; and
- `sink_failed` when synchronous handoff does not complete.

Messages do not expose credentials, proofs, enrollment paths, raw audio,
transcripts, or sink exception text.

## Privacy, security, and availability limits

- The service is local and in-process. It provides no transport encryption,
  remote identity, network access control, or cross-process session recovery.
- HMAC authentication is only as secure as both credential providers and the
  surrounding process. Credential rotation and hardware-backed storage remain
  external responsibilities.
- Enrollment files contain opaque authorization and device references. Operators
  must keep personal names, room names, addresses, and network identifiers out of
  those values.
- Authentication challenges, receipts, and active sessions disappear on restart.
  A restart requires a new challenge and session.
- The service performs synchronous handoff and has no reconnect, queue, retry
  scheduler, latency policy, or backpressure signal. Those belong to B10.2.
- Accepted audio is not yet converted into rolling windows or evaluated. That is
  A10.2.
- No part of B10.1 creates an incident, sends a notification, authorizes alert
  delivery, or permits external action.

## Verification

Run the focused B10.1 suite:

```powershell
python -m pytest tests/test_streaming_ingestion.py -q
```

Run the live-contract and ingestion suites together:

```powershell
python -m pytest tests/test_live_streaming.py tests/test_streaming_ingestion.py -q
```

Run compilation and the complete project suite:

```powershell
python -m compileall -q src tests scripts
python -m pytest -q
```

## Next task

A10.2 will implement a sink that converts accepted PCM chunks into deterministic
rolling live windows and feeds those windows into the existing pipeline without
changing the B10.1 authentication, sequence, consent, or no-delivery boundaries.
