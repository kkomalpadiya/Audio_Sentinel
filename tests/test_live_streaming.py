"""A10.1 authorized-device and live-streaming contract tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from audio_sentinel.contracts import (
    ConsentRecord,
    ConsentStatus,
    ProcessingScope,
)
from audio_sentinel import live_streaming as live


NOW = datetime(2026, 10, 2, 18, 0, tzinfo=UTC)
FORMAT = live.StreamAudioFormat(
    sample_rate_hz=16_000,
    channels=1,
    chunk_duration_ms=20,
)


def consent(
    scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH,
    *,
    expires_at: datetime | None = None,
    retain_raw: bool = False,
) -> ConsentRecord:
    return ConsentRecord(
        consent_id="consent-live-001",
        status=ConsentStatus.GRANTED,
        processing_scope=scope,
        device_authorized=True,
        raw_audio_retention_allowed=retain_raw,
        granted_at=NOW - timedelta(hours=1),
        expires_at=expires_at,
    )


def enrollment(**changes: object) -> live.DeviceEnrollment:
    values: dict[str, object] = {
        "enrollment_id": "enrollment-001",
        "device_id": "device-001",
        "status": live.EnrollmentStatus.ACTIVE,
        "authorization_reference": "device-approval-001",
        "credential_fingerprint_sha256": "a" * 64,
        "authorized_scopes": (
            ProcessingScope.ACOUSTIC_ONLY,
            ProcessingScope.ACOUSTIC_AND_SPEECH,
        ),
        "audio_formats": (FORMAT,),
        "enrolled_at": NOW - timedelta(days=1),
        "expires_at": NOW + timedelta(days=30),
    }
    values.update(changes)
    return live.DeviceEnrollment(**values)


def request(**changes: object) -> live.StreamSessionRequest:
    values: dict[str, object] = {
        "session_id": "session-001",
        "enrollment_id": "enrollment-001",
        "device_id": "device-001",
        "session_challenge_sha256": "b" * 64,
        "requested_scope": ProcessingScope.ACOUSTIC_AND_SPEECH,
        "audio_format": FORMAT,
        "consent": consent(expires_at=NOW + timedelta(minutes=20)),
        "requested_at": NOW - timedelta(seconds=2),
    }
    values.update(changes)
    return live.StreamSessionRequest(**values)


def authentication(**changes: object) -> live.DeviceAuthenticationReceipt:
    values: dict[str, object] = {
        "authentication_id": "authentication-001",
        "verifier_id": "local-verifier-001",
        "enrollment_id": "enrollment-001",
        "device_id": "device-001",
        "credential_fingerprint_sha256": "a" * 64,
        "session_challenge_sha256": "b" * 64,
        "authenticated_at": NOW - timedelta(seconds=1),
        "expires_at": NOW + timedelta(minutes=5),
    }
    values.update(changes)
    return live.DeviceAuthenticationReceipt(**values)


def session() -> live.AuthorizedStreamSession:
    return live.authorize_stream_session(
        enrollment(), request(), authentication(), now=NOW
    )


def chunk(**changes: object) -> live.StreamAudioChunk:
    values: dict[str, object] = {
        "session_id": "session-001",
        "sequence_number": 0,
        "start_sample": 0,
        "sample_count": 320,
        "captured_at": NOW,
        "payload": bytes(640),
    }
    values.update(changes)
    return live.StreamAudioChunk(**values)


def test_audio_format_has_exact_bounded_pcm_geometry() -> None:
    assert FORMAT.encoding == "pcm_s16le"
    assert FORMAT.sample_width_bytes == 2
    assert FORMAT.max_chunk_samples == 320
    assert FORMAT.max_chunk_bytes == 640

    with pytest.raises(ValidationError, match="integral sample count"):
        live.StreamAudioFormat(
            sample_rate_hz=44_100,
            channels=1,
            chunk_duration_ms=11,
        )


def test_enrollment_is_canonical_bounded_and_contains_no_secret() -> None:
    record = enrollment()

    assert record.schema_version == "1.0"
    assert len(live.device_enrollment_sha256(record)) == 64
    serialized = record.model_dump_json()
    assert "credential_fingerprint_sha256" in serialized
    assert "credential_proof" not in serialized
    assert "secret" not in serialized

    with pytest.raises(ValidationError, match="canonical order"):
        enrollment(
            authorized_scopes=(
                ProcessingScope.ACOUSTIC_AND_SPEECH,
                ProcessingScope.ACOUSTIC_ONLY,
            )
        )
    with pytest.raises(ValidationError, match="exclude none"):
        enrollment(authorized_scopes=(ProcessingScope.NONE,))


def test_revoked_enrollment_requires_a_complete_bounded_revocation() -> None:
    revoked = enrollment(
        status=live.EnrollmentStatus.REVOKED,
        revoked_at=NOW,
        revocation_reference="revocation-001",
    )
    assert revoked.status is live.EnrollmentStatus.REVOKED

    with pytest.raises(ValidationError, match="requires revocation"):
        enrollment(status=live.EnrollmentStatus.REVOKED)
    with pytest.raises(ValidationError, match="cannot contain revocation"):
        enrollment(revoked_at=NOW, revocation_reference="revocation-001")


def test_session_request_enforces_consent_scope_and_aware_time() -> None:
    acoustic = request(
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        consent=consent(ProcessingScope.ACOUSTIC_AND_SPEECH),
    )
    assert acoustic.requested_scope is ProcessingScope.ACOUSTIC_ONLY

    with pytest.raises(ValidationError, match="exceeds the consent scope"):
        request(
            requested_scope=ProcessingScope.ACOUSTIC_AND_SPEECH,
            consent=consent(ProcessingScope.ACOUSTIC_ONLY),
        )
    with pytest.raises(ValidationError, match="timezone"):
        request(requested_at=NOW.replace(tzinfo=None))


def test_authorization_binds_device_authentication_consent_scope_and_format() -> None:
    stream = session()

    assert stream.enrollment_sha256 == live.device_enrollment_sha256(enrollment())
    assert stream.expires_at == authentication().expires_at
    assert stream.raw_audio_persistence_allowed is False
    assert stream.notification_delivery == "not_sent"
    assert stream.alert_delivery_authorized is False
    assert stream.external_action_authorized is False


def test_authorization_allows_explicit_scope_reduction_and_retention_state() -> None:
    stream_request = request(
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        consent=consent(
            ProcessingScope.ACOUSTIC_AND_SPEECH,
            expires_at=NOW + timedelta(minutes=2),
            retain_raw=True,
        ),
    )
    stream = live.authorize_stream_session(
        enrollment(), stream_request, authentication(), now=NOW
    )

    assert stream.processing_scope is ProcessingScope.ACOUSTIC_ONLY
    assert stream.raw_audio_persistence_allowed is True
    assert stream.expires_at == stream_request.consent.expires_at


@pytest.mark.parametrize(
    "record,stream_request,auth,code",
    [
        (
            enrollment(
                status=live.EnrollmentStatus.REVOKED,
                revoked_at=NOW,
                revocation_reference="revocation-001",
            ),
            request(),
            authentication(),
            "inactive_enrollment",
        ),
        (
            enrollment(),
            request(device_id="device-other"),
            authentication(),
            "identity_mismatch",
        ),
        (
            enrollment(),
            request(),
            authentication(session_challenge_sha256="c" * 64),
            "authentication_mismatch",
        ),
        (
            enrollment(authorized_scopes=(ProcessingScope.ACOUSTIC_ONLY,)),
            request(),
            authentication(),
            "scope_not_authorized",
        ),
        (
            enrollment(),
            request(),
            authentication(expires_at=NOW),
            "authentication_expired",
        ),
    ],
)
def test_authorization_fails_closed_with_stable_safe_codes(
    record: live.DeviceEnrollment,
    stream_request: live.StreamSessionRequest,
    auth: live.DeviceAuthenticationReceipt,
    code: str,
) -> None:
    with pytest.raises(live.LiveStreamingInterfaceError) as captured:
        live.authorize_stream_session(record, stream_request, auth, now=NOW)

    assert captured.value.code == code
    assert "credential_proof" not in str(captured.value)


def test_authorization_rejects_expired_consent_at_the_actual_open_time() -> None:
    stream_request = request(
        consent=consent(expires_at=NOW + timedelta(seconds=1))
    )

    with pytest.raises(live.LiveStreamingInterfaceError) as captured:
        live.authorize_stream_session(
            enrollment(),
            stream_request,
            authentication(expires_at=NOW + timedelta(minutes=1)),
            now=NOW + timedelta(seconds=1),
        )

    assert captured.value.code == "consent_not_active"


def test_authorization_revalidates_mutable_nested_consent_at_open_time() -> None:
    stream_request = request()
    stream_request.consent.processing_scope = ProcessingScope.ACOUSTIC_ONLY

    with pytest.raises(live.LiveStreamingInterfaceError) as captured:
        live.authorize_stream_session(
            enrollment(), stream_request, authentication(), now=NOW
        )

    assert captured.value.code == "scope_not_authorized"


def test_chunk_validation_returns_audio_free_contiguous_receipt() -> None:
    receipt = live.validate_stream_chunk(
        session(),
        chunk(),
        expected_sequence_number=0,
        expected_start_sample=0,
        now=NOW,
    )

    assert receipt.end_sample == 320
    assert receipt.payload_sha256 == (
        "9e132485d5107211de325a45e7917cbe3e4b5b9cde3e4ee91d7d2102317759ee"
    )
    assert receipt.raw_audio_persisted is False
    assert "payload" not in receipt.model_dump()


@pytest.mark.parametrize(
    "audio_chunk,expected_sequence,expected_start,code",
    [
        (chunk(session_id="session-other"), 0, 0, "session_mismatch"),
        (chunk(sequence_number=1), 0, 0, "sequence_mismatch"),
        (chunk(start_sample=1), 0, 0, "sample_gap"),
        (chunk(sample_count=321, payload=bytes(642)), 0, 0, "chunk_too_large"),
        (chunk(payload=bytes(638)), 0, 0, "payload_size_mismatch"),
        (
            chunk(captured_at=NOW + timedelta(seconds=1)),
            0,
            0,
            "invalid_time",
        ),
    ],
)
def test_chunk_validation_rejects_protocol_breaks(
    audio_chunk: live.StreamAudioChunk,
    expected_sequence: int,
    expected_start: int,
    code: str,
) -> None:
    with pytest.raises(live.LiveStreamingInterfaceError) as captured:
        live.validate_stream_chunk(
            session(),
            audio_chunk,
            expected_sequence_number=expected_sequence,
            expected_start_sample=expected_start,
            now=NOW,
        )

    assert captured.value.code == code


def test_close_request_must_match_the_accepted_stream_position() -> None:
    close = live.StreamCloseRequest(
        session_id="session-001",
        reason=live.StreamCloseReason.CLIENT_REQUEST,
        next_sequence_number=1,
        next_start_sample=320,
        requested_at=NOW,
    )
    receipt = live.validate_stream_close(
        session(),
        close,
        expected_sequence_number=1,
        expected_start_sample=320,
        now=NOW,
    )

    assert receipt.reason is live.StreamCloseReason.CLIENT_REQUEST
    assert receipt.notification_delivery == "not_sent"
    assert receipt.alert_delivery_authorized is False

    with pytest.raises(live.LiveStreamingInterfaceError) as captured:
        live.validate_stream_close(
            session(),
            close,
            expected_sequence_number=2,
            expected_start_sample=320,
            now=NOW,
        )
    assert captured.value.code == "close_position_mismatch"


class StubRegistry:
    def register(
        self, record: live.DeviceEnrollment
    ) -> live.DeviceEnrollment:
        return record

    def get(self, enrollment_id: str) -> live.DeviceEnrollment | None:
        return enrollment() if enrollment_id == "enrollment-001" else None

    def revoke(
        self, enrollment_id: str, *, revoked_at: datetime, revocation_reference: str
    ) -> live.DeviceEnrollment:
        return enrollment(
            status=live.EnrollmentStatus.REVOKED,
            revoked_at=revoked_at,
            revocation_reference=revocation_reference,
        )


class StubAuthenticator:
    def authenticate(
        self,
        record: live.DeviceEnrollment,
        stream_request: live.StreamSessionRequest,
        credential_proof: bytes,
    ) -> live.DeviceAuthenticationReceipt:
        del record, stream_request, credential_proof
        return authentication()


class StubIngestor:
    def open_stream(
        self,
        stream_request: live.StreamSessionRequest,
        record: live.DeviceEnrollment,
        auth: live.DeviceAuthenticationReceipt,
        *,
        now: datetime,
    ) -> live.AuthorizedStreamSession:
        return live.authorize_stream_session(record, stream_request, auth, now=now)

    def ingest_chunk(
        self,
        stream: live.AuthorizedStreamSession,
        audio_chunk: live.StreamAudioChunk,
        *,
        now: datetime,
    ) -> live.StreamChunkReceipt:
        return live.validate_stream_chunk(
            stream,
            audio_chunk,
            expected_sequence_number=0,
            expected_start_sample=0,
            now=now,
        )

    def close_stream(
        self,
        stream: live.AuthorizedStreamSession,
        close: live.StreamCloseRequest,
        *,
        now: datetime,
    ) -> live.StreamCloseReceipt:
        return live.validate_stream_close(
            stream,
            close,
            expected_sequence_number=close.next_sequence_number,
            expected_start_sample=close.next_start_sample,
            now=now,
        )


def test_stub_implementations_satisfy_runtime_protocols() -> None:
    assert isinstance(StubRegistry(), live.DeviceEnrollmentRegistry)
    assert isinstance(StubAuthenticator(), live.DeviceAuthenticator)
    assert isinstance(StubIngestor(), live.StreamingAudioIngestor)


def test_records_are_strict_and_immutable() -> None:
    record = enrollment()
    with pytest.raises(ValidationError):
        record.status = live.EnrollmentStatus.REVOKED  # type: ignore[misc]
    with pytest.raises(ValidationError):
        live.StreamAudioChunk(
            session_id="session-001",
            sequence_number=True,
            start_sample=0,
            sample_count=1,
            captured_at=NOW,
            payload=b"\x00\x00",
        )


def test_schema_export_matches_checked_in_documents(tmp_path: Path) -> None:
    documents = live.live_streaming_schema_documents()
    exported = live.write_live_streaming_schemas(tmp_path)
    schema_root = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "v1"

    assert set(documents) == {
        "device-enrollment.schema.json",
        "device-authentication-receipt.schema.json",
        "stream-session-request.schema.json",
        "authorized-stream-session.schema.json",
        "stream-audio-chunk.schema.json",
        "stream-chunk-receipt.schema.json",
        "stream-close-request.schema.json",
        "stream-close-receipt.schema.json",
    }
    for filename, document in documents.items():
        assert json.loads(exported[filename].read_text(encoding="utf-8")) == document
        assert json.loads((schema_root / filename).read_text(encoding="utf-8")) == document


def test_interface_import_has_no_network_or_model_runtime_dependency() -> None:
    code = """
import sys
class RejectImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'onnxruntime'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RejectImports())
def reject_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError('Unexpected network access')
sys.addaudithook(reject_network)
from audio_sentinel.live_streaming import DeviceEnrollment
assert DeviceEnrollment.model_fields['schema_version'].default == '1.0'
"""
    environment = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
    )

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        capture_output=True,
        timeout=30,
    )
