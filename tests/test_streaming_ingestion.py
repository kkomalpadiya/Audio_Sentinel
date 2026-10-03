"""B10.1 local enrollment, authentication, client, and ingestion tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import os
from pathlib import Path
import subprocess
import sys

import pytest

from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import (
    ConsentRecord,
    ConsentStatus,
    ProcessingScope,
)
from audio_sentinel.live_streaming import (
    DeviceAuthenticator,
    DeviceEnrollment,
    DeviceEnrollmentRegistry,
    EnrollmentStatus,
    StreamAudioChunk,
    StreamAudioFormat,
    StreamCloseReason,
    StreamSessionRequest,
    StreamingAudioIngestor,
)
from audio_sentinel import streaming_ingestion as ingestion


NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
SECRET = b"local-test-device-credential-0001"
FORMAT = StreamAudioFormat(
    sample_rate_hz=16_000,
    channels=1,
    chunk_duration_ms=20,
)


class CollectingSink:
    def __init__(self) -> None:
        self.chunks: list[tuple[object, bytes, object]] = []
        self.closes: list[tuple[object, object]] = []

    def accept_chunk(self, session, chunk, receipt) -> None:
        self.chunks.append((session, bytes(chunk.payload), receipt))

    def close_stream(self, session, receipt) -> None:
        self.closes.append((session, receipt))


class FailOnceSink(CollectingSink):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def accept_chunk(self, session, chunk, receipt) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("private sink detail")
        super().accept_chunk(session, chunk, receipt)


def active_consent() -> ConsentRecord:
    return ConsentRecord(
        consent_id="consent-live-b10-001",
        status=ConsentStatus.GRANTED,
        processing_scope=ProcessingScope.ACOUSTIC_AND_SPEECH,
        device_authorized=True,
        granted_at=NOW - timedelta(hours=1),
        expires_at=NOW + timedelta(hours=1),
    )


def enrollment(**changes: object) -> DeviceEnrollment:
    values: dict[str, object] = {
        "enrollment_id": "enrollment-b10-001",
        "device_id": "device-b10-001",
        "status": EnrollmentStatus.ACTIVE,
        "authorization_reference": "device-approval-b10-001",
        "credential_fingerprint_sha256": hashlib.sha256(SECRET).hexdigest(),
        "authorized_scopes": (
            ProcessingScope.ACOUSTIC_ONLY,
            ProcessingScope.ACOUSTIC_AND_SPEECH,
        ),
        "audio_formats": (FORMAT,),
        "enrolled_at": NOW - timedelta(days=1),
        "expires_at": NOW + timedelta(days=30),
    }
    values.update(changes)
    return DeviceEnrollment(**values)


def setup_registry(
    settings: AudioSentinelSettings,
) -> ingestion.FileDeviceEnrollmentRegistry:
    settings.ensure_directories()
    registry = ingestion.FileDeviceEnrollmentRegistry(settings.paths)
    registry.register(enrollment())
    return registry


def setup_service(
    settings: AudioSentinelSettings,
    *,
    current: list[datetime] | None = None,
    sink: CollectingSink | None = None,
    max_active_sessions: int = 32,
    max_pending_challenges: int = 128,
) -> tuple[
    ingestion.FileDeviceEnrollmentRegistry,
    ingestion.LocalDeviceAuthenticator,
    ingestion.LocalStreamingIngestionService,
    CollectingSink,
    list[datetime],
]:
    clock_value = current or [NOW]
    registry = setup_registry(settings)
    target_sink = sink or CollectingSink()
    authenticator = ingestion.LocalDeviceAuthenticator(
        lambda _record: SECRET,
        clock=lambda: clock_value[0],
        max_pending_challenges=max_pending_challenges,
    )
    service = ingestion.LocalStreamingIngestionService(
        registry,
        authenticator,
        target_sink,
        clock=lambda: clock_value[0],
        max_active_sessions=max_active_sessions,
    )
    return registry, authenticator, service, target_sink, clock_value


def setup_client(
    service: ingestion.LocalStreamingIngestionService,
    current: list[datetime],
    *,
    session_id: str = "session-b10-001",
    credential: bytes = SECRET,
) -> ingestion.LocalStreamingClient:
    return ingestion.LocalStreamingClient(
        service,
        enrollment_id="enrollment-b10-001",
        device_id="device-b10-001",
        credential_provider=lambda: credential,
        clock=lambda: current[0],
        session_id_factory=lambda: session_id,
    )


def test_file_registry_persists_reloads_and_reuses_exact_enrollment(
    temporary_settings: AudioSentinelSettings,
) -> None:
    temporary_settings.ensure_directories()
    registry = ingestion.FileDeviceEnrollmentRegistry(temporary_settings.paths)
    record = enrollment()

    assert registry.register(record) == record
    assert registry.register(record) == record
    assert registry.get(record.enrollment_id) == record
    directory = (
        temporary_settings.paths.processed_data
        / ingestion.LIVE_ENROLLMENT_DIRECTORY
        / record.enrollment_id
    )
    assert {item.name for item in directory.iterdir()} == {
        ingestion.ENROLLMENT_FILENAME
    }
    payload = (directory / ingestion.ENROLLMENT_FILENAME).read_bytes()
    assert SECRET not in payload
    assert b"credential_proof" not in payload


def test_registry_rejects_identity_conflict_and_unexpected_inventory(
    temporary_settings: AudioSentinelSettings,
) -> None:
    registry = setup_registry(temporary_settings)

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        registry.register(enrollment(authorization_reference="different-approval"))
    assert captured.value.code == "enrollment_conflict"

    directory = (
        temporary_settings.paths.processed_data
        / ingestion.LIVE_ENROLLMENT_DIRECTORY
        / "enrollment-b10-001"
    )
    (directory / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        registry.get("enrollment-b10-001")
    assert captured.value.code == "invalid_registry_record"


def test_registry_rejects_invalid_identifier_without_path_escape(
    temporary_settings: AudioSentinelSettings,
) -> None:
    temporary_settings.ensure_directories()
    registry = ingestion.FileDeviceEnrollmentRegistry(temporary_settings.paths)

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        registry.get("../outside")
    assert captured.value.code == "invalid_enrollment_id"
    assert not (temporary_settings.paths.processed_data.parent / "outside").exists()


def test_revocation_is_append_only_idempotent_and_blocks_new_challenges(
    temporary_settings: AudioSentinelSettings,
) -> None:
    registry, _authenticator, service, _sink, _current = setup_service(
        temporary_settings
    )
    initial_path = (
        temporary_settings.paths.processed_data
        / ingestion.LIVE_ENROLLMENT_DIRECTORY
        / "enrollment-b10-001"
        / ingestion.ENROLLMENT_FILENAME
    )
    initial_bytes = initial_path.read_bytes()

    revoked = registry.revoke(
        "enrollment-b10-001",
        revoked_at=NOW,
        revocation_reference="revocation-b10-001",
    )

    assert revoked.status is EnrollmentStatus.REVOKED
    assert registry.revoke(
        "enrollment-b10-001",
        revoked_at=NOW,
        revocation_reference="revocation-b10-001",
    ) == revoked
    assert initial_path.read_bytes() == initial_bytes
    assert (initial_path.parent / ingestion.REVOCATION_FILENAME).is_file()
    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        service.issue_challenge("enrollment-b10-001", "device-b10-001")
    assert captured.value.code == "inactive_enrollment"


def test_revocation_rejects_conflicting_second_record(
    temporary_settings: AudioSentinelSettings,
) -> None:
    registry = setup_registry(temporary_settings)
    registry.revoke(
        "enrollment-b10-001",
        revoked_at=NOW,
        revocation_reference="revocation-b10-001",
    )

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        registry.revoke(
            "enrollment-b10-001",
            revoked_at=NOW + timedelta(seconds=1),
            revocation_reference="other-revocation",
        )
    assert captured.value.code == "revocation_conflict"


def test_hmac_challenge_authentication_is_one_time_and_receipt_is_claimed_once(
    temporary_settings: AudioSentinelSettings,
) -> None:
    registry, authenticator, _service, _sink, _current = setup_service(
        temporary_settings
    )
    record = registry.get("enrollment-b10-001")
    assert record is not None
    challenge = authenticator.issue_challenge(record, now=NOW)
    request = StreamSessionRequest(
        session_id="session-b10-001",
        enrollment_id=record.enrollment_id,
        device_id=record.device_id,
        session_challenge_sha256=challenge.challenge_sha256,
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
        consent=active_consent(),
        requested_at=NOW,
    )
    proof = hmac.new(SECRET, challenge.challenge, hashlib.sha256).digest()

    receipt = authenticator.authenticate(record, request, proof)
    authenticator.claim_receipt(receipt, now=NOW)
    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        authenticator.claim_receipt(receipt, now=NOW)
    assert captured.value.code == "authentication_invalid"
    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        authenticator.authenticate(record, request, proof)
    assert captured.value.code == "challenge_invalid"


def test_wrong_hmac_consumes_challenge_and_returns_safe_error(
    temporary_settings: AudioSentinelSettings,
) -> None:
    registry, authenticator, _service, _sink, _current = setup_service(
        temporary_settings
    )
    record = registry.get("enrollment-b10-001")
    assert record is not None
    challenge = authenticator.issue_challenge(record, now=NOW)
    request = StreamSessionRequest(
        session_id="session-b10-001",
        enrollment_id=record.enrollment_id,
        device_id=record.device_id,
        session_challenge_sha256=challenge.challenge_sha256,
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
        consent=active_consent(),
        requested_at=NOW,
    )

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        authenticator.authenticate(record, request, bytes(32))
    assert captured.value.code == "authentication_failed"
    assert "secret" not in str(captured.value).lower()
    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        authenticator.authenticate(record, request, bytes(32))
    assert captured.value.code == "challenge_invalid"


def test_forged_authentication_receipt_cannot_open_a_stream(
    temporary_settings: AudioSentinelSettings,
) -> None:
    registry, authenticator, service, _sink, _current = setup_service(
        temporary_settings
    )
    record = registry.get("enrollment-b10-001")
    assert record is not None
    challenge = authenticator.issue_challenge(record, now=NOW)
    request = StreamSessionRequest(
        session_id="session-b10-forged",
        enrollment_id=record.enrollment_id,
        device_id=record.device_id,
        session_challenge_sha256=challenge.challenge_sha256,
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
        consent=active_consent(),
        requested_at=NOW,
    )
    proof = hmac.new(SECRET, challenge.challenge, hashlib.sha256).digest()
    authentic = authenticator.authenticate(record, request, proof)
    forged = authentic.model_copy(
        update={"authentication_id": "authentication-forged"}
    )

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        service.open_stream(request, record, forged, now=NOW)
    assert captured.value.code == "authentication_invalid"
    assert service.active_session_count == 0


def test_expired_and_capacity_limited_challenges_fail_closed(
    temporary_settings: AudioSentinelSettings,
) -> None:
    current = [NOW]
    registry, authenticator, _service, _sink, _current = setup_service(
        temporary_settings,
        current=current,
        max_pending_challenges=1,
    )
    record = registry.get("enrollment-b10-001")
    assert record is not None
    challenge = authenticator.issue_challenge(record)
    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        authenticator.issue_challenge(record)
    assert captured.value.code == "challenge_capacity_reached"

    current[0] += timedelta(seconds=31)
    replacement = authenticator.issue_challenge(record)
    assert replacement.challenge_sha256 != challenge.challenge_sha256


def test_client_service_ingests_contiguous_pcm_and_closes(
    temporary_settings: AudioSentinelSettings,
) -> None:
    _registry, _authenticator, service, sink, current = setup_service(
        temporary_settings
    )
    client = setup_client(service, current)

    session = client.open(
        consent=active_consent(),
        requested_scope=ProcessingScope.ACOUSTIC_AND_SPEECH,
        audio_format=FORMAT,
    )
    first = client.send_pcm(bytes(640))
    current[0] += timedelta(milliseconds=20)
    second = client.send_pcm(bytes(320))
    closed = client.close()

    assert session.session_id == "session-b10-001"
    assert (first.start_sample, first.end_sample) == (0, 320)
    assert (second.start_sample, second.end_sample) == (320, 480)
    assert (closed.next_sequence_number, closed.next_start_sample) == (2, 480)
    assert len(sink.chunks) == 2
    assert sink.chunks[0][1] == bytes(640)
    assert len(sink.closes) == 1
    assert client.session is None
    assert service.active_session_count == 0


def test_service_rejects_duplicate_or_out_of_order_chunk_without_advancing(
    temporary_settings: AudioSentinelSettings,
) -> None:
    _registry, _authenticator, service, sink, current = setup_service(
        temporary_settings
    )
    client = setup_client(service, current)
    session = client.open(
        consent=active_consent(),
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
    )
    first = StreamAudioChunk(
        session_id=session.session_id,
        sequence_number=0,
        start_sample=0,
        sample_count=320,
        captured_at=NOW,
        payload=bytes(640),
    )
    service.accept_chunk(session, first)

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        service.accept_chunk(session, first)
    assert captured.value.code == "sequence_mismatch"
    assert len(sink.chunks) == 1


def test_concurrent_duplicate_chunk_has_one_acceptance_and_one_rejection(
    temporary_settings: AudioSentinelSettings,
) -> None:
    _registry, _authenticator, service, sink, current = setup_service(
        temporary_settings
    )
    client = setup_client(service, current)
    session = client.open(
        consent=active_consent(),
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
    )
    audio_chunk = StreamAudioChunk(
        session_id=session.session_id,
        sequence_number=0,
        start_sample=0,
        sample_count=320,
        captured_at=NOW,
        payload=bytes(640),
    )

    def submit() -> str:
        try:
            service.accept_chunk(session, audio_chunk)
            return "accepted"
        except ingestion.StreamingIngestionError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _value: submit(), range(2)))

    assert sorted(results) == ["accepted", "sequence_mismatch"]
    assert len(sink.chunks) == 1


def test_sink_failure_keeps_stream_position_retryable(
    temporary_settings: AudioSentinelSettings,
) -> None:
    sink = FailOnceSink()
    _registry, _authenticator, service, _sink, current = setup_service(
        temporary_settings,
        sink=sink,
    )
    client = setup_client(service, current)
    client.open(
        consent=active_consent(),
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
    )

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        client.send_pcm(bytes(640))
    assert captured.value.code == "sink_failed"
    receipt = client.send_pcm(bytes(640))
    assert receipt.sequence_number == 0
    assert len(sink.chunks) == 1


def test_midstream_revocation_ends_service_state_before_handoff(
    temporary_settings: AudioSentinelSettings,
) -> None:
    registry, _authenticator, service, sink, current = setup_service(
        temporary_settings
    )
    client = setup_client(service, current)
    client.open(
        consent=active_consent(),
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
    )
    registry.revoke(
        "enrollment-b10-001",
        revoked_at=NOW,
        revocation_reference="revocation-b10-midstream",
    )

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        client.send_pcm(bytes(640))
    assert captured.value.code == "enrollment_inactive"
    assert service.active_session_count == 0
    assert sink.chunks == []


def test_client_rejects_second_open_and_unaligned_pcm(
    temporary_settings: AudioSentinelSettings,
) -> None:
    _registry, _authenticator, service, _sink, current = setup_service(
        temporary_settings
    )
    client = setup_client(service, current)
    client.open(
        consent=active_consent(),
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
    )

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        client.open(
            consent=active_consent(),
            requested_scope=ProcessingScope.ACOUSTIC_ONLY,
            audio_format=FORMAT,
        )
    assert captured.value.code == "client_session_active"
    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        client.send_pcm(b"\x00")
    assert captured.value.code == "invalid_payload"


def test_session_capacity_is_bounded_and_does_not_replace_active_session(
    temporary_settings: AudioSentinelSettings,
) -> None:
    _registry, _authenticator, service, _sink, current = setup_service(
        temporary_settings,
        max_active_sessions=1,
    )
    first = setup_client(service, current, session_id="session-b10-first")
    second = setup_client(service, current, session_id="session-b10-second")
    first.open(
        consent=active_consent(),
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
    )

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        second.open(
            consent=active_consent(),
            requested_scope=ProcessingScope.ACOUSTIC_ONLY,
            audio_format=FORMAT,
        )
    assert captured.value.code == "session_capacity_reached"
    assert service.active_session_count == 1


def test_authorization_expiry_blocks_later_chunk(
    temporary_settings: AudioSentinelSettings,
) -> None:
    current = [NOW]
    _registry, _authenticator, service, sink, _current = setup_service(
        temporary_settings,
        current=current,
    )
    client = setup_client(service, current)
    client.open(
        consent=active_consent(),
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
    )
    current[0] += timedelta(seconds=30)

    with pytest.raises(ingestion.StreamingIngestionError) as captured:
        client.send_pcm(bytes(640), captured_at=NOW)
    assert captured.value.code == "authorization_expired"
    assert sink.chunks == []


def test_service_does_not_create_audio_files(
    temporary_settings: AudioSentinelSettings,
) -> None:
    _registry, _authenticator, service, _sink, current = setup_service(
        temporary_settings
    )
    client = setup_client(service, current)
    client.open(
        consent=active_consent(),
        requested_scope=ProcessingScope.ACOUSTIC_ONLY,
        audio_format=FORMAT,
    )
    client.send_pcm(bytes(640))
    client.close(StreamCloseReason.CLIENT_REQUEST)

    files = [item for item in temporary_settings.paths.root.rglob("*") if item.is_file()]
    assert {item.suffix for item in files} == {".json"}
    assert all(item.name in {ingestion.ENROLLMENT_FILENAME} for item in files)


def test_implementations_satisfy_a10_1_protocols(
    temporary_settings: AudioSentinelSettings,
) -> None:
    registry, authenticator, service, sink, _current = setup_service(
        temporary_settings
    )

    assert isinstance(registry, DeviceEnrollmentRegistry)
    assert isinstance(authenticator, DeviceAuthenticator)
    assert isinstance(service, StreamingAudioIngestor)
    assert isinstance(sink, ingestion.StreamingAudioSink)


def test_module_import_has_no_network_or_model_runtime_dependency() -> None:
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
from audio_sentinel.streaming_ingestion import LocalStreamingClient
assert LocalStreamingClient.__name__ == 'LocalStreamingClient'
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
