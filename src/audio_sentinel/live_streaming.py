"""A10.1 contracts for authorized live-device enrollment and audio streaming."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.contracts import (
    ConsentRecord,
    ConsentStatus,
    ProcessingScope,
)
from audio_sentinel.preparation import Identifier


DEVICE_ENROLLMENT_SCHEMA_VERSION = "1.0"
STREAMING_INTERFACE_VERSION = "1.0"
MAX_STREAM_CHUNK_BYTES = 1_048_576


class LiveStreamingInterfaceError(RuntimeError):
    """Stable failure code with a safe, non-sensitive explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class EnrollmentStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"


class StreamCloseReason(str, Enum):
    CLIENT_REQUEST = "client_request"
    DEVICE_DISCONNECTED = "device_disconnected"
    CONSENT_ENDED = "consent_ended"
    AUTHORIZATION_EXPIRED = "authorization_expired"
    PROTOCOL_ERROR = "protocol_error"
    SERVICE_SHUTDOWN = "service_shutdown"


class LiveStreamingRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


def _clock(value: datetime) -> datetime:
    try:
        return _require_aware(value, "now")
    except ValueError as error:
        raise LiveStreamingInterfaceError("invalid_time", str(error)) from error


def _scope_rank(scope: ProcessingScope) -> int:
    return {
        ProcessingScope.NONE: 0,
        ProcessingScope.ACOUSTIC_ONLY: 1,
        ProcessingScope.ACOUSTIC_AND_SPEECH: 2,
    }[scope]


class StreamAudioFormat(LiveStreamingRecord):
    """Bounded in-memory PCM format accepted by the future ingestion service."""

    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate_hz: int = Field(ge=8_000, le=192_000, strict=True)
    channels: int = Field(ge=1, le=8, strict=True)
    sample_width_bytes: Literal[2] = 2
    chunk_duration_ms: int = Field(ge=10, le=1_000, strict=True)

    @model_validator(mode="after")
    def validate_integral_chunk_size(self) -> "StreamAudioFormat":
        if self.sample_rate_hz * self.chunk_duration_ms % 1_000:
            raise ValueError(
                "sample_rate_hz and chunk_duration_ms must produce an integral sample count"
            )
        if self.max_chunk_bytes > MAX_STREAM_CHUNK_BYTES:
            raise ValueError("audio format exceeds the maximum stream chunk size")
        return self

    @property
    def max_chunk_samples(self) -> int:
        return self.sample_rate_hz * self.chunk_duration_ms // 1_000

    @property
    def max_chunk_bytes(self) -> int:
        return self.max_chunk_samples * self.channels * self.sample_width_bytes


class DeviceEnrollment(LiveStreamingRecord):
    """Privacy-minimized authorization record for one physical capture device."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/device-enrollment.schema.json"
        },
    )

    schema_version: Literal["1.0"] = DEVICE_ENROLLMENT_SCHEMA_VERSION
    enrollment_id: Identifier
    device_id: Identifier
    status: EnrollmentStatus
    authorization_reference: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        description="Opaque reference to device approval held outside this record.",
    )
    credential_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    authorized_scopes: tuple[ProcessingScope, ...] = Field(min_length=1, max_length=2)
    audio_formats: tuple[StreamAudioFormat, ...] = Field(min_length=1, max_length=16)
    enrolled_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    revocation_reference: str | None = Field(
        default=None,
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )

    @field_validator("enrolled_at", "expires_at", "revoked_at")
    @classmethod
    def validate_timestamps(
        cls, value: datetime | None, info: object
    ) -> datetime | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "timestamp")
        return _require_aware(value, field_name)

    @model_validator(mode="after")
    def validate_enrollment(self) -> "DeviceEnrollment":
        if self.expires_at <= self.enrolled_at:
            raise ValueError("expires_at must be later than enrolled_at")

        expected_scopes = tuple(
            scope
            for scope in (
                ProcessingScope.ACOUSTIC_ONLY,
                ProcessingScope.ACOUSTIC_AND_SPEECH,
            )
            if scope in self.authorized_scopes
        )
        if self.authorized_scopes != expected_scopes:
            raise ValueError(
                "authorized_scopes must be unique, exclude none, and use canonical order"
            )

        format_keys = tuple(
            (
                item.encoding,
                item.sample_rate_hz,
                item.channels,
                item.sample_width_bytes,
                item.chunk_duration_ms,
            )
            for item in self.audio_formats
        )
        if len(set(format_keys)) != len(format_keys):
            raise ValueError("audio_formats must not contain duplicates")
        if format_keys != tuple(sorted(format_keys)):
            raise ValueError("audio_formats must use canonical order")

        revocation_fields_present = (
            self.revoked_at is not None,
            self.revocation_reference is not None,
        )
        if self.status is EnrollmentStatus.ACTIVE and any(revocation_fields_present):
            raise ValueError("active enrollment cannot contain revocation fields")
        if self.status is EnrollmentStatus.REVOKED and not all(revocation_fields_present):
            raise ValueError("revoked enrollment requires revocation time and reference")
        if self.revoked_at is not None:
            if not self.enrolled_at <= self.revoked_at <= self.expires_at:
                raise ValueError("revoked_at must fall within the enrollment interval")
        return self


class StreamSessionRequest(LiveStreamingRecord):
    """One explicit request to start an authorized live-audio session."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/stream-session-request.schema.json"
        },
    )

    interface_version: Literal["1.0"] = STREAMING_INTERFACE_VERSION
    session_id: Identifier
    enrollment_id: Identifier
    device_id: Identifier
    session_challenge_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    requested_scope: ProcessingScope
    audio_format: StreamAudioFormat
    consent: ConsentRecord
    requested_at: datetime

    @field_validator("requested_at")
    @classmethod
    def validate_requested_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "requested_at")

    @model_validator(mode="after")
    def validate_request(self) -> "StreamSessionRequest":
        if self.requested_scope is ProcessingScope.NONE:
            raise ValueError("requested_scope must permit acoustic processing")
        if self.consent.status is not ConsentStatus.GRANTED:
            raise ValueError("streaming requires granted consent")
        if not self.consent.device_authorized:
            raise ValueError("streaming requires an authorized device")
        if _scope_rank(self.requested_scope) > _scope_rank(
            self.consent.processing_scope
        ):
            raise ValueError("requested_scope exceeds the consent scope")
        if self.consent.granted_at is None:
            raise ValueError("streaming consent requires granted_at")
        _require_aware(self.consent.granted_at, "consent.granted_at")
        if self.consent.expires_at is not None:
            _require_aware(self.consent.expires_at, "consent.expires_at")
        if self.requested_at < self.consent.granted_at:
            raise ValueError("requested_at cannot precede consent grant")
        if (
            self.consent.expires_at is not None
            and self.requested_at >= self.consent.expires_at
        ):
            raise ValueError("requested_at must precede consent expiry")
        return self


class DeviceAuthenticationReceipt(LiveStreamingRecord):
    """Short-lived receipt from a future trusted device authenticator."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/device-authentication-receipt.schema.json"
        },
    )

    schema_version: Literal["1.0"] = DEVICE_ENROLLMENT_SCHEMA_VERSION
    authentication_id: Identifier
    verifier_id: Identifier
    enrollment_id: Identifier
    device_id: Identifier
    credential_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    session_challenge_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    authenticated_at: datetime
    expires_at: datetime

    @field_validator("authenticated_at", "expires_at")
    @classmethod
    def validate_timestamps(cls, value: datetime, info: object) -> datetime:
        return _require_aware(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def validate_interval(self) -> "DeviceAuthenticationReceipt":
        if self.expires_at <= self.authenticated_at:
            raise ValueError("authentication expiry must follow authentication time")
        return self


class AuthorizedStreamSession(LiveStreamingRecord):
    """Bounded authorization result; it is not proof of an incident or alert authority."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/authorized-stream-session.schema.json"
        },
    )

    interface_version: Literal["1.0"] = STREAMING_INTERFACE_VERSION
    session_id: Identifier
    enrollment_id: Identifier
    enrollment_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    authentication_id: Identifier
    device_id: Identifier
    consent_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    processing_scope: ProcessingScope
    audio_format: StreamAudioFormat
    authorized_at: datetime
    expires_at: datetime
    raw_audio_persistence_allowed: bool
    notification_delivery: Literal["not_sent"] = "not_sent"
    alert_delivery_authorized: Literal[False] = False
    external_action_authorized: Literal[False] = False

    @field_validator("authorized_at", "expires_at")
    @classmethod
    def validate_timestamps(cls, value: datetime, info: object) -> datetime:
        return _require_aware(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def validate_session(self) -> "AuthorizedStreamSession":
        if self.processing_scope is ProcessingScope.NONE:
            raise ValueError("authorized session requires an acoustic processing scope")
        if self.expires_at <= self.authorized_at:
            raise ValueError("session expiry must follow authorization time")
        return self


class StreamAudioChunk(LiveStreamingRecord):
    """One ordered, in-memory PCM chunk supplied to the ingestion boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/stream-audio-chunk.schema.json"
        },
    )

    interface_version: Literal["1.0"] = STREAMING_INTERFACE_VERSION
    session_id: Identifier
    sequence_number: int = Field(ge=0, strict=True)
    start_sample: int = Field(ge=0, strict=True)
    sample_count: int = Field(gt=0, strict=True)
    captured_at: datetime
    payload: bytes = Field(min_length=1, max_length=MAX_STREAM_CHUNK_BYTES)

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "captured_at")


class StreamChunkReceipt(LiveStreamingRecord):
    """Audio-free acceptance metadata returned by the ingestion boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/stream-chunk-receipt.schema.json"
        },
    )

    interface_version: Literal["1.0"] = STREAMING_INTERFACE_VERSION
    session_id: Identifier
    sequence_number: int = Field(ge=0, strict=True)
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(gt=0, strict=True)
    sample_count: int = Field(gt=0, strict=True)
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    captured_at: datetime
    accepted_at: datetime
    raw_audio_persisted: Literal[False] = False

    @field_validator("captured_at", "accepted_at")
    @classmethod
    def validate_timestamps(cls, value: datetime, info: object) -> datetime:
        return _require_aware(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def validate_span(self) -> "StreamChunkReceipt":
        if self.end_sample != self.start_sample + self.sample_count:
            raise ValueError("end_sample must equal start_sample plus sample_count")
        if self.accepted_at < self.captured_at:
            raise ValueError("accepted_at cannot precede captured_at")
        return self


class StreamCloseRequest(LiveStreamingRecord):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/stream-close-request.schema.json"
        },
    )

    interface_version: Literal["1.0"] = STREAMING_INTERFACE_VERSION
    session_id: Identifier
    reason: StreamCloseReason
    next_sequence_number: int = Field(ge=0, strict=True)
    next_start_sample: int = Field(ge=0, strict=True)
    requested_at: datetime

    @field_validator("requested_at")
    @classmethod
    def validate_requested_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "requested_at")


class StreamCloseReceipt(LiveStreamingRecord):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/stream-close-receipt.schema.json"
        },
    )

    interface_version: Literal["1.0"] = STREAMING_INTERFACE_VERSION
    session_id: Identifier
    reason: StreamCloseReason
    next_sequence_number: int = Field(ge=0, strict=True)
    next_start_sample: int = Field(ge=0, strict=True)
    closed_at: datetime
    notification_delivery: Literal["not_sent"] = "not_sent"
    alert_delivery_authorized: Literal[False] = False
    external_action_authorized: Literal[False] = False

    @field_validator("closed_at")
    @classmethod
    def validate_closed_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "closed_at")


def device_enrollment_sha256(enrollment: DeviceEnrollment) -> str:
    """Hash the complete canonical enrollment without inventing a device identity."""

    if not isinstance(enrollment, DeviceEnrollment):
        raise LiveStreamingInterfaceError(
            "invalid_enrollment", "Enrollment did not satisfy the live-device contract."
        )
    payload = json.dumps(
        enrollment.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def authorize_stream_session(
    enrollment: DeviceEnrollment,
    request: StreamSessionRequest,
    authentication: DeviceAuthenticationReceipt,
    *,
    now: datetime,
) -> AuthorizedStreamSession:
    """Verify enrollment, authentication, consent, scope, format, and time bounds."""

    if not isinstance(enrollment, DeviceEnrollment):
        raise LiveStreamingInterfaceError(
            "invalid_enrollment", "Enrollment did not satisfy the live-device contract."
        )
    if not isinstance(request, StreamSessionRequest):
        raise LiveStreamingInterfaceError(
            "invalid_request", "Stream request did not satisfy the live-device contract."
        )
    if not isinstance(authentication, DeviceAuthenticationReceipt):
        raise LiveStreamingInterfaceError(
            "invalid_authentication",
            "Authentication receipt did not satisfy the live-device contract.",
        )
    current = _clock(now)

    if enrollment.status is not EnrollmentStatus.ACTIVE:
        raise LiveStreamingInterfaceError(
            "inactive_enrollment", "The device enrollment is not active."
        )
    if not enrollment.enrolled_at <= current < enrollment.expires_at:
        raise LiveStreamingInterfaceError(
            "inactive_enrollment", "The device enrollment is outside its active interval."
        )
    if request.requested_at > current:
        raise LiveStreamingInterfaceError(
            "invalid_time", "The stream request time is later than the authorization time."
        )

    identity = (enrollment.enrollment_id, enrollment.device_id)
    if identity != (request.enrollment_id, request.device_id):
        raise LiveStreamingInterfaceError(
            "identity_mismatch", "The stream request does not match the enrollment."
        )
    if identity != (authentication.enrollment_id, authentication.device_id):
        raise LiveStreamingInterfaceError(
            "authentication_mismatch",
            "The authentication receipt does not match the enrollment.",
        )
    if (
        authentication.credential_fingerprint_sha256
        != enrollment.credential_fingerprint_sha256
        or authentication.session_challenge_sha256
        != request.session_challenge_sha256
    ):
        raise LiveStreamingInterfaceError(
            "authentication_mismatch",
            "The authentication receipt is not bound to this enrollment and session.",
        )
    if not (
        request.requested_at
        <= authentication.authenticated_at
        <= current
        < authentication.expires_at
    ):
        raise LiveStreamingInterfaceError(
            "authentication_expired",
            "Device authentication is missing from the current authorization interval.",
        )

    if request.requested_scope not in enrollment.authorized_scopes:
        raise LiveStreamingInterfaceError(
            "scope_not_authorized", "The enrollment does not authorize the requested scope."
        )
    if request.audio_format not in enrollment.audio_formats:
        raise LiveStreamingInterfaceError(
            "format_not_authorized", "The enrollment does not authorize the audio format."
        )

    try:
        consent = ConsentRecord.model_validate(
            request.consent.model_dump(mode="python")
        )
    except (TypeError, ValueError) as error:
        raise LiveStreamingInterfaceError(
            "invalid_consent", "Consent did not satisfy the live-processing contract."
        ) from error
    if (
        consent.status is not ConsentStatus.GRANTED
        or not consent.device_authorized
    ):
        raise LiveStreamingInterfaceError(
            "consent_not_active", "Consent is not active for live processing."
        )
    if _scope_rank(request.requested_scope) > _scope_rank(
        consent.processing_scope
    ):
        raise LiveStreamingInterfaceError(
            "scope_not_authorized", "The requested scope exceeds current consent."
        )
    if consent.granted_at is None:
        raise LiveStreamingInterfaceError(
            "consent_not_active", "Consent is not active for live processing."
        )
    try:
        granted_at = _require_aware(consent.granted_at, "consent.granted_at")
        consent_expiry = (
            _require_aware(consent.expires_at, "consent.expires_at")
            if consent.expires_at is not None
            else None
        )
    except ValueError as error:
        raise LiveStreamingInterfaceError("invalid_time", str(error)) from error
    if current < granted_at or (
        consent_expiry is not None and current >= consent_expiry
    ):
        raise LiveStreamingInterfaceError(
            "consent_not_active", "Consent is not active for live processing."
        )

    expiration_candidates = [enrollment.expires_at, authentication.expires_at]
    if consent_expiry is not None:
        expiration_candidates.append(consent_expiry)
    expires_at = min(expiration_candidates)
    if expires_at <= current:
        raise LiveStreamingInterfaceError(
            "authorization_expired", "No live authorization interval remains."
        )

    return AuthorizedStreamSession(
        session_id=request.session_id,
        enrollment_id=enrollment.enrollment_id,
        enrollment_sha256=device_enrollment_sha256(enrollment),
        authentication_id=authentication.authentication_id,
        device_id=enrollment.device_id,
        consent_id=consent.consent_id,
        processing_scope=request.requested_scope,
        audio_format=request.audio_format,
        authorized_at=current,
        expires_at=expires_at,
        raw_audio_persistence_allowed=consent.raw_audio_retention_allowed,
    )


def validate_stream_chunk(
    session: AuthorizedStreamSession,
    chunk: StreamAudioChunk,
    *,
    expected_sequence_number: int,
    expected_start_sample: int,
    now: datetime,
) -> StreamChunkReceipt:
    """Validate one contiguous bounded chunk and return audio-free receipt metadata."""

    if not isinstance(session, AuthorizedStreamSession) or not isinstance(
        chunk, StreamAudioChunk
    ):
        raise LiveStreamingInterfaceError(
            "invalid_chunk", "Stream session or chunk did not satisfy the contract."
        )
    if type(expected_sequence_number) is not int or expected_sequence_number < 0:
        raise LiveStreamingInterfaceError(
            "invalid_expectation", "Expected sequence number must be a non-negative integer."
        )
    if type(expected_start_sample) is not int or expected_start_sample < 0:
        raise LiveStreamingInterfaceError(
            "invalid_expectation", "Expected start sample must be a non-negative integer."
        )
    current = _clock(now)
    if current >= session.expires_at:
        raise LiveStreamingInterfaceError(
            "authorization_expired", "The live stream authorization has expired."
        )
    if chunk.session_id != session.session_id:
        raise LiveStreamingInterfaceError(
            "session_mismatch", "The audio chunk does not match the authorized session."
        )
    if chunk.sequence_number != expected_sequence_number:
        raise LiveStreamingInterfaceError(
            "sequence_mismatch", "The audio chunk is missing, duplicated, or out of order."
        )
    if chunk.start_sample != expected_start_sample:
        raise LiveStreamingInterfaceError(
            "sample_gap", "The audio chunk is not contiguous with accepted audio."
        )
    if not session.authorized_at <= chunk.captured_at <= current:
        raise LiveStreamingInterfaceError(
            "invalid_time", "The audio capture time is outside the accepted interval."
        )
    if chunk.sample_count > session.audio_format.max_chunk_samples:
        raise LiveStreamingInterfaceError(
            "chunk_too_large", "The audio chunk exceeds the authorized duration."
        )
    expected_bytes = (
        chunk.sample_count
        * session.audio_format.channels
        * session.audio_format.sample_width_bytes
    )
    if len(chunk.payload) != expected_bytes:
        raise LiveStreamingInterfaceError(
            "payload_size_mismatch",
            "The audio payload size does not match its declared sample count and format.",
        )

    return StreamChunkReceipt(
        session_id=session.session_id,
        sequence_number=chunk.sequence_number,
        start_sample=chunk.start_sample,
        end_sample=chunk.start_sample + chunk.sample_count,
        sample_count=chunk.sample_count,
        payload_sha256=hashlib.sha256(chunk.payload).hexdigest(),
        captured_at=chunk.captured_at,
        accepted_at=current,
    )


def validate_stream_close(
    session: AuthorizedStreamSession,
    request: StreamCloseRequest,
    *,
    expected_sequence_number: int,
    expected_start_sample: int,
    now: datetime,
) -> StreamCloseReceipt:
    """Validate a close boundary without treating closure as alert authority."""

    if not isinstance(session, AuthorizedStreamSession) or not isinstance(
        request, StreamCloseRequest
    ):
        raise LiveStreamingInterfaceError(
            "invalid_close", "Stream session or close request did not satisfy the contract."
        )
    current = _clock(now)
    if request.session_id != session.session_id:
        raise LiveStreamingInterfaceError(
            "session_mismatch", "The close request does not match the authorized session."
        )
    if (
        request.next_sequence_number != expected_sequence_number
        or request.next_start_sample != expected_start_sample
    ):
        raise LiveStreamingInterfaceError(
            "close_position_mismatch",
            "The close request does not match the accepted stream position.",
        )
    if request.requested_at > current:
        raise LiveStreamingInterfaceError(
            "invalid_time", "The close request time is later than the close time."
        )
    return StreamCloseReceipt(
        session_id=session.session_id,
        reason=request.reason,
        next_sequence_number=expected_sequence_number,
        next_start_sample=expected_start_sample,
        closed_at=current,
    )


@runtime_checkable
class DeviceEnrollmentRegistry(Protocol):
    """Storage boundary for a future enrollment implementation."""

    def register(self, enrollment: DeviceEnrollment) -> DeviceEnrollment: ...

    def get(self, enrollment_id: str) -> DeviceEnrollment | None: ...

    def revoke(
        self, enrollment_id: str, *, revoked_at: datetime, revocation_reference: str
    ) -> DeviceEnrollment: ...


@runtime_checkable
class DeviceAuthenticator(Protocol):
    """Authentication boundary; implementations must not persist credential proofs."""

    def authenticate(
        self,
        enrollment: DeviceEnrollment,
        request: StreamSessionRequest,
        credential_proof: bytes,
    ) -> DeviceAuthenticationReceipt: ...


@runtime_checkable
class StreamingAudioIngestor(Protocol):
    """Lifecycle boundary implemented by B10.1, not by this contract task."""

    def open_stream(
        self,
        request: StreamSessionRequest,
        enrollment: DeviceEnrollment,
        authentication: DeviceAuthenticationReceipt,
        *,
        now: datetime,
    ) -> AuthorizedStreamSession: ...

    def ingest_chunk(
        self, session: AuthorizedStreamSession, chunk: StreamAudioChunk, *, now: datetime
    ) -> StreamChunkReceipt: ...

    def close_stream(
        self,
        session: AuthorizedStreamSession,
        request: StreamCloseRequest,
        *,
        now: datetime,
    ) -> StreamCloseReceipt: ...


def live_streaming_schema_documents() -> dict[str, dict[str, object]]:
    """Return portable schemas for the public A10.1 records."""

    models: tuple[type[BaseModel], ...] = (
        DeviceEnrollment,
        DeviceAuthenticationReceipt,
        StreamSessionRequest,
        AuthorizedStreamSession,
        StreamAudioChunk,
        StreamChunkReceipt,
        StreamCloseRequest,
        StreamCloseReceipt,
    )
    return {
        model.model_json_schema()["$id"].rsplit("/", 1)[-1]: model.model_json_schema()
        for model in models
    }


def write_live_streaming_schemas(output_directory: Path) -> dict[str, Path]:
    """Write A10.1 schemas for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in live_streaming_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
