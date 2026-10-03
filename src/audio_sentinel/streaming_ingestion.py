"""B10.1 local enrollment, authentication, and streaming-ingestion service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
from tempfile import mkdtemp
from threading import RLock
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.config import Paths
from audio_sentinel.contracts import ConsentRecord, ProcessingScope
from audio_sentinel.live_streaming import (
    AuthorizedStreamSession,
    DeviceAuthenticationReceipt,
    DeviceAuthenticator,
    DeviceEnrollment,
    DeviceEnrollmentRegistry,
    EnrollmentStatus,
    LiveStreamingInterfaceError,
    StreamAudioChunk,
    StreamAudioFormat,
    StreamChunkReceipt,
    StreamCloseReason,
    StreamCloseReceipt,
    StreamCloseRequest,
    StreamSessionRequest,
    StreamingAudioIngestor,
    authorize_stream_session,
    device_enrollment_sha256,
    validate_stream_chunk,
    validate_stream_close,
)
from audio_sentinel.persistence import _is_link


LIVE_ENROLLMENT_DIRECTORY = "live-device-enrollments"
ENROLLMENT_FILENAME = "enrollment.json"
REVOCATION_FILENAME = "revoked.json"
DEFAULT_MAX_ENROLLMENT_BYTES = 131_072
DEFAULT_CHALLENGE_BYTES = 32
DEFAULT_CHALLENGE_TTL_SECONDS = 30
DEFAULT_AUTHENTICATION_TTL_SECONDS = 30
DEFAULT_MAX_PENDING_CHALLENGES = 128
DEFAULT_MAX_ACTIVE_SESSIONS = 32
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


class StreamingIngestionError(RuntimeError):
    """Stable B10.1 failure code with a safe, non-sensitive explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class IngestionRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


def _require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise StreamingIngestionError(
            "invalid_time", f"{name} must include a valid UTC offset."
        )
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise StreamingIngestionError(
            "invalid_time", f"{name} must include a valid UTC offset."
        ) from error


def _now(clock: Callable[[], datetime]) -> datetime:
    try:
        value = clock()
    except Exception as error:
        raise StreamingIngestionError(
            "clock_failed", "The local service clock was unavailable."
        ) from error
    if not isinstance(value, datetime):
        raise StreamingIngestionError(
            "clock_failed", "The local service clock returned an invalid value."
        )
    return _require_aware(value, "service time")


class IssuedStreamChallenge(IngestionRecord):
    """Short-lived one-time challenge returned to the local device client."""

    challenge_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    enrollment_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    device_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    challenge: bytes = Field(
        min_length=DEFAULT_CHALLENGE_BYTES,
        max_length=DEFAULT_CHALLENGE_BYTES,
    )
    challenge_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    issued_at: datetime
    expires_at: datetime

    @field_validator("issued_at", "expires_at")
    @classmethod
    def validate_time(cls, value: datetime, info: object) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{getattr(info, 'field_name', 'timestamp')} must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_challenge(self) -> "IssuedStreamChallenge":
        if self.expires_at <= self.issued_at:
            raise ValueError("challenge expiry must follow issue time")
        if hashlib.sha256(self.challenge).hexdigest() != self.challenge_sha256:
            raise ValueError("challenge digest does not match challenge bytes")
        return self


@runtime_checkable
class StreamingAudioSink(Protocol):
    """Synchronous handoff boundary; the ingestion service retains no audio."""

    def accept_chunk(
        self,
        session: AuthorizedStreamSession,
        chunk: StreamAudioChunk,
        receipt: StreamChunkReceipt,
    ) -> None: ...

    def close_stream(
        self,
        session: AuthorizedStreamSession,
        receipt: StreamCloseReceipt,
    ) -> None: ...


def _managed_parent(paths: Paths, *, create: bool) -> Path:
    try:
        root = paths.root.resolve(strict=True)
        processed = Path(os.path.abspath(paths.processed_data))
        parts = processed.relative_to(root).parts
        if not parts:
            raise ValueError("processed data cannot be the project root")
        for other in (paths.raw_data, paths.interim_data, paths.models):
            other_path = Path(os.path.abspath(other))
            if processed.is_relative_to(other_path) or other_path.is_relative_to(
                processed
            ):
                raise ValueError("managed directories overlap")
        current = root
        for part in (*parts, LIVE_ENROLLMENT_DIRECTORY):
            current = current / part
            if _is_link(current):
                raise ValueError("linked registry component")
            if create:
                current.mkdir(exist_ok=True)
            elif not current.exists():
                return processed / LIVE_ENROLLMENT_DIRECTORY
            if current.exists() and not current.is_dir():
                raise ValueError("registry component is not a directory")
        return current.resolve(strict=True) if current.exists() else current
    except (OSError, RuntimeError, ValueError) as error:
        raise StreamingIngestionError(
            "invalid_registry_path",
            "Device enrollments must remain in unlinked managed processed data.",
        ) from error


def _validate_identifier(value: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise StreamingIngestionError(
            "invalid_enrollment_id", "Enrollment ID did not satisfy the contract."
        )
    return value


def _read_document(path: Path, max_bytes: int) -> bytes:
    try:
        if _is_link(path):
            raise ValueError("linked document")
        resolved = path.resolve(strict=True)
        if resolved != path or not path.is_file():
            raise ValueError("noncanonical document")
        size = path.stat().st_size
        if size <= 0 or size > max_bytes:
            raise ValueError("document size outside bounds")
        with path.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        if len(payload) != size or len(payload) > max_bytes:
            raise ValueError("document changed or exceeded bounds")
        return payload
    except (OSError, RuntimeError, ValueError) as error:
        raise StreamingIngestionError(
            "invalid_registry_record",
            "The device enrollment record could not be read or verified.",
        ) from error


def _document_bytes(record: DeviceEnrollment) -> bytes:
    return (record.model_dump_json(indent=2) + "\n").encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise
    except OSError as error:
        raise StreamingIngestionError(
            "registry_write_failed", "The device enrollment could not be saved."
        ) from error


def _remove_stage(stage: Path, parent: Path) -> None:
    try:
        if stage.parent != parent or not stage.name.startswith(".pending-"):
            return
        if _is_link(stage) or not stage.is_dir():
            return
        for item in stage.iterdir():
            if _is_link(item) or not item.is_file() or item.parent != stage:
                return
        for item in stage.iterdir():
            item.unlink()
        stage.rmdir()
    except OSError:
        return


def _same_base_enrollment(
    initial: DeviceEnrollment, revoked: DeviceEnrollment
) -> bool:
    excluded = {"status", "revoked_at", "revocation_reference"}
    return initial.model_dump(exclude=excluded) == revoked.model_dump(exclude=excluded)


class FileDeviceEnrollmentRegistry(DeviceEnrollmentRegistry):
    """Append-only local enrollment registry under managed processed data."""

    def __init__(
        self,
        paths: Paths,
        *,
        max_document_bytes: int = DEFAULT_MAX_ENROLLMENT_BYTES,
    ) -> None:
        if type(max_document_bytes) is not int or max_document_bytes <= 0:
            raise ValueError("max_document_bytes must be a positive integer")
        self._paths = paths
        self._max_document_bytes = max_document_bytes
        self._lock = RLock()

    def _directory(self, enrollment_id: str, *, create_parent: bool) -> Path:
        identifier = _validate_identifier(enrollment_id)
        parent = _managed_parent(self._paths, create=create_parent)
        directory = parent / identifier
        if _is_link(directory):
            raise StreamingIngestionError(
                "invalid_registry_record",
                "The device enrollment record could not be read or verified.",
            )
        return directory

    def _load_directory(self, directory: Path) -> DeviceEnrollment:
        try:
            if _is_link(directory) or directory.resolve(strict=True) != directory:
                raise ValueError("noncanonical directory")
            if not directory.is_dir():
                raise ValueError("not a directory")
            inventory = {item.name for item in directory.iterdir()}
            if inventory not in (
                {ENROLLMENT_FILENAME},
                {ENROLLMENT_FILENAME, REVOCATION_FILENAME},
            ):
                raise ValueError("unexpected registry inventory")
            initial = DeviceEnrollment.model_validate_json(
                _read_document(
                    directory / ENROLLMENT_FILENAME, self._max_document_bytes
                )
            )
            if initial.enrollment_id != directory.name:
                raise ValueError("identity/path mismatch")
            if initial.status is not EnrollmentStatus.ACTIVE:
                raise ValueError("initial enrollment must be active")
            if REVOCATION_FILENAME not in inventory:
                return initial
            revoked = DeviceEnrollment.model_validate_json(
                _read_document(
                    directory / REVOCATION_FILENAME, self._max_document_bytes
                )
            )
            if (
                revoked.enrollment_id != directory.name
                or revoked.status is not EnrollmentStatus.REVOKED
                or not _same_base_enrollment(initial, revoked)
            ):
                raise ValueError("revocation does not match initial enrollment")
            return revoked
        except StreamingIngestionError:
            raise
        except Exception as error:
            raise StreamingIngestionError(
                "invalid_registry_record",
                "The device enrollment record could not be read or verified.",
            ) from error

    def register(self, enrollment: DeviceEnrollment) -> DeviceEnrollment:
        if not isinstance(enrollment, DeviceEnrollment):
            raise StreamingIngestionError(
                "invalid_enrollment", "Enrollment did not satisfy the contract."
            )
        try:
            verified = DeviceEnrollment.model_validate(
                enrollment.model_dump(mode="python")
            )
        except Exception as error:
            raise StreamingIngestionError(
                "invalid_enrollment", "Enrollment did not satisfy the contract."
            ) from error
        if verified.status is not EnrollmentStatus.ACTIVE:
            raise StreamingIngestionError(
                "invalid_enrollment", "New enrollment must start active."
            )
        payload = _document_bytes(verified)
        if len(payload) > self._max_document_bytes:
            raise StreamingIngestionError(
                "enrollment_too_large", "Enrollment exceeds the configured size limit."
            )

        with self._lock:
            directory = self._directory(
                verified.enrollment_id, create_parent=True
            )
            if directory.exists():
                existing = self._load_directory(directory)
                if existing != verified:
                    raise StreamingIngestionError(
                        "enrollment_conflict",
                        "An enrollment with this identity already differs.",
                    )
                return existing

            parent = directory.parent
            stage = Path(mkdtemp(prefix=".pending-", dir=parent)).resolve()
            try:
                staged = stage / ENROLLMENT_FILENAME
                _write_exclusive(staged, payload)
                if DeviceEnrollment.model_validate_json(
                    _read_document(staged, self._max_document_bytes)
                ) != verified:
                    raise StreamingIngestionError(
                        "registry_write_failed",
                        "The device enrollment failed readback verification.",
                    )
                try:
                    stage.rename(directory)
                    stage = None  # type: ignore[assignment]
                except OSError as error:
                    if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) and not isinstance(
                        error, FileExistsError
                    ):
                        raise
                    existing = self._load_directory(directory)
                    if existing != verified:
                        raise StreamingIngestionError(
                            "enrollment_conflict",
                            "An enrollment with this identity already differs.",
                        )
                    return existing
                loaded = self._load_directory(directory)
                if loaded != verified:
                    raise StreamingIngestionError(
                        "registry_write_failed",
                        "The persisted enrollment failed verification.",
                    )
                return loaded
            finally:
                if stage is not None:
                    _remove_stage(stage, parent)

    def get(self, enrollment_id: str) -> DeviceEnrollment | None:
        with self._lock:
            directory = self._directory(enrollment_id, create_parent=False)
            if not directory.exists():
                return None
            return self._load_directory(directory)

    def revoke(
        self,
        enrollment_id: str,
        *,
        revoked_at: datetime,
        revocation_reference: str,
    ) -> DeviceEnrollment:
        revoked_time = _require_aware(revoked_at, "revoked_at")
        with self._lock:
            existing = self.get(enrollment_id)
            if existing is None:
                raise StreamingIngestionError(
                    "enrollment_not_found", "The enrollment was not found."
                )
            if existing.status is EnrollmentStatus.REVOKED:
                if (
                    existing.revoked_at == revoked_time
                    and existing.revocation_reference == revocation_reference
                ):
                    return existing
                raise StreamingIngestionError(
                    "revocation_conflict",
                    "The enrollment already has a different revocation record.",
                )
            try:
                revoked = DeviceEnrollment.model_validate(
                    {
                        **existing.model_dump(mode="python"),
                        "status": EnrollmentStatus.REVOKED,
                        "revoked_at": revoked_time,
                        "revocation_reference": revocation_reference,
                    }
                )
            except Exception as error:
                raise StreamingIngestionError(
                    "invalid_revocation", "The revocation did not satisfy the contract."
                ) from error
            payload = _document_bytes(revoked)
            if len(payload) > self._max_document_bytes:
                raise StreamingIngestionError(
                    "enrollment_too_large", "Revocation exceeds the configured size limit."
                )
            path = self._directory(enrollment_id, create_parent=False) / REVOCATION_FILENAME
            try:
                _write_exclusive(path, payload)
            except FileExistsError:
                loaded = self._load_directory(path.parent)
                if loaded != revoked:
                    raise StreamingIngestionError(
                        "revocation_conflict",
                        "The enrollment already has a different revocation record.",
                    )
                return loaded
            loaded = self._load_directory(path.parent)
            if loaded != revoked:
                raise StreamingIngestionError(
                    "registry_write_failed", "The persisted revocation failed verification."
                )
            return loaded


class LocalDeviceAuthenticator(DeviceAuthenticator):
    """One-time HMAC-SHA256 challenge verifier using an external secret resolver."""

    def __init__(
        self,
        credential_resolver: Callable[[DeviceEnrollment], bytes],
        *,
        verifier_id: str = "local-hmac-sha256-v1",
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        challenge_ttl_seconds: int = DEFAULT_CHALLENGE_TTL_SECONDS,
        authentication_ttl_seconds: int = DEFAULT_AUTHENTICATION_TTL_SECONDS,
        max_pending_challenges: int = DEFAULT_MAX_PENDING_CHALLENGES,
    ) -> None:
        if _IDENTIFIER.fullmatch(verifier_id) is None:
            raise ValueError("verifier_id must be a valid identifier")
        for name, value in (
            ("challenge_ttl_seconds", challenge_ttl_seconds),
            ("authentication_ttl_seconds", authentication_ttl_seconds),
            ("max_pending_challenges", max_pending_challenges),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._credential_resolver = credential_resolver
        self._verifier_id = verifier_id
        self._clock = clock
        self._challenge_ttl = timedelta(seconds=challenge_ttl_seconds)
        self._authentication_ttl = timedelta(seconds=authentication_ttl_seconds)
        self._max_pending_challenges = max_pending_challenges
        self._challenges: dict[str, IssuedStreamChallenge] = {}
        self._receipts: dict[str, DeviceAuthenticationReceipt] = {}
        self._lock = RLock()

    def _cleanup(self, current: datetime) -> None:
        self._challenges = {
            digest: challenge
            for digest, challenge in self._challenges.items()
            if challenge.expires_at > current
        }
        self._receipts = {
            identity: receipt
            for identity, receipt in self._receipts.items()
            if receipt.expires_at > current
        }

    def issue_challenge(
        self, enrollment: DeviceEnrollment, *, now: datetime | None = None
    ) -> IssuedStreamChallenge:
        if not isinstance(enrollment, DeviceEnrollment):
            raise StreamingIngestionError(
                "invalid_enrollment", "Enrollment did not satisfy the contract."
            )
        current = _require_aware(now, "challenge time") if now is not None else _now(self._clock)
        if (
            enrollment.status is not EnrollmentStatus.ACTIVE
            or not enrollment.enrolled_at <= current < enrollment.expires_at
        ):
            raise StreamingIngestionError(
                "inactive_enrollment", "The device enrollment is not active."
            )
        with self._lock:
            self._cleanup(current)
            if len(self._challenges) >= self._max_pending_challenges:
                raise StreamingIngestionError(
                    "challenge_capacity_reached",
                    "The authenticator cannot issue another challenge yet.",
                )
            while True:
                challenge_bytes = secrets.token_bytes(DEFAULT_CHALLENGE_BYTES)
                digest = hashlib.sha256(challenge_bytes).hexdigest()
                if digest not in self._challenges:
                    break
            challenge = IssuedStreamChallenge(
                challenge_id=f"challenge-{secrets.token_hex(16)}",
                enrollment_id=enrollment.enrollment_id,
                device_id=enrollment.device_id,
                challenge=challenge_bytes,
                challenge_sha256=digest,
                issued_at=current,
                expires_at=min(current + self._challenge_ttl, enrollment.expires_at),
            )
            self._challenges[digest] = challenge
            return challenge

    def authenticate(
        self,
        enrollment: DeviceEnrollment,
        request: StreamSessionRequest,
        credential_proof: bytes,
    ) -> DeviceAuthenticationReceipt:
        if not isinstance(enrollment, DeviceEnrollment) or not isinstance(
            request, StreamSessionRequest
        ):
            raise StreamingIngestionError(
                "invalid_authentication_request",
                "Authentication inputs did not satisfy the contract.",
            )
        if not isinstance(credential_proof, bytes) or len(credential_proof) != 32:
            raise StreamingIngestionError(
                "authentication_failed", "Device authentication failed."
            )
        current = _now(self._clock)
        with self._lock:
            self._cleanup(current)
            challenge = self._challenges.pop(
                request.session_challenge_sha256, None
            )
            if challenge is None or challenge.expires_at <= current:
                raise StreamingIngestionError(
                    "challenge_invalid", "The session challenge is missing or expired."
                )
            if (
                challenge.enrollment_id != enrollment.enrollment_id
                or challenge.device_id != enrollment.device_id
                or request.enrollment_id != enrollment.enrollment_id
                or request.device_id != enrollment.device_id
            ):
                raise StreamingIngestionError(
                    "challenge_mismatch", "The session challenge does not match the device."
                )

            secret_value: bytearray | None = None
            try:
                resolved = self._credential_resolver(enrollment)
                if not isinstance(resolved, bytes) or len(resolved) < 32:
                    raise ValueError("invalid resolved secret")
                secret_value = bytearray(resolved)
                fingerprint = hashlib.sha256(secret_value).hexdigest()
                expected = hmac.new(
                    secret_value, challenge.challenge, hashlib.sha256
                ).digest()
            except Exception as error:
                raise StreamingIngestionError(
                    "authentication_failed", "Device authentication failed."
                ) from error
            finally:
                if secret_value is not None:
                    secret_value[:] = b"\x00" * len(secret_value)
            if not hmac.compare_digest(
                fingerprint, enrollment.credential_fingerprint_sha256
            ) or not hmac.compare_digest(expected, credential_proof):
                raise StreamingIngestionError(
                    "authentication_failed", "Device authentication failed."
                )

            expires_at = min(
                current + self._authentication_ttl,
                enrollment.expires_at,
            )
            if expires_at <= current:
                raise StreamingIngestionError(
                    "inactive_enrollment", "The device enrollment is not active."
                )
            receipt = DeviceAuthenticationReceipt(
                authentication_id=f"authentication-{secrets.token_hex(16)}",
                verifier_id=self._verifier_id,
                enrollment_id=enrollment.enrollment_id,
                device_id=enrollment.device_id,
                credential_fingerprint_sha256=enrollment.credential_fingerprint_sha256,
                session_challenge_sha256=challenge.challenge_sha256,
                authenticated_at=current,
                expires_at=expires_at,
            )
            self._receipts[receipt.authentication_id] = receipt
            return receipt

    def claim_receipt(
        self, receipt: DeviceAuthenticationReceipt, *, now: datetime
    ) -> None:
        current = _require_aware(now, "receipt claim time")
        if not isinstance(receipt, DeviceAuthenticationReceipt):
            raise StreamingIngestionError(
                "authentication_invalid", "Authentication receipt was invalid."
            )
        with self._lock:
            self._cleanup(current)
            issued = self._receipts.pop(receipt.authentication_id, None)
            if issued is None or issued != receipt or receipt.expires_at <= current:
                raise StreamingIngestionError(
                    "authentication_invalid",
                    "Authentication receipt was invalid or already used.",
                )


@dataclass
class _SessionState:
    session: AuthorizedStreamSession
    next_sequence_number: int = 0
    next_start_sample: int = 0


class LocalStreamingIngestionService(StreamingAudioIngestor):
    """Thread-safe local session service with synchronous, non-retaining handoff."""

    def __init__(
        self,
        registry: FileDeviceEnrollmentRegistry,
        authenticator: LocalDeviceAuthenticator,
        sink: StreamingAudioSink,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_active_sessions: int = DEFAULT_MAX_ACTIVE_SESSIONS,
    ) -> None:
        if not isinstance(sink, StreamingAudioSink):
            raise TypeError("sink must satisfy StreamingAudioSink")
        if type(max_active_sessions) is not int or max_active_sessions <= 0:
            raise ValueError("max_active_sessions must be a positive integer")
        self._registry = registry
        self._authenticator = authenticator
        self._sink = sink
        self._clock = clock
        self._max_active_sessions = max_active_sessions
        self._sessions: dict[str, _SessionState] = {}
        self._lock = RLock()

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def issue_challenge(
        self, enrollment_id: str, device_id: str
    ) -> IssuedStreamChallenge:
        enrollment = self._registry.get(enrollment_id)
        if enrollment is None or enrollment.device_id != device_id:
            raise StreamingIngestionError(
                "enrollment_unavailable",
                "An active matching enrollment was not available.",
            )
        return self._authenticator.issue_challenge(
            enrollment, now=_now(self._clock)
        )

    def open_authenticated_stream(
        self,
        request: StreamSessionRequest,
        credential_proof: bytes,
    ) -> AuthorizedStreamSession:
        enrollment = self._registry.get(request.enrollment_id)
        if enrollment is None:
            raise StreamingIngestionError(
                "enrollment_unavailable",
                "An active matching enrollment was not available.",
            )
        authentication = self._authenticator.authenticate(
            enrollment, request, credential_proof
        )
        return self.open_stream(
            request,
            enrollment,
            authentication,
            now=_now(self._clock),
        )

    def open_stream(
        self,
        request: StreamSessionRequest,
        enrollment: DeviceEnrollment,
        authentication: DeviceAuthenticationReceipt,
        *,
        now: datetime,
    ) -> AuthorizedStreamSession:
        current = _require_aware(now, "stream open time")
        self._authenticator.claim_receipt(authentication, now=current)
        current_enrollment = self._registry.get(enrollment.enrollment_id)
        if current_enrollment is None or current_enrollment != enrollment:
            raise StreamingIngestionError(
                "enrollment_changed",
                "The enrollment changed before the stream could open.",
            )
        try:
            session = authorize_stream_session(
                current_enrollment, request, authentication, now=current
            )
        except LiveStreamingInterfaceError as error:
            raise StreamingIngestionError(error.code, str(error)) from error
        with self._lock:
            if session.session_id in self._sessions:
                raise StreamingIngestionError(
                    "session_conflict", "A stream with this identity is already active."
                )
            if len(self._sessions) >= self._max_active_sessions:
                raise StreamingIngestionError(
                    "session_capacity_reached",
                    "The service cannot open another stream yet.",
                )
            self._sessions[session.session_id] = _SessionState(session=session)
        return session

    def accept_chunk(
        self, session: AuthorizedStreamSession, chunk: StreamAudioChunk
    ) -> StreamChunkReceipt:
        return self.ingest_chunk(session, chunk, now=_now(self._clock))

    def ingest_chunk(
        self,
        session: AuthorizedStreamSession,
        chunk: StreamAudioChunk,
        *,
        now: datetime,
    ) -> StreamChunkReceipt:
        current = _require_aware(now, "chunk acceptance time")
        with self._lock:
            state = self._sessions.get(session.session_id)
            if state is None:
                raise StreamingIngestionError(
                    "session_not_active", "The stream session is not active."
                )
            if state.session != session:
                raise StreamingIngestionError(
                    "session_mismatch", "The supplied session does not match active state."
                )
            enrollment = self._registry.get(session.enrollment_id)
            if (
                enrollment is None
                or enrollment.status is not EnrollmentStatus.ACTIVE
                or device_enrollment_sha256(enrollment) != session.enrollment_sha256
            ):
                del self._sessions[session.session_id]
                raise StreamingIngestionError(
                    "enrollment_inactive",
                    "The enrollment is no longer active for this stream.",
                )
            try:
                receipt = validate_stream_chunk(
                    session,
                    chunk,
                    expected_sequence_number=state.next_sequence_number,
                    expected_start_sample=state.next_start_sample,
                    now=current,
                )
            except LiveStreamingInterfaceError as error:
                raise StreamingIngestionError(error.code, str(error)) from error
            try:
                self._sink.accept_chunk(session, chunk, receipt)
            except Exception as error:
                raise StreamingIngestionError(
                    "sink_failed", "The accepted audio chunk could not be handed off."
                ) from error
            state.next_sequence_number += 1
            state.next_start_sample = receipt.end_sample
            return receipt

    def finish_stream(
        self,
        session: AuthorizedStreamSession,
        request: StreamCloseRequest,
    ) -> StreamCloseReceipt:
        return self.close_stream(session, request, now=_now(self._clock))

    def close_stream(
        self,
        session: AuthorizedStreamSession,
        request: StreamCloseRequest,
        *,
        now: datetime,
    ) -> StreamCloseReceipt:
        current = _require_aware(now, "stream close time")
        with self._lock:
            state = self._sessions.get(session.session_id)
            if state is None:
                raise StreamingIngestionError(
                    "session_not_active", "The stream session is not active."
                )
            if state.session != session:
                raise StreamingIngestionError(
                    "session_mismatch", "The supplied session does not match active state."
                )
            try:
                receipt = validate_stream_close(
                    session,
                    request,
                    expected_sequence_number=state.next_sequence_number,
                    expected_start_sample=state.next_start_sample,
                    now=current,
                )
            except LiveStreamingInterfaceError as error:
                raise StreamingIngestionError(error.code, str(error)) from error
            try:
                self._sink.close_stream(session, receipt)
            except Exception as error:
                raise StreamingIngestionError(
                    "sink_failed", "The stream close could not be handed off."
                ) from error
            del self._sessions[session.session_id]
            return receipt


class LocalStreamingClient:
    """Stateful in-process client that builds correct requests and chunk positions."""

    def __init__(
        self,
        service: LocalStreamingIngestionService,
        *,
        enrollment_id: str,
        device_id: str,
        credential_provider: Callable[[], bytes],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        session_id_factory: Callable[[], str] = lambda: f"session-{secrets.token_hex(16)}",
    ) -> None:
        self._service = service
        self._enrollment_id = _validate_identifier(enrollment_id)
        self._device_id = _validate_identifier(device_id)
        self._credential_provider = credential_provider
        self._clock = clock
        self._session_id_factory = session_id_factory
        self._session: AuthorizedStreamSession | None = None
        self._next_sequence_number = 0
        self._next_start_sample = 0
        self._lock = RLock()

    @property
    def session(self) -> AuthorizedStreamSession | None:
        with self._lock:
            return self._session

    def open(
        self,
        *,
        consent: ConsentRecord,
        requested_scope: ProcessingScope,
        audio_format: StreamAudioFormat,
    ) -> AuthorizedStreamSession:
        with self._lock:
            if self._session is not None:
                raise StreamingIngestionError(
                    "client_session_active", "The client already has an active stream."
                )
            current = _now(self._clock)
            challenge = self._service.issue_challenge(
                self._enrollment_id, self._device_id
            )
            request = StreamSessionRequest(
                session_id=self._session_id_factory(),
                enrollment_id=self._enrollment_id,
                device_id=self._device_id,
                session_challenge_sha256=challenge.challenge_sha256,
                requested_scope=requested_scope,
                audio_format=audio_format,
                consent=consent,
                requested_at=current,
            )
            secret: bytearray | None = None
            try:
                supplied = self._credential_provider()
                if not isinstance(supplied, bytes) or len(supplied) < 32:
                    raise ValueError("invalid credential")
                secret = bytearray(supplied)
                proof = hmac.new(
                    secret, challenge.challenge, hashlib.sha256
                ).digest()
            except Exception as error:
                raise StreamingIngestionError(
                    "credential_unavailable",
                    "The local device credential was unavailable.",
                ) from error
            finally:
                if secret is not None:
                    secret[:] = b"\x00" * len(secret)
            session = self._service.open_authenticated_stream(request, proof)
            self._session = session
            self._next_sequence_number = 0
            self._next_start_sample = 0
            return session

    def send_pcm(
        self,
        payload: bytes,
        *,
        captured_at: datetime | None = None,
    ) -> StreamChunkReceipt:
        with self._lock:
            session = self._session
            if session is None:
                raise StreamingIngestionError(
                    "client_session_missing", "The client has no active stream."
                )
            if not isinstance(payload, bytes) or not payload:
                raise StreamingIngestionError(
                    "invalid_payload", "PCM payload must be nonempty bytes."
                )
            bytes_per_sample_frame = (
                session.audio_format.channels
                * session.audio_format.sample_width_bytes
            )
            if len(payload) % bytes_per_sample_frame:
                raise StreamingIngestionError(
                    "invalid_payload", "PCM payload is not aligned to the audio format."
                )
            sample_count = len(payload) // bytes_per_sample_frame
            try:
                chunk = StreamAudioChunk(
                    session_id=session.session_id,
                    sequence_number=self._next_sequence_number,
                    start_sample=self._next_start_sample,
                    sample_count=sample_count,
                    captured_at=(
                        _require_aware(captured_at, "capture time")
                        if captured_at is not None
                        else _now(self._clock)
                    ),
                    payload=payload,
                )
            except StreamingIngestionError:
                raise
            except Exception as error:
                raise StreamingIngestionError(
                    "invalid_payload", "PCM payload did not satisfy the stream contract."
                ) from error
            receipt = self._service.accept_chunk(session, chunk)
            self._next_sequence_number += 1
            self._next_start_sample = receipt.end_sample
            return receipt

    def close(
        self,
        reason: StreamCloseReason = StreamCloseReason.CLIENT_REQUEST,
    ) -> StreamCloseReceipt:
        with self._lock:
            session = self._session
            if session is None:
                raise StreamingIngestionError(
                    "client_session_missing", "The client has no active stream."
                )
            requested_at = _now(self._clock)
            request = StreamCloseRequest(
                session_id=session.session_id,
                reason=reason,
                next_sequence_number=self._next_sequence_number,
                next_start_sample=self._next_start_sample,
                requested_at=requested_at,
            )
            receipt = self._service.finish_stream(session, request)
            self._session = None
            self._next_sequence_number = 0
            self._next_start_sample = 0
            return receipt
