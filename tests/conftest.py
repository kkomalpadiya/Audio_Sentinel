from datetime import UTC, datetime
from pathlib import Path

import pytest

from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import ConsentRecord, ConsentStatus, ProcessingScope


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def temporary_settings(tmp_path: Path) -> AudioSentinelSettings:
    return AudioSentinelSettings.from_project_root(tmp_path / "audio-sentinel-test")


@pytest.fixture
def active_consent() -> ConsentRecord:
    return ConsentRecord(
        consent_id="test-consent-001",
        status=ConsentStatus.GRANTED,
        processing_scope=ProcessingScope.ACOUSTIC_AND_SPEECH,
        device_authorized=True,
        granted_at=datetime(2026, 9, 5, tzinfo=UTC),
    )
