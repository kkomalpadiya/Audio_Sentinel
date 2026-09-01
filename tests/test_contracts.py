from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.contracts import (
    ConsentRecord,
    ConsentStatus,
    EventAnnotation,
    EventLabel,
    PreparedClipRecord,
    ProcessingScope,
    RiskLevel,
    json_schema_documents,
    write_json_schemas,
)


def active_consent(scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH) -> ConsentRecord:
    return ConsentRecord(
        consent_id="consent-001",
        status=ConsentStatus.GRANTED,
        processing_scope=scope,
        device_authorized=True,
        granted_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


def test_prepared_clip_accepts_authorized_speech_annotation() -> None:
    clip = PreparedClipRecord(
        clip_id="clip-001",
        source_dataset="custom_threat_speech",
        audio_path="custom_threat_speech/clip-001.wav",
        sample_rate_hz=16_000,
        duration_seconds=4.2,
        consent=active_consent(),
        annotations=[
            EventAnnotation(
                label=EventLabel.DISTRESS_SPEECH,
                risk_level=RiskLevel.MEDIUM,
                start_seconds=0.5,
                end_seconds=2.0,
                confidence=0.91,
            )
        ],
    )

    assert clip.schema_version == "1.0"
    assert clip.consent.permits_speech_processing


def test_acoustic_only_consent_rejects_speech_labels() -> None:
    with pytest.raises(ValidationError, match="speech event labels require acoustic_and_speech consent"):
        PreparedClipRecord(
            clip_id="clip-002",
            source_dataset="custom_threat_speech",
            audio_path="custom_threat_speech/clip-002.wav",
            sample_rate_hz=16_000,
            duration_seconds=3.0,
            consent=active_consent(ProcessingScope.ACOUSTIC_ONLY),
            annotations=[
                EventAnnotation(
                    label=EventLabel.THREATENING_SPEECH,
                    risk_level=RiskLevel.HIGH,
                    start_seconds=0.0,
                    end_seconds=1.0,
                    confidence=0.8,
                )
            ],
        )


def test_denied_consent_cannot_enable_processing() -> None:
    with pytest.raises(ValidationError, match="denied or withdrawn consent"):
        ConsentRecord(
            consent_id="consent-002",
            status=ConsentStatus.DENIED,
            processing_scope=ProcessingScope.ACOUSTIC_ONLY,
            device_authorized=False,
        )


def test_event_risk_cannot_fall_below_label_policy() -> None:
    with pytest.raises(ValidationError, match="gunshot requires at least high risk"):
        EventAnnotation(
            label=EventLabel.GUNSHOT,
            risk_level=RiskLevel.MEDIUM,
            start_seconds=0.0,
            end_seconds=0.2,
            confidence=0.8,
        )


def test_v1_taxonomy_covers_the_existing_project_label_plan() -> None:
    expected_labels = {
        EventLabel.AMBIENT,
        EventLabel.NO_SPEECH,
        EventLabel.SPEECH_PRESENT,
        EventLabel.SIREN,
        EventLabel.GLASS_BREAK,
        EventLabel.CROWD_PANIC,
        EventLabel.DISTRESS_SPEECH,
        EventLabel.THREATENING_SPEECH,
        EventLabel.WEAPON_REFERENCE,
        EventLabel.NON_THREATENING_SPEECH,
        EventLabel.GUNSHOT,
        EventLabel.EXPLOSION,
    }

    assert expected_labels <= set(EventLabel)


def test_prepared_clip_rejects_path_traversal_and_out_of_bounds_annotations() -> None:
    with pytest.raises(ValidationError, match="relative path"):
        PreparedClipRecord(
            clip_id="clip-003",
            source_dataset="fsd50k",
            audio_path="../outside.wav",
            sample_rate_hz=16_000,
            duration_seconds=1.0,
            consent=active_consent(),
        )

    with pytest.raises(ValidationError, match="cannot exceed duration_seconds"):
        PreparedClipRecord(
            clip_id="clip-004",
            source_dataset="fsd50k",
            audio_path="fsd50k/clip-004.wav",
            sample_rate_hz=16_000,
            duration_seconds=1.0,
            consent=active_consent(),
            annotations=[
                EventAnnotation(
                    label=EventLabel.SIREN,
                    risk_level=RiskLevel.LOW,
                    start_seconds=0.5,
                    end_seconds=1.1,
                    confidence=0.7,
                )
            ],
        )


def test_json_schema_export_is_complete_and_portable(tmp_path) -> None:
    documents = json_schema_documents()
    exported = write_json_schemas(tmp_path)

    assert set(documents) == {
        "consent-record.schema.json",
        "event-annotation.schema.json",
        "prepared-clip.schema.json",
    }
    for filename, path in exported.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["$id"].endswith(filename)
        assert document == documents[filename]


def test_checked_in_schemas_match_the_public_contracts() -> None:
    schema_directory = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "v1"

    for filename, expected_document in json_schema_documents().items():
        document = json.loads((schema_directory / filename).read_text(encoding="utf-8"))
        assert document == expected_document
