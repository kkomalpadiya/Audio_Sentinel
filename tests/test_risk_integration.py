"""A6.2 tests for evidence-to-risk-input integration."""

from datetime import UTC, datetime
import copy
import hashlib
import json
from pathlib import Path

import pytest

from audio_sentinel.acoustic_aggregation import AcousticAggregationSettings
from audio_sentinel.acoustic_evidence import (
    AcousticEvidenceDocument,
    EvidenceEvent,
    EvidenceModel,
    EvidenceSource,
    EvidenceWindow,
)
from audio_sentinel.acoustic_inference import expected_yamnet_patches
from audio_sentinel.acoustic_model import LABEL_MAPPING, YAMNET
from audio_sentinel.contracts import EventLabel, ProcessingScope
from audio_sentinel.language_contracts import LanguageEvidenceDocument
from audio_sentinel.risk_contracts import RiskInputStatus, RiskSource
from audio_sentinel.risk_integration import RiskIntegrationError, integrate_risk_inputs
from audio_sentinel.risk_scoring import score_risk
from audio_sentinel.speech_contracts import SpeechEvidenceDocument


NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@pytest.fixture
def speech_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (project_root / "docs" / "examples" / "speech-evidence.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def speech(speech_data: dict[str, object]) -> SpeechEvidenceDocument:
    return SpeechEvidenceDocument.model_validate(speech_data)


@pytest.fixture
def source(speech: SpeechEvidenceDocument) -> RiskSource:
    return RiskSource(
        clip_id=speech.source.clip_id,
        consent_id=speech.source.consent_id,
        processing_scope=speech.source.processing_scope,
        sample_rate_hz=speech.source.sample_rate_hz,
        num_samples=speech.source.num_samples,
    )


@pytest.fixture
def language_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (project_root / "docs" / "examples" / "language-evidence.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def language(
    language_data: dict[str, object], speech: SpeechEvidenceDocument
) -> LanguageEvidenceDocument:
    language_data["source"]["speech_evidence_id"] = speech.evidence_id
    language_data["source"]["speech_evidence_sha256"] = canonical_hash(
        speech.model_dump(mode="json")
    )
    language_data["input_transcript_count"] = 1
    language_data["analyzed_transcript_count"] = 1
    language_data["finding_count"] = 1
    language_data["analyses"] = language_data["analyses"][:1]
    return LanguageEvidenceDocument.model_validate(language_data)


def acoustic_document(
    speech: SpeechEvidenceDocument,
    *,
    manifest_sha256: str | None = None,
) -> AcousticEvidenceDocument:
    winner = LABEL_MAPPING[EventLabel.EXPLOSION][0]
    num_samples = speech.source.num_samples
    patch_count = expected_yamnet_patches(num_samples)
    end_sample = min(YAMNET.patch_support_samples, num_samples)
    payload = {
        "schema_version": "1.0",
        "document_type": "acoustic_event_candidates",
        "sample_rate_hz": YAMNET.sample_rate_hz,
        "source": {
            "preparation_manifest_path": speech.source.preparation_manifest_path,
            "preparation_manifest_sha256": (
                speech.source.preparation_manifest_sha256
                if manifest_sha256 is None
                else manifest_sha256
            ),
            "raw_audio_sha256": speech.source.raw_audio_sha256,
            "clip_id": speech.source.clip_id,
        },
        "model": {
            "model_id": YAMNET.model_id,
            "model_version": "1",
            "model_handle": YAMNET.model_handle,
            "model_path": "yamnet/1",
            "artifact_sha256": YAMNET.artifact_sha256,
            "class_map_sha256": YAMNET.class_map_sha256,
            "label_mapping_version": "1.0",
            "runtime_distribution": YAMNET.runtime_distribution,
            "runtime_version": YAMNET.runtime_version,
            "exported_with_tensorflow": "2.3.0",
            "exported_with_tensorflow_git": "test",
            "signature_name": "serving_default",
            "input": {"name": "waveform", "shape": [None], "dtype": "float32"},
            "outputs": [
                {"name": "output_0", "shape": [None, 521], "dtype": "float32"},
                {"name": "output_1", "shape": [None, 1024], "dtype": "float32"},
                {"name": "output_2", "shape": [None, 64], "dtype": "float32"},
            ],
            "num_classes": 521,
        },
        "aggregation": AcousticAggregationSettings.uniform(0.5).model_dump(
            mode="json"
        ),
        "input_window_count": 1,
        "input_patch_count": patch_count,
        "windows": [
            {
                "window_id": "window-risk-001",
                "audio_path": "prepared/example/window-risk-001.wav",
                "window_audio_sha256": "d" * 64,
                "window_seconds": num_samples / YAMNET.sample_rate_hz,
                "start_sample": 0,
                "end_sample": num_samples,
                "padding_samples": 0,
                "input_num_samples": num_samples,
                "patch_count": patch_count,
            }
        ],
        "events": [
            {
                "label": "explosion",
                "start_sample": 0,
                "end_sample": end_sample,
                "start_seconds": 0,
                "end_seconds": end_sample / YAMNET.sample_rate_hz,
                "peak_score": 0.93,
                "source_window_ids": ["window-risk-001"],
                "contributions": [
                    {
                        "label": "explosion",
                        "window_id": "window-risk-001",
                        "window_audio_sha256": "d" * 64,
                        "patch_index": 0,
                        "start_sample": 0,
                        "end_sample": end_sample,
                        "start_seconds": 0,
                        "end_seconds": end_sample / YAMNET.sample_rate_hz,
                        "score": 0.93,
                        "winning_class": {
                            "index": winner.index,
                            "mid": winner.mid,
                            "display_name": winner.display_name,
                        },
                    }
                ],
            }
        ],
    }
    typed_payload = {
        **payload,
        "source": EvidenceSource.model_validate(payload["source"]),
        "model": EvidenceModel.model_validate(payload["model"]),
        "aggregation": AcousticAggregationSettings.model_validate(
            payload["aggregation"]
        ),
        "windows": tuple(
            EvidenceWindow.model_validate(item) for item in payload["windows"]
        ),
        "events": tuple(EvidenceEvent.model_validate(item) for item in payload["events"]),
    }
    temporary = AcousticEvidenceDocument.model_construct(
        evidence_id="0" * 64,
        created_at=NOW,
        **typed_payload,
    )
    identity = temporary.model_dump(
        mode="json", exclude={"evidence_id", "created_at"}
    )
    return AcousticEvidenceDocument(
        evidence_id=canonical_hash(identity),
        created_at=NOW,
        **typed_payload,
    )


def test_integrates_all_three_evidence_branches_and_scores_them(
    source: RiskSource,
    speech: SpeechEvidenceDocument,
    language: LanguageEvidenceDocument,
) -> None:
    acoustic = acoustic_document(speech)
    inputs = integrate_risk_inputs(
        source, acoustic=acoustic, speech=speech, language=language
    )

    assert inputs.acoustic.status is RiskInputStatus.PRESENT
    assert inputs.acoustic.event_count == 1
    assert inputs.acoustic.signals[0].source_event_index == 0
    assert inputs.acoustic.max_peak_score == 0.93
    assert inputs.speech.status is RiskInputStatus.PRESENT
    assert inputs.speech.segment_count == 2
    assert inputs.speech.accepted_transcript_count == 1
    assert inputs.speech.review_required_transcript_count == 1
    assert inputs.language.status is RiskInputStatus.PRESENT
    assert inputs.language.finding_count == 1
    assert inputs.language.signals[0].category.value == "distress"
    assert score_risk(inputs, now=NOW).score == 82


def test_references_hash_the_complete_exact_evidence_documents(
    source: RiskSource,
    speech: SpeechEvidenceDocument,
    language: LanguageEvidenceDocument,
) -> None:
    acoustic = acoustic_document(speech)
    inputs = integrate_risk_inputs(
        source, acoustic=acoustic, speech=speech, language=language
    )

    assert inputs.acoustic.evidence.evidence_sha256 == canonical_hash(
        acoustic.model_dump(mode="json")
    )
    assert inputs.speech.evidence.evidence_sha256 == canonical_hash(
        speech.model_dump(mode="json")
    )
    assert inputs.language.evidence.evidence_sha256 == canonical_hash(
        language.model_dump(mode="json")
    )


def test_missing_documents_produce_explicit_missing_statuses(
    source: RiskSource,
) -> None:
    inputs = integrate_risk_inputs(source)

    assert inputs.acoustic.status is RiskInputStatus.MISSING
    assert inputs.speech.status is RiskInputStatus.MISSING
    assert inputs.language.status is RiskInputStatus.MISSING


def test_missing_language_is_distinct_from_no_accepted_text(
    source: RiskSource, speech: SpeechEvidenceDocument
) -> None:
    with_accepted_text = integrate_risk_inputs(source, speech=speech)
    assert with_accepted_text.language.status is RiskInputStatus.MISSING

    payload = speech.model_dump(mode="python")
    payload["transcription_model"] = None
    payload["transcribed_segment_count"] = 0
    for segment in payload["segments"]:
        segment["transcript"] = None
        segment["assessment"] = {
            "reliability": "not_transcribed",
            "reason_codes": ["transcript_not_attempted"],
            "downstream_text_allowed": False,
            "human_review_required": False,
        }
    no_text = SpeechEvidenceDocument.model_validate(payload)

    without_accepted_text = integrate_risk_inputs(source, speech=no_text)
    assert without_accepted_text.language.status is RiskInputStatus.NO_ACCEPTED_TEXT
    assert without_accepted_text.speech.accepted_transcript_count == 0


def test_empty_language_artifact_is_accepted_as_no_accepted_text(
    source: RiskSource,
    speech: SpeechEvidenceDocument,
    language_data: dict[str, object],
) -> None:
    speech_payload = speech.model_dump(mode="python")
    speech_payload["transcription_model"] = None
    speech_payload["transcribed_segment_count"] = 0
    for segment in speech_payload["segments"]:
        segment["transcript"] = None
        segment["assessment"] = {
            "reliability": "not_transcribed",
            "reason_codes": ["transcript_not_attempted"],
            "downstream_text_allowed": False,
            "human_review_required": False,
        }
    no_text = SpeechEvidenceDocument.model_validate(speech_payload)
    language_data["source"].update(
        speech_evidence_id=no_text.evidence_id,
        speech_evidence_sha256=canonical_hash(no_text.model_dump(mode="json")),
    )
    language_data.update(
        input_transcript_count=0,
        analyzed_transcript_count=0,
        finding_count=0,
        analyses=[],
    )
    empty_language = LanguageEvidenceDocument.model_validate(language_data)

    inputs = integrate_risk_inputs(source, speech=no_text, language=empty_language)

    assert inputs.language.status is RiskInputStatus.NO_ACCEPTED_TEXT
    assert inputs.language.evidence is None


def test_acoustic_only_scope_marks_later_branches_not_permitted(
    source: RiskSource, speech: SpeechEvidenceDocument
) -> None:
    acoustic_source = source.model_copy(
        update={"processing_scope": ProcessingScope.ACOUSTIC_ONLY}
    )
    acoustic = acoustic_document(speech)

    inputs = integrate_risk_inputs(acoustic_source, acoustic=acoustic)

    assert inputs.acoustic.status is RiskInputStatus.PRESENT
    assert inputs.speech.status is RiskInputStatus.NOT_PERMITTED
    assert inputs.language.status is RiskInputStatus.NOT_PERMITTED


def test_acoustic_only_scope_rejects_speech_or_language_evidence(
    source: RiskSource,
    speech: SpeechEvidenceDocument,
    language: LanguageEvidenceDocument,
) -> None:
    acoustic_source = source.model_copy(
        update={"processing_scope": ProcessingScope.ACOUSTIC_ONLY}
    )

    with pytest.raises(RiskIntegrationError) as error:
        integrate_risk_inputs(acoustic_source, speech=speech, language=language)

    assert error.value.code == "evidence_not_permitted"


def test_language_requires_speech_evidence(
    source: RiskSource, language: LanguageEvidenceDocument
) -> None:
    with pytest.raises(RiskIntegrationError) as error:
        integrate_risk_inputs(source, language=language)

    assert error.value.code == "missing_speech_evidence"


def test_rejects_cross_branch_preparation_mismatch(
    source: RiskSource, speech: SpeechEvidenceDocument
) -> None:
    acoustic = acoustic_document(speech, manifest_sha256="f" * 64)

    with pytest.raises(RiskIntegrationError) as error:
        integrate_risk_inputs(source, acoustic=acoustic, speech=speech)

    assert error.value.code == "source_mismatch"


def test_rejects_evidence_for_a_different_risk_source(
    source: RiskSource, speech: SpeechEvidenceDocument
) -> None:
    different = source.model_copy(update={"clip_id": "different-clip"})

    with pytest.raises(RiskIntegrationError) as error:
        integrate_risk_inputs(different, speech=speech)

    assert error.value.code == "source_mismatch"


def test_rejects_language_with_wrong_speech_artifact_hash(
    source: RiskSource,
    speech: SpeechEvidenceDocument,
    language: LanguageEvidenceDocument,
) -> None:
    payload = language.model_dump(mode="python")
    payload["source"]["speech_evidence_sha256"] = "f" * 64
    changed = LanguageEvidenceDocument.model_validate(payload)

    with pytest.raises(RiskIntegrationError) as error:
        integrate_risk_inputs(source, speech=speech, language=changed)

    assert error.value.code == "provenance_mismatch"


def test_rejects_language_that_differs_from_accepted_transcript(
    source: RiskSource,
    speech: SpeechEvidenceDocument,
    language: LanguageEvidenceDocument,
) -> None:
    payload = language.model_dump(mode="python")
    payload["analyses"][0]["transcript"]["transcript_sha256"] = "f" * 64
    changed = LanguageEvidenceDocument.model_validate(payload)

    with pytest.raises(RiskIntegrationError) as error:
        integrate_risk_inputs(source, speech=speech, language=changed)

    assert error.value.code == "transcript_mismatch"


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("source", {"clip_id": "wrong"}, "invalid_source"),
        ("acoustic", {"events": []}, "invalid_acoustic_evidence"),
        ("speech", {"segments": []}, "invalid_speech_evidence"),
        ("language", {"analyses": []}, "invalid_language_evidence"),
    ],
)
def test_rejects_untyped_inputs(
    source: RiskSource,
    speech: SpeechEvidenceDocument,
    language: LanguageEvidenceDocument,
    field: str,
    value: object,
    code: str,
) -> None:
    arguments = {
        "source": source,
        "acoustic": None,
        "speech": speech,
        "language": language,
    }
    arguments[field] = value

    with pytest.raises(RiskIntegrationError) as error:
        integrate_risk_inputs(
            arguments.pop("source"),
            acoustic=arguments["acoustic"],
            speech=arguments["speech"],
            language=arguments["language"],
        )

    assert error.value.code == code


def test_risk_inputs_do_not_copy_transcript_or_raw_audio(
    source: RiskSource,
    speech: SpeechEvidenceDocument,
    language: LanguageEvidenceDocument,
) -> None:
    inputs = integrate_risk_inputs(source, speech=speech, language=language)
    serialized = inputs.model_dump_json()

    assert "Please call for help" not in serialized
    assert "The door may be open" not in serialized
    assert "audio_bytes" not in serialized
    assert "speaker" not in serialized
