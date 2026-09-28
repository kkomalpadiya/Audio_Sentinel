"""B8.1 tests for versioned final JSON report serialization."""

from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
import json
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from audio_sentinel.config import AudioSettings
from audio_sentinel.consensus_contracts import ConsensusOutcome
from audio_sentinel.contracts import ProcessingScope
from audio_sentinel.evaluator import OfflineEvaluatorModels
from audio_sentinel.final_report import (
    FINAL_REPORT_DIRECTORY,
    FINAL_REPORT_FILENAME,
    FinalReportDocument,
    FinalReportError,
    FinalReportPersistencePolicy,
    build_final_report,
    final_report_schema_documents,
    load_final_report,
    save_final_report,
)
from audio_sentinel.risk_contracts import RiskEvidenceKind, RiskInputStatus
from test_evaluator import (
    NOW,
    FakeVadSession,
    evaluator,
    input_clip,
    loaded_acoustic,
    loaded_transcriber,
    loaded_vad,
)


def evaluated_result(
    settings,
    *,
    scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH,
    speech: bool = True,
):
    service, *_ = evaluator(settings, speech=speech)
    return service.evaluate(
        input_clip(settings, scope),
        audio_settings=AudioSettings(
            normalize_loudness=False,
            window_seconds=(1.0,),
        ),
        now=NOW,
    )


def no_accepted_text_result(settings):
    service, acoustic, _vad, transcriber = evaluator(
        settings,
        acoustic_label=None,
        transcript="harmless words",
    )
    service = replace(
        service,
        models=OfflineEvaluatorModels(
            acoustic=loaded_acoustic(acoustic),
            vad=loaded_vad(FakeVadSession(0.1)),
            transcription=loaded_transcriber(transcriber),
        ),
    )
    return service.evaluate(
        input_clip(settings, ProcessingScope.ACOUSTIC_AND_SPEECH),
        audio_settings=AudioSettings(
            normalize_loudness=False,
            window_seconds=(1.0,),
        ),
        now=NOW,
    )


def relative_report_path(report: FinalReportDocument) -> str:
    return f"{FINAL_REPORT_DIRECTORY}/{report.report_id}/{FINAL_REPORT_FILENAME}"


def test_builds_versioned_privacy_minimized_alert_candidate_report(
    temporary_settings,
):
    result = evaluated_result(temporary_settings)

    report = build_final_report(result, now=NOW)
    payload = report.model_dump_json()

    assert report.schema_version == report.format_version == "1.0"
    assert report.document_type == "offline_evaluation_report"
    assert report.report_id.startswith("report-")
    assert report.source.clip_id == result.clip_id
    assert report.source.raw_audio_sha256 == result.prepared.manifest.source.sha256
    assert tuple(item.kind for item in report.evidence) == tuple(RiskEvidenceKind)
    assert all(item.evidence_sha256 for item in report.evidence)
    assert report.risk_assessment == result.risk_assessment
    assert report.agreement == result.agreement
    assert report.decision == result.decision
    assert report.summary.outcome is ConsensusOutcome.ALERT
    assert report.summary.alert_candidate is True
    assert report.summary.notification_delivery == "not_sent"
    assert report.summary.alert_delivery_authorized is False
    assert "I will kill you" not in payload
    assert str(result.prepared.audio.audio_path) not in payload
    assert str(result.acoustic_evidence.evidence_path) not in payload
    assert ".wav" not in payload
    assert '"transcript":' not in payload.lower()


def test_report_id_is_semantic_and_excludes_only_report_creation_time(
    temporary_settings,
):
    result = evaluated_result(temporary_settings)

    first = build_final_report(result, now=NOW)
    second = build_final_report(result, now=NOW + timedelta(hours=2))

    assert first.report_id == second.report_id
    assert first.created_at != second.created_at
    assert first.model_copy(update={"created_at": second.created_at}) == second


def test_acoustic_only_report_marks_consent_excluded_branches(temporary_settings):
    result = evaluated_result(
        temporary_settings,
        scope=ProcessingScope.ACOUSTIC_ONLY,
        speech=False,
    )

    report = build_final_report(result, now=NOW)
    speech, language = report.evidence[1:]

    assert speech.input_status is RiskInputStatus.NOT_PERMITTED
    assert language.input_status is RiskInputStatus.NOT_PERMITTED
    assert speech.evidence_id is speech.evidence_sha256 is None
    assert language.evidence_id is language.evidence_sha256 is None
    assert report.summary.speech_segment_count is None
    assert report.summary.language_finding_count is None
    assert report.summary.outcome is ConsensusOutcome.REVIEW
    assert report.summary.alert_candidate is False


def test_no_accepted_text_still_pins_completed_language_artifact(
    temporary_settings,
):
    result = no_accepted_text_result(temporary_settings)

    report = build_final_report(result, now=NOW)
    language = report.evidence[2]

    assert language.kind is RiskEvidenceKind.LANGUAGE
    assert language.input_status is RiskInputStatus.NO_ACCEPTED_TEXT
    assert language.evidence_id == result.language.evidence.evidence_id
    assert language.evidence_sha256 is not None
    assert report.summary.language_finding_count == 0
    assert report.summary.outcome is ConsensusOutcome.NO_ACTION


def test_save_load_and_retry_reuse_one_immutable_report_bundle(temporary_settings):
    result = evaluated_result(temporary_settings)

    first = save_final_report(temporary_settings.paths, result, now=NOW)
    second = save_final_report(
        temporary_settings.paths,
        result,
        now=NOW + timedelta(minutes=5),
    )
    loaded = load_final_report(
        temporary_settings.paths,
        relative_report_path(first.report),
    )

    assert first.reused is False
    assert second.reused is True
    assert second.report == first.report == loaded
    assert first.report_path == (
        temporary_settings.paths.processed_data
        / FINAL_REPORT_DIRECTORY
        / first.report.report_id
        / FINAL_REPORT_FILENAME
    )
    assert {item.name for item in first.directory.iterdir()} == {
        FINAL_REPORT_FILENAME
    }
    assert first.report_path.read_bytes().endswith(b"\n")


def test_save_rejects_output_over_limit_without_creating_report_directory(
    temporary_settings,
):
    result = evaluated_result(temporary_settings)

    with pytest.raises(FinalReportError) as captured:
        save_final_report(
            temporary_settings.paths,
            result,
            policy=FinalReportPersistencePolicy(max_document_bytes=32),
            now=NOW,
        )

    assert captured.value.code == "output_too_large"
    assert not (
        temporary_settings.paths.processed_data / FINAL_REPORT_DIRECTORY
    ).exists()


def test_save_rejects_acoustic_evidence_that_changed_after_evaluation(
    temporary_settings,
):
    result = evaluated_result(temporary_settings)
    payload = json.loads(
        result.acoustic_evidence.evidence_path.read_text(encoding="utf-8")
    )
    payload["created_at"] = "2026-09-27T12:01:00Z"
    result.acoustic_evidence.evidence_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(FinalReportError) as captured:
        save_final_report(temporary_settings.paths, result, now=NOW)

    assert captured.value.code == "source_changed"
    assert not (
        temporary_settings.paths.processed_data / FINAL_REPORT_DIRECTORY
    ).exists()


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../report.json",
        Path("C:/outside/report.json"),
    ),
)
def test_load_rejects_paths_outside_processed_data(
    temporary_settings,
    unsafe_path,
):
    with pytest.raises(FinalReportError) as captured:
        load_final_report(temporary_settings.paths, unsafe_path)

    assert captured.value.code == "invalid_path"


def test_load_rejects_tampering_and_unexpected_bundle_files(temporary_settings):
    result = evaluated_result(temporary_settings)
    saved = save_final_report(temporary_settings.paths, result, now=NOW)
    original = saved.report_path.read_text(encoding="utf-8")
    payload = json.loads(original)
    payload["risk_assessment"]["score"] = 0
    saved.report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FinalReportError) as captured:
        load_final_report(
            temporary_settings.paths,
            relative_report_path(saved.report),
        )

    assert captured.value.code == "invalid_document"

    saved.report_path.write_text(original, encoding="utf-8")
    (saved.directory / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(FinalReportError) as captured:
        load_final_report(
            temporary_settings.paths,
            relative_report_path(saved.report),
        )

    assert captured.value.code == "output_conflict"


def test_load_rejects_valid_report_under_wrong_identity_path(temporary_settings):
    result = evaluated_result(temporary_settings)
    saved = save_final_report(temporary_settings.paths, result, now=NOW)
    wrong = (
        temporary_settings.paths.processed_data
        / FINAL_REPORT_DIRECTORY
        / ("report-" + "0" * 64)
    )
    shutil.copytree(saved.directory, wrong)

    with pytest.raises(FinalReportError) as captured:
        load_final_report(
            temporary_settings.paths,
            f"{FINAL_REPORT_DIRECTORY}/{wrong.name}/{FINAL_REPORT_FILENAME}",
        )

    assert captured.value.code == "output_conflict"


def test_load_honors_document_byte_limit(temporary_settings):
    result = evaluated_result(temporary_settings)
    saved = save_final_report(temporary_settings.paths, result, now=NOW)

    with pytest.raises(FinalReportError) as captured:
        load_final_report(
            temporary_settings.paths,
            relative_report_path(saved.report),
            policy=FinalReportPersistencePolicy(max_document_bytes=32),
        )

    assert captured.value.code == "file_too_large"


def test_builder_rejects_an_internally_inconsistent_evaluation(temporary_settings):
    result = evaluated_result(temporary_settings)
    inconsistent = replace(
        result,
        risk_inputs=result.risk_inputs.model_copy(
            update={
                "source": result.risk_inputs.source.model_copy(
                    update={"clip_id": "different-clip"}
                )
            }
        ),
    )

    with pytest.raises(FinalReportError) as captured:
        build_final_report(inconsistent, now=NOW)

    assert captured.value.code == "invalid_evaluation"
    assert "different-clip" not in str(captured.value)


def test_contract_is_frozen_strict_and_schema_is_checked_in(
    temporary_settings,
    project_root,
):
    report = build_final_report(evaluated_result(temporary_settings), now=NOW)

    with pytest.raises(ValidationError):
        FinalReportDocument.model_validate(
            report.model_dump(mode="python") | {"unexpected": True}
        )
    with pytest.raises((ValidationError, FrozenInstanceError)):
        report.report_id = "report-" + "0" * 64

    checked_in = json.loads(
        (project_root / "docs/schemas/v1/final-report.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert final_report_schema_documents() == {
        "final-report.schema.json": checked_in
    }
