"""B7.1 evidence-agreement and conflict-detection rule tests."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.consensus_contracts import BranchAgreement
from audio_sentinel.consensus_rules import (
    BUILTIN_AGREEMENT_RULE_SET_ID,
    BUILTIN_AGREEMENT_RULE_SET_SHA256,
    MAX_AGREEMENT_RULE_SET_BYTES,
    AgreementReasonCode,
    AgreementRuleSet,
    BranchAgreementEvaluation,
    ConsensusAgreementError,
    ConsensusAgreementEvaluation,
    LoadedAgreementRuleSet,
    agreement_schema_documents,
    evaluate_evidence_agreement,
    load_agreement_rule_set_bytes,
    load_builtin_agreement_rule_set,
    risk_assessment_sha256,
    write_agreement_schemas,
)
from audio_sentinel.contracts import EventLabel, ProcessingScope
from audio_sentinel.language_contracts import LanguageCategory, LanguageReasonCode
from audio_sentinel.risk_contracts import (
    AcousticRiskInputs,
    AcousticRiskSignal,
    LanguageRiskInputs,
    LanguageRiskSignal,
    RiskAssessmentDocument,
    RiskEvidenceKind,
    RiskEvidenceReference,
    RiskInputSet,
    RiskInputStatus,
    RiskSource,
    SpeechRiskInputs,
)
from audio_sentinel.risk_scoring import score_risk


NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


def test_builtin_rules_are_pinned_complete_and_offline() -> None:
    loaded = load_builtin_agreement_rule_set()

    assert loaded.rule_set.rule_set_id == BUILTIN_AGREEMENT_RULE_SET_ID
    assert loaded.artifact_sha256 == BUILTIN_AGREEMENT_RULE_SET_SHA256
    assert loaded.artifact_size_bytes > 0
    assert loaded.rule_set.acoustic_support.minimum_peak_score == 0.85
    assert loaded.rule_set.acoustic_support.support_labels == (
        EventLabel.SIREN,
        EventLabel.SMOKE_ALARM,
        EventLabel.GLASS_BREAK,
        EventLabel.CROWD_PANIC,
        EventLabel.DISTRESS_SPEECH,
        EventLabel.THREATENING_SPEECH,
        EventLabel.WEAPON_REFERENCE,
        EventLabel.GUNSHOT,
        EventLabel.EXPLOSION,
    )


@pytest.mark.parametrize("raw", [b"", b"[]", b"{}", b"not-json"])
def test_rule_loader_rejects_empty_or_invalid_artifacts(raw: bytes) -> None:
    with pytest.raises(ValueError):
        load_agreement_rule_set_bytes(raw)


def test_rule_loader_rejects_wrong_type_size_and_checksum() -> None:
    with pytest.raises(TypeError, match="must be bytes"):
        load_agreement_rule_set_bytes("{}")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="byte limit"):
        load_agreement_rule_set_bytes(b"x" * (MAX_AGREEMENT_RULE_SET_BYTES + 1))

    raw = _builtin_rule_bytes()
    with pytest.raises(ValueError, match="checksum"):
        load_agreement_rule_set_bytes(raw, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda data: data.update({"unexpected": True}), "contract validation"),
        (
            lambda data: data["acoustic_support"]["support_labels"].reverse(),
            "contract validation",
        ),
        (
            lambda data: data["speech_consistency"]["speech_like_labels"].pop(),
            "contract validation",
        ),
        (
            lambda data: data["language_support"]["neutral_categories"].pop(),
            "contract validation",
        ),
        (
            lambda data: data["conflicts"].update(
                {"no_speech_vs_vad_speech": False}
            ),
            "contract validation",
        ),
    ],
)
def test_rule_contract_rejects_unknown_fields_and_taxonomy_drift(
    mutate, message: str
) -> None:
    data = json.loads(_builtin_rule_bytes())
    mutate(data)

    with pytest.raises(ValueError, match=message):
        load_agreement_rule_set_bytes(_json_bytes(data))


def test_high_confidence_acoustic_and_active_language_support_risk() -> None:
    assessment = _assessment(
        acoustic=[(EventLabel.GUNSHOT, 0.91)],
        speech_segments=1,
        accepted_transcripts=1,
        language=[LanguageCategory.THREAT],
    )

    result = evaluate_evidence_agreement(assessment, now=NOW)

    assert result.supporting_branches == (
        RiskEvidenceKind.ACOUSTIC,
        RiskEvidenceKind.LANGUAGE,
    )
    assert result.conflicting_branches == ()
    assert result.has_conflict is False
    assert [branch.agreement for branch in result.branches] == [
        BranchAgreement.SUPPORTS_RISK,
        BranchAgreement.NEUTRAL,
        BranchAgreement.SUPPORTS_RISK,
    ]
    assert result.branch_states == tuple(
        branch.as_branch_state() for branch in result.branches
    )


@pytest.mark.parametrize("peak", [0.0, 0.849999])
def test_low_confidence_acoustic_risk_signal_is_neutral(peak: float) -> None:
    result = evaluate_evidence_agreement(
        _assessment(acoustic=[(EventLabel.GLASS_BREAK, peak)]),
        now=NOW,
    )

    acoustic = result.branches[0]
    assert acoustic.agreement is BranchAgreement.NEUTRAL
    assert acoustic.reason_codes == (
        AgreementReasonCode.LOW_CONFIDENCE_RISK_SIGNAL,
    )


def test_acoustic_support_threshold_is_inclusive() -> None:
    result = evaluate_evidence_agreement(
        _assessment(acoustic=[(EventLabel.GLASS_BREAK, 0.85)]),
        now=NOW,
    )

    assert result.branches[0].agreement is BranchAgreement.SUPPORTS_RISK


@pytest.mark.parametrize(
    "category,expected_reason",
    [
        (LanguageCategory.NO_CONCERNING_MATCH, AgreementReasonCode.SAFE_LANGUAGE_CONTEXT),
        (LanguageCategory.CONTEXT_SUPPRESSED, AgreementReasonCode.SAFE_LANGUAGE_CONTEXT),
        (LanguageCategory.AMBIGUOUS, AgreementReasonCode.AMBIGUOUS_LANGUAGE),
    ],
)
def test_safe_and_ambiguous_language_remains_neutral(
    category: LanguageCategory,
    expected_reason: AgreementReasonCode,
) -> None:
    result = evaluate_evidence_agreement(
        _assessment(
            speech_segments=1,
            accepted_transcripts=1,
            language=[category],
        ),
        now=NOW,
    )

    language = result.branches[2]
    assert language.agreement is BranchAgreement.NEUTRAL
    assert language.reason_codes == (expected_reason,)


def test_high_confidence_no_speech_conflicts_with_vad_speech_and_language() -> None:
    result = evaluate_evidence_agreement(
        _assessment(
            acoustic=[(EventLabel.NO_SPEECH, 0.95)],
            speech_segments=1,
            accepted_transcripts=1,
            language=[LanguageCategory.DISTRESS],
        ),
        now=NOW,
    )

    assert result.conflicting_branches == tuple(RiskEvidenceKind)
    assert result.has_conflict is True
    for branch in result.branches:
        assert branch.agreement is BranchAgreement.CONFLICTS_RISK
        assert AgreementReasonCode.ACOUSTIC_NO_SPEECH_CONFLICT in branch.reason_codes


@pytest.mark.parametrize(
    "label",
    [
        EventLabel.SPEECH_PRESENT,
        EventLabel.DISTRESS_SPEECH,
        EventLabel.THREATENING_SPEECH,
        EventLabel.WEAPON_REFERENCE,
    ],
)
def test_high_confidence_acoustic_speech_conflicts_with_zero_vad_segments(
    label: EventLabel,
) -> None:
    result = evaluate_evidence_agreement(
        _assessment(acoustic=[(label, 0.9)]),
        now=NOW,
    )

    assert result.conflicting_branches == (
        RiskEvidenceKind.ACOUSTIC,
        RiskEvidenceKind.SPEECH,
    )
    assert AgreementReasonCode.ACOUSTIC_SPEECH_ABSENCE_CONFLICT in (
        result.branches[0].reason_codes
    )


def test_non_threatening_acoustic_speech_conflicts_with_active_language() -> None:
    result = evaluate_evidence_agreement(
        _assessment(
            acoustic=[(EventLabel.NON_THREATENING_SPEECH, 0.9)],
            speech_segments=1,
            accepted_transcripts=1,
            language=[LanguageCategory.WEAPON_REFERENCE],
        ),
        now=NOW,
    )

    assert result.conflicting_branches == (
        RiskEvidenceKind.ACOUSTIC,
        RiskEvidenceKind.LANGUAGE,
    )
    assert result.branches[1].agreement is BranchAgreement.NEUTRAL
    assert all(
        AgreementReasonCode.NON_THREATENING_LANGUAGE_CONFLICT
        in result.branches[index].reason_codes
        for index in (0, 2)
    )


def test_language_findings_without_accepted_transcript_are_a_conflict() -> None:
    result = evaluate_evidence_agreement(
        _assessment(
            speech_segments=1,
            accepted_transcripts=0,
            language=[LanguageCategory.THREAT],
        ),
        now=NOW,
    )

    assert result.conflicting_branches == (
        RiskEvidenceKind.SPEECH,
        RiskEvidenceKind.LANGUAGE,
    )
    assert AgreementReasonCode.LANGUAGE_WITHOUT_ACCEPTED_TRANSCRIPT_CONFLICT in (
        result.branches[2].reason_codes
    )


@pytest.mark.parametrize(
    "kind,status,agreement,reason",
    [
        (
            RiskEvidenceKind.ACOUSTIC,
            RiskInputStatus.MISSING,
            BranchAgreement.UNAVAILABLE,
            AgreementReasonCode.BRANCH_MISSING,
        ),
        (
            RiskEvidenceKind.SPEECH,
            RiskInputStatus.NOT_PERMITTED,
            BranchAgreement.NOT_PERMITTED,
            AgreementReasonCode.BRANCH_NOT_PERMITTED,
        ),
        (
            RiskEvidenceKind.LANGUAGE,
            RiskInputStatus.NO_ACCEPTED_TEXT,
            BranchAgreement.NOT_APPLICABLE,
            AgreementReasonCode.BRANCH_NOT_APPLICABLE,
        ),
    ],
)
def test_non_present_branch_statuses_are_preserved(
    kind: RiskEvidenceKind,
    status: RiskInputStatus,
    agreement: BranchAgreement,
    reason: AgreementReasonCode,
) -> None:
    if kind is RiskEvidenceKind.ACOUSTIC:
        assessment = _assessment(acoustic_status=status)
    elif kind is RiskEvidenceKind.SPEECH:
        assessment = _assessment(
            scope=ProcessingScope.ACOUSTIC_ONLY,
            speech_status=status,
            language_status=RiskInputStatus.NOT_PERMITTED,
        )
    else:
        assessment = _assessment(language_status=status)

    result = evaluate_evidence_agreement(assessment, now=NOW)
    branch = result.branches[list(RiskEvidenceKind).index(kind)]

    assert branch.input_status is status
    assert branch.agreement is agreement
    assert branch.reason_codes == (reason,)


def test_multiple_signals_use_max_confidence_semantics_without_count_inflation() -> None:
    result = evaluate_evidence_agreement(
        _assessment(
            acoustic=[
                (EventLabel.GUNSHOT, 0.4),
                (EventLabel.GLASS_BREAK, 0.86),
                (EventLabel.EXPLOSION, 0.2),
            ]
        ),
        now=NOW,
    )

    assert result.branches[0].agreement is BranchAgreement.SUPPORTS_RISK
    assert result.branches[0].reason_codes == (
        AgreementReasonCode.HIGH_CONFIDENCE_RISK_SIGNAL,
    )


def test_custom_threshold_is_respected_at_its_boundary() -> None:
    data = json.loads(_builtin_rule_bytes())
    data["acoustic_support"]["minimum_peak_score"] = 0.9
    loaded = load_agreement_rule_set_bytes(_json_bytes(data))
    assessment = _assessment(acoustic=[(EventLabel.SIREN, 0.89)])

    below = evaluate_evidence_agreement(assessment, rule_set=loaded, now=NOW)
    assert below.branches[0].agreement is BranchAgreement.NEUTRAL

    at_boundary = evaluate_evidence_agreement(
        _assessment(acoustic=[(EventLabel.SIREN, 0.9)]),
        rule_set=loaded,
        now=NOW,
    )
    assert at_boundary.branches[0].agreement is BranchAgreement.SUPPORTS_RISK


def test_evaluation_identity_is_deterministic_and_time_independent() -> None:
    assessment = _assessment(
        acoustic=[(EventLabel.GUNSHOT, 0.9)],
        speech_segments=1,
        accepted_transcripts=1,
        language=[LanguageCategory.THREAT],
    )

    first = evaluate_evidence_agreement(assessment, now=NOW)
    second = evaluate_evidence_agreement(
        assessment,
        now=datetime(2026, 9, 27, 8, tzinfo=UTC),
    )

    assert first.evaluation_id == second.evaluation_id
    assert first.assessment_sha256 == risk_assessment_sha256(assessment)
    assert first.created_at != second.created_at


def test_evaluation_identity_changes_with_evidence_or_rules() -> None:
    first = evaluate_evidence_agreement(
        _assessment(acoustic=[(EventLabel.SIREN, 0.9)]), now=NOW
    )
    changed_evidence = evaluate_evidence_agreement(
        _assessment(acoustic=[(EventLabel.GLASS_BREAK, 0.9)]), now=NOW
    )
    data = json.loads(_builtin_rule_bytes())
    data["acoustic_support"]["minimum_peak_score"] = 0.8
    changed_rules = evaluate_evidence_agreement(
        _assessment(acoustic=[(EventLabel.SIREN, 0.9)]),
        rule_set=load_agreement_rule_set_bytes(_json_bytes(data)),
        now=NOW,
    )

    assert first.evaluation_id != changed_evidence.evaluation_id
    assert first.evaluation_id != changed_rules.evaluation_id


def test_invalid_input_time_and_loaded_rule_types_fail_safely() -> None:
    with pytest.raises(ConsensusAgreementError) as caught:
        evaluate_evidence_agreement("not-an-assessment", now=NOW)  # type: ignore[arg-type]
    assert caught.value.code == "invalid_assessment"
    assert "not-an-assessment" not in str(caught.value)

    with pytest.raises(ConsensusAgreementError) as caught:
        evaluate_evidence_agreement(_assessment(), now=datetime(2026, 9, 26, 12))
    assert caught.value.code == "invalid_time"

    with pytest.raises(ValueError, match="invalid type"):
        evaluate_evidence_agreement(_assessment(), rule_set="bad", now=NOW)  # type: ignore[arg-type]


def test_loaded_rule_integrity_metadata_is_revalidated() -> None:
    loaded = load_builtin_agreement_rule_set()

    with pytest.raises(ValueError, match="integrity"):
        evaluate_evidence_agreement(
            _assessment(),
            rule_set=LoadedAgreementRuleSet(
                rule_set=loaded.rule_set,
                artifact_sha256="bad",
                artifact_size_bytes=loaded.artifact_size_bytes,
            ),
            now=NOW,
        )


def test_public_results_are_immutable_and_json_portable() -> None:
    result = evaluate_evidence_agreement(_assessment(), now=NOW)

    with pytest.raises(ValidationError, match="frozen"):
        result.has_conflict = True  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        result.branches[0].agreement = BranchAgreement.CONFLICTS_RISK  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        load_builtin_agreement_rule_set().artifact_size_bytes = 0  # type: ignore[misc]

    round_trip = ConsensusAgreementEvaluation.model_validate_json(result.model_dump_json())
    assert round_trip == result


def test_evaluation_contract_rejects_tampered_summaries_and_identity() -> None:
    result = evaluate_evidence_agreement(
        _assessment(acoustic=[(EventLabel.SIREN, 0.9)]), now=NOW
    )
    payload = result.model_dump(mode="json")
    payload["supporting_branches"] = []
    with pytest.raises(ValidationError, match="supporting_branches"):
        ConsensusAgreementEvaluation.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["evaluation_id"] = "agreement-tampered"
    with pytest.raises(ValidationError, match="canonical"):
        ConsensusAgreementEvaluation.model_validate(payload)


def test_branch_evaluation_requires_canonical_reasons_and_conflict_consistency() -> None:
    with pytest.raises(ValidationError, match="conflict reason"):
        BranchAgreementEvaluation(
            kind="acoustic",
            input_status="present",
            agreement="conflicts_risk",
            reason_codes=(AgreementReasonCode.NO_RISK_SIGNAL,),
        )

    with pytest.raises(ValidationError, match="canonical"):
        BranchAgreementEvaluation(
            kind="acoustic",
            input_status="present",
            agreement="conflicts_risk",
            reason_codes=(
                AgreementReasonCode.ACOUSTIC_NO_SPEECH_CONFLICT,
                AgreementReasonCode.NO_RISK_SIGNAL,
            ),
        )

    with pytest.raises(ValidationError, match="exact status reason"):
        BranchAgreementEvaluation(
            kind="acoustic",
            input_status="missing",
            agreement="unavailable",
            reason_codes=(AgreementReasonCode.NO_RISK_SIGNAL,),
        )

    with pytest.raises(ValidationError, match="explicit support reason"):
        BranchAgreementEvaluation(
            kind="acoustic",
            input_status="present",
            agreement="supports_risk",
            reason_codes=(AgreementReasonCode.NO_RISK_SIGNAL,),
        )


def test_output_is_privacy_minimized() -> None:
    serialized = evaluate_evidence_agreement(
        _assessment(
            speech_segments=1,
            accepted_transcripts=1,
            language=[LanguageCategory.THREAT],
        ),
        now=NOW,
    ).model_dump_json()

    for forbidden in (
        "transcript_text",
        "matched_text",
        "audio_bytes",
        "speaker",
        "audio_path",
        "alert_id",
        "phone_number",
    ):
        assert forbidden not in serialized


def test_checked_in_schemas_match_generated_contracts(project_root: Path) -> None:
    generated = agreement_schema_documents()
    for filename, schema in generated.items():
        checked_in = json.loads(
            (project_root / "docs" / "schemas" / "v1" / filename).read_text(
                encoding="utf-8"
            )
        )
        assert checked_in == schema


def test_checked_in_example_validates(project_root: Path) -> None:
    example = ConsensusAgreementEvaluation.model_validate_json(
        (project_root / "docs" / "examples" / "consensus-agreement.json").read_text(
            encoding="utf-8"
        )
    )

    assert example.supporting_branches == ("acoustic", "language")
    assert example.conflicting_branches == ()
    assert example.rule_set.artifact_sha256 == BUILTIN_AGREEMENT_RULE_SET_SHA256


def test_schema_export_writes_portable_json(tmp_path: Path) -> None:
    exported = write_agreement_schemas(tmp_path)

    assert set(exported) == {
        "consensus-agreement.schema.json",
        "consensus-agreement-rule-set.schema.json",
    }
    for filename, destination in exported.items():
        assert json.loads(destination.read_text(encoding="utf-8")) == (
            agreement_schema_documents()[filename]
        )


def test_rule_contract_is_immutable() -> None:
    rules = load_builtin_agreement_rule_set().rule_set
    with pytest.raises(ValidationError, match="frozen"):
        rules.rule_set_version = "changed"  # type: ignore[misc]
    assert AgreementRuleSet.model_validate_json(rules.model_dump_json()) == rules


def _assessment(
    *,
    acoustic: list[tuple[EventLabel, float]] | None = None,
    acoustic_status: RiskInputStatus = RiskInputStatus.PRESENT,
    speech_segments: int = 0,
    accepted_transcripts: int = 0,
    language: list[LanguageCategory] | None = None,
    speech_status: RiskInputStatus = RiskInputStatus.PRESENT,
    language_status: RiskInputStatus | None = None,
    scope: ProcessingScope = ProcessingScope.ACOUSTIC_AND_SPEECH,
) -> RiskAssessmentDocument:
    acoustic = acoustic or []
    language = language or []
    if language_status is None:
        language_status = (
            RiskInputStatus.PRESENT
            if language
            else RiskInputStatus.NO_ACCEPTED_TEXT
        )
    if scope is ProcessingScope.ACOUSTIC_ONLY:
        speech_status = RiskInputStatus.NOT_PERMITTED
        language_status = RiskInputStatus.NOT_PERMITTED

    acoustic_signals = tuple(
        AcousticRiskSignal(
            label=label,
            start_sample=index * 16_000,
            end_sample=(index + 1) * 16_000,
            start_seconds=float(index),
            end_seconds=float(index + 1),
            peak_score=peak,
            source_event_index=index,
        )
        for index, (label, peak) in enumerate(acoustic)
    )
    language_signals = tuple(
        LanguageRiskSignal(
            finding_id=f"finding-{index:03d}",
            segment_id=f"segment-{index:03d}",
            category=category,
            reason_codes=_language_reasons(category),
            start_sample=(index + 5) * 16_000,
            end_sample=(index + 6) * 16_000,
            rule_match_count=(0 if category is LanguageCategory.NO_CONCERNING_MATCH else 1),
        )
        for index, category in enumerate(language)
    )
    inputs = RiskInputSet(
        source=RiskSource(
            clip_id="clip-consensus-test",
            consent_id="consent-consensus-test",
            processing_scope=scope,
            sample_rate_hz=16_000,
            num_samples=160_000,
        ),
        acoustic=AcousticRiskInputs(
            status=acoustic_status,
            evidence=(
                _evidence(RiskEvidenceKind.ACOUSTIC)
                if acoustic_status is RiskInputStatus.PRESENT
                else None
            ),
            event_count=(len(acoustic_signals) if acoustic_status is RiskInputStatus.PRESENT else 0),
            max_peak_score=(
                max((signal.peak_score for signal in acoustic_signals), default=None)
                if acoustic_status is RiskInputStatus.PRESENT
                else None
            ),
            signals=(acoustic_signals if acoustic_status is RiskInputStatus.PRESENT else ()),
        ),
        speech=SpeechRiskInputs(
            status=speech_status,
            evidence=(
                _evidence(RiskEvidenceKind.SPEECH)
                if speech_status is RiskInputStatus.PRESENT
                else None
            ),
            segment_count=(speech_segments if speech_status is RiskInputStatus.PRESENT else 0),
            accepted_transcript_count=(
                accepted_transcripts if speech_status is RiskInputStatus.PRESENT else 0
            ),
            review_required_transcript_count=0,
            max_vad_score=(
                0.9
                if speech_status is RiskInputStatus.PRESENT and speech_segments > 0
                else None
            ),
        ),
        language=LanguageRiskInputs(
            status=language_status,
            evidence=(
                _evidence(RiskEvidenceKind.LANGUAGE)
                if language_status is RiskInputStatus.PRESENT
                else None
            ),
            finding_count=(
                len(language_signals)
                if language_status is RiskInputStatus.PRESENT
                else 0
            ),
            signals=(
                language_signals
                if language_status is RiskInputStatus.PRESENT
                else ()
            ),
        ),
    )
    return score_risk(inputs, now=NOW)


def _evidence(kind: RiskEvidenceKind) -> RiskEvidenceReference:
    return RiskEvidenceReference(
        kind=kind,
        document_type={
            RiskEvidenceKind.ACOUSTIC: "acoustic_event_candidates",
            RiskEvidenceKind.SPEECH: "speech_evidence_candidates",
            RiskEvidenceKind.LANGUAGE: "language_evidence",
        }[kind],
        evidence_id=f"{kind.value}-evidence-test",
        evidence_sha256={
            RiskEvidenceKind.ACOUSTIC: "a" * 64,
            RiskEvidenceKind.SPEECH: "b" * 64,
            RiskEvidenceKind.LANGUAGE: "c" * 64,
        }[kind],
    )


def _language_reasons(
    category: LanguageCategory,
) -> tuple[LanguageReasonCode, ...]:
    if category is LanguageCategory.NO_CONCERNING_MATCH:
        return (LanguageReasonCode.NO_RULE_MATCH,)
    if category is LanguageCategory.AMBIGUOUS:
        return (
            LanguageReasonCode.PHRASE_MATCH,
            LanguageReasonCode.INSUFFICIENT_CONTEXT,
        )
    if category is LanguageCategory.CONTEXT_SUPPRESSED:
        return (
            LanguageReasonCode.PHRASE_MATCH,
            LanguageReasonCode.EXPLICIT_NEGATION,
        )
    return (LanguageReasonCode.PHRASE_MATCH,)


def _builtin_rule_bytes() -> bytes:
    return (
        Path(__file__).resolve().parents[1]
        / "src"
        / "audio_sentinel"
        / "resources"
        / "consensus-agreement-rules-v1.json"
    ).read_bytes()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")
