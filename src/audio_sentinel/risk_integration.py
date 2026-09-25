"""A6.2 integration of acoustic, speech, and language evidence into risk inputs."""

from __future__ import annotations

import hashlib
import json

from audio_sentinel.acoustic_evidence import AcousticEvidenceDocument
from audio_sentinel.contracts import ProcessingScope
from audio_sentinel.language_contracts import LanguageEvidenceDocument
from audio_sentinel.risk_contracts import (
    AcousticRiskInputs,
    AcousticRiskSignal,
    LanguageRiskInputs,
    LanguageRiskSignal,
    RiskEvidenceKind,
    RiskEvidenceReference,
    RiskInputSet,
    RiskInputStatus,
    RiskSource,
    SpeechRiskInputs,
)
from audio_sentinel.speech_contracts import (
    SpeechEvidenceDocument,
    SpeechSegmentEvidence,
    TranscriptReliability,
)


class RiskIntegrationError(RuntimeError):
    """Stable integration failure code plus a safe, non-sensitive explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _artifact_hash(document: object) -> str:
    return _canonical_hash(document.model_dump(mode="json"))


def _validated_source(source: RiskSource) -> RiskSource:
    if not isinstance(source, RiskSource):
        raise RiskIntegrationError(
            "invalid_source", "A6.1 risk source metadata is required."
        )
    try:
        validated = RiskSource.model_validate(source.model_dump(mode="python"))
    except Exception as error:
        raise RiskIntegrationError(
            "invalid_source", "Risk source metadata failed contract validation."
        ) from error
    if validated != source or validated.processing_scope is ProcessingScope.NONE:
        raise RiskIntegrationError(
            "invalid_source", "Risk source metadata is not eligible for evidence integration."
        )
    return validated


def _validated_acoustic(
    document: AcousticEvidenceDocument,
) -> AcousticEvidenceDocument:
    if not isinstance(document, AcousticEvidenceDocument):
        raise RiskIntegrationError(
            "invalid_acoustic_evidence", "Acoustic evidence has an invalid type."
        )
    try:
        validated = AcousticEvidenceDocument.model_validate(
            document.model_dump(mode="python")
        )
    except Exception as error:
        raise RiskIntegrationError(
            "invalid_acoustic_evidence", "Acoustic evidence failed contract validation."
        ) from error
    if validated != document:
        raise RiskIntegrationError(
            "invalid_acoustic_evidence", "Acoustic evidence integrity validation failed."
        )
    return validated


def _validated_speech(document: SpeechEvidenceDocument) -> SpeechEvidenceDocument:
    if not isinstance(document, SpeechEvidenceDocument):
        raise RiskIntegrationError(
            "invalid_speech_evidence", "Speech evidence has an invalid type."
        )
    try:
        validated = SpeechEvidenceDocument.model_validate(
            document.model_dump(mode="python")
        )
    except Exception as error:
        raise RiskIntegrationError(
            "invalid_speech_evidence", "Speech evidence failed contract validation."
        ) from error
    if validated != document:
        raise RiskIntegrationError(
            "invalid_speech_evidence", "Speech evidence integrity validation failed."
        )
    return validated


def _validated_language(
    document: LanguageEvidenceDocument,
) -> LanguageEvidenceDocument:
    if not isinstance(document, LanguageEvidenceDocument):
        raise RiskIntegrationError(
            "invalid_language_evidence", "Language evidence has an invalid type."
        )
    try:
        validated = LanguageEvidenceDocument.model_validate(
            document.model_dump(mode="python")
        )
    except Exception as error:
        raise RiskIntegrationError(
            "invalid_language_evidence", "Language evidence failed contract validation."
        ) from error
    if validated != document:
        raise RiskIntegrationError(
            "invalid_language_evidence", "Language evidence integrity validation failed."
        )
    return validated


def _reference(
    kind: RiskEvidenceKind,
    document: AcousticEvidenceDocument | SpeechEvidenceDocument | LanguageEvidenceDocument,
) -> RiskEvidenceReference:
    return RiskEvidenceReference(
        kind=kind,
        document_type=document.document_type,
        evidence_id=document.evidence_id,
        evidence_sha256=_artifact_hash(document),
    )


def _integrate_acoustic(
    source: RiskSource,
    document: AcousticEvidenceDocument | None,
) -> AcousticRiskInputs:
    if document is None:
        return AcousticRiskInputs(status=RiskInputStatus.MISSING)
    evidence = _validated_acoustic(document)
    if (
        evidence.source.clip_id != source.clip_id
        or evidence.sample_rate_hz != source.sample_rate_hz
        or any(window.end_sample > source.num_samples for window in evidence.windows)
        or any(event.end_sample > source.num_samples for event in evidence.events)
    ):
        raise RiskIntegrationError(
            "source_mismatch", "Acoustic evidence does not match the risk source."
        )
    signals = tuple(
        AcousticRiskSignal(
            label=event.label,
            start_sample=event.start_sample,
            end_sample=event.end_sample,
            start_seconds=event.start_seconds,
            end_seconds=event.end_seconds,
            peak_score=event.peak_score,
            source_event_index=index,
        )
        for index, event in enumerate(evidence.events)
    )
    return AcousticRiskInputs(
        status=RiskInputStatus.PRESENT,
        evidence=_reference(RiskEvidenceKind.ACOUSTIC, evidence),
        event_count=len(signals),
        max_peak_score=max((signal.peak_score for signal in signals), default=None),
        signals=signals,
    )


def _check_speech_source(source: RiskSource, evidence: SpeechEvidenceDocument) -> None:
    recorded = evidence.source
    if (
        source.processing_scope is not ProcessingScope.ACOUSTIC_AND_SPEECH
        or recorded.clip_id != source.clip_id
        or recorded.consent_id != source.consent_id
        or recorded.sample_rate_hz != source.sample_rate_hz
        or recorded.num_samples != source.num_samples
    ):
        raise RiskIntegrationError(
            "source_mismatch", "Speech evidence does not match the risk source."
        )


def _check_cross_branch_source(
    acoustic: AcousticEvidenceDocument | None,
    speech: SpeechEvidenceDocument,
) -> None:
    if acoustic is None:
        return
    if (
        acoustic.source.preparation_manifest_path
        != speech.source.preparation_manifest_path
        or acoustic.source.preparation_manifest_sha256
        != speech.source.preparation_manifest_sha256
        or acoustic.source.raw_audio_sha256 != speech.source.raw_audio_sha256
    ):
        raise RiskIntegrationError(
            "source_mismatch",
            "Acoustic and speech evidence do not share the same prepared source.",
        )


def _accepted_segments(
    evidence: SpeechEvidenceDocument,
) -> tuple[SpeechSegmentEvidence, ...]:
    return tuple(
        segment
        for segment in evidence.segments
        if segment.assessment.reliability is TranscriptReliability.ACCEPTED
        and segment.assessment.downstream_text_allowed
        and segment.transcript is not None
    )


def _integrate_speech(evidence: SpeechEvidenceDocument) -> SpeechRiskInputs:
    accepted_count = len(_accepted_segments(evidence))
    review_count = sum(
        segment.assessment.reliability is TranscriptReliability.REVIEW_REQUIRED
        for segment in evidence.segments
    )
    return SpeechRiskInputs(
        status=RiskInputStatus.PRESENT,
        evidence=_reference(RiskEvidenceKind.SPEECH, evidence),
        segment_count=evidence.segment_count,
        accepted_transcript_count=accepted_count,
        review_required_transcript_count=review_count,
        max_vad_score=max(
            (segment.vad_score for segment in evidence.segments), default=None
        ),
    )


def _check_language_source(
    source: RiskSource,
    speech: SpeechEvidenceDocument,
    language: LanguageEvidenceDocument,
) -> None:
    recorded = language.source
    if (
        recorded.speech_evidence_id != speech.evidence_id
        or recorded.speech_evidence_sha256 != _artifact_hash(speech)
        or recorded.clip_id != source.clip_id
        or recorded.consent_id != source.consent_id
        or recorded.sample_rate_hz != source.sample_rate_hz
        or recorded.num_samples != source.num_samples
    ):
        raise RiskIntegrationError(
            "provenance_mismatch",
            "Language evidence does not reference the supplied speech evidence.",
        )


def _check_accepted_transcripts(
    speech: SpeechEvidenceDocument,
    language: LanguageEvidenceDocument,
) -> None:
    accepted = _accepted_segments(speech)
    if len(language.analyses) != len(accepted):
        raise RiskIntegrationError(
            "transcript_mismatch",
            "Language evidence does not cover the accepted speech transcripts.",
        )
    for segment, analysis in zip(accepted, language.analyses, strict=True):
        transcript = segment.transcript
        assert transcript is not None
        reference = analysis.transcript
        if (
            reference.segment_id != segment.segment_id
            or reference.start_sample != segment.start_sample
            or reference.end_sample != segment.end_sample
            or reference.start_seconds != segment.start_seconds
            or reference.end_seconds != segment.end_seconds
            or reference.transcript_sha256
            != hashlib.sha256(transcript.text.encode("utf-8")).hexdigest()
            or reference.transcript_character_count != len(transcript.text)
            or reference.transcript_utf8_bytes != len(transcript.text.encode("utf-8"))
            or reference.confidence_score != transcript.confidence_score
            or reference.confidence_kind is not transcript.confidence_kind
        ):
            raise RiskIntegrationError(
                "transcript_mismatch",
                "Language evidence differs from an accepted speech transcript.",
            )


def _integrate_language(language: LanguageEvidenceDocument) -> LanguageRiskInputs:
    signals = tuple(
        sorted(
            (
                LanguageRiskSignal(
                    finding_id=finding.finding_id,
                    segment_id=analysis.transcript.segment_id,
                    category=finding.category,
                    reason_codes=finding.reason_codes,
                    start_sample=analysis.transcript.start_sample,
                    end_sample=analysis.transcript.end_sample,
                    rule_match_count=len(finding.matches),
                )
                for analysis in language.analyses
                for finding in analysis.findings
            ),
            key=lambda signal: (
                signal.start_sample,
                signal.end_sample,
                signal.finding_id,
            ),
        )
    )
    return LanguageRiskInputs(
        status=RiskInputStatus.PRESENT,
        evidence=_reference(RiskEvidenceKind.LANGUAGE, language),
        finding_count=len(signals),
        signals=signals,
    )


def integrate_risk_inputs(
    source: RiskSource,
    *,
    acoustic: AcousticEvidenceDocument | None = None,
    speech: SpeechEvidenceDocument | None = None,
    language: LanguageEvidenceDocument | None = None,
) -> RiskInputSet:
    """Build privacy-minimized A6.1 inputs from validated Phase 3-5 evidence."""

    validated_source = _validated_source(source)
    acoustic_inputs = _integrate_acoustic(validated_source, acoustic)
    validated_acoustic = None if acoustic is None else _validated_acoustic(acoustic)

    if validated_source.processing_scope is ProcessingScope.ACOUSTIC_ONLY:
        if speech is not None or language is not None:
            raise RiskIntegrationError(
                "evidence_not_permitted",
                "Speech or language evidence is not permitted for this risk source.",
            )
        return RiskInputSet(
            source=validated_source,
            acoustic=acoustic_inputs,
            speech=SpeechRiskInputs(status=RiskInputStatus.NOT_PERMITTED),
            language=LanguageRiskInputs(status=RiskInputStatus.NOT_PERMITTED),
        )

    if speech is None:
        if language is not None:
            raise RiskIntegrationError(
                "missing_speech_evidence",
                "Language evidence cannot be integrated without speech evidence.",
            )
        return RiskInputSet(
            source=validated_source,
            acoustic=acoustic_inputs,
            speech=SpeechRiskInputs(status=RiskInputStatus.MISSING),
            language=LanguageRiskInputs(status=RiskInputStatus.MISSING),
        )

    validated_speech = _validated_speech(speech)
    _check_speech_source(validated_source, validated_speech)
    _check_cross_branch_source(validated_acoustic, validated_speech)
    speech_inputs = _integrate_speech(validated_speech)
    accepted_count = speech_inputs.accepted_transcript_count

    if language is None:
        language_status = (
            RiskInputStatus.NO_ACCEPTED_TEXT
            if accepted_count == 0
            else RiskInputStatus.MISSING
        )
        language_inputs = LanguageRiskInputs(status=language_status)
    else:
        validated_language = _validated_language(language)
        _check_language_source(
            validated_source, validated_speech, validated_language
        )
        _check_accepted_transcripts(validated_speech, validated_language)
        language_inputs = (
            LanguageRiskInputs(status=RiskInputStatus.NO_ACCEPTED_TEXT)
            if accepted_count == 0
            else _integrate_language(validated_language)
        )

    return RiskInputSet(
        source=validated_source,
        acoustic=acoustic_inputs,
        speech=speech_inputs,
        language=language_inputs,
    )
