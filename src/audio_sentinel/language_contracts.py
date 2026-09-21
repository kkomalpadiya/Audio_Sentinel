"""A5.1 versioned, explainable language-evidence contract."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_sentinel.preparation import Identifier
from audio_sentinel.speech_contracts import TranscriptConfidenceKind


LANGUAGE_EVIDENCE_SCHEMA_VERSION = "1.0"
LANGUAGE_RULE_FORMAT_VERSION = "1.0"


class LanguageCategory(str, Enum):
    """Limited evidence categories; none is an incident or risk decision."""

    NO_CONCERNING_MATCH = "no_concerning_match"
    DISTRESS = "distress"
    THREAT = "threat"
    WEAPON_REFERENCE = "weapon_reference"
    AMBIGUOUS = "ambiguous"
    CONTEXT_SUPPRESSED = "context_suppressed"


class LanguageReasonCode(str, Enum):
    """Portable explanations for why a language category was recorded."""

    NO_RULE_MATCH = "no_rule_match"
    KEYWORD_MATCH = "keyword_match"
    PHRASE_MATCH = "phrase_match"
    EXPLICIT_NEGATION = "explicit_negation"
    HYPOTHETICAL_OR_CONDITIONAL = "hypothetical_or_conditional"
    QUOTED_OR_REPORTED_SPEECH = "quoted_or_reported_speech"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    CONFLICTING_SIGNALS = "conflicting_signals"


class LanguageRuleKind(str, Enum):
    KEYWORD = "keyword"
    PHRASE = "phrase"
    NEGATION = "negation"


_MATCH_REASONS = frozenset(
    {LanguageReasonCode.KEYWORD_MATCH, LanguageReasonCode.PHRASE_MATCH}
)
_SUPPRESSION_REASONS = frozenset(
    {
        LanguageReasonCode.EXPLICIT_NEGATION,
        LanguageReasonCode.HYPOTHETICAL_OR_CONDITIONAL,
        LanguageReasonCode.QUOTED_OR_REPORTED_SPEECH,
    }
)
_AMBIGUITY_REASONS = frozenset(
    {LanguageReasonCode.INSUFFICIENT_CONTEXT, LanguageReasonCode.CONFLICTING_SIGNALS}
)
_REASON_ORDER = {reason: index for index, reason in enumerate(LanguageReasonCode)}
_CATEGORY_ORDER = {category: index for index, category in enumerate(LanguageCategory)}


class LanguageContractRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class LanguageEvidenceSource(LanguageContractRecord):
    """Privacy-minimized link to accepted Phase 4 speech evidence."""

    speech_evidence_id: Identifier
    speech_evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    clip_id: Identifier
    consent_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    processing_scope: Literal["acoustic_and_speech"] = "acoustic_and_speech"
    sample_rate_hz: int = Field(ge=8_000, le=192_000, strict=True)
    num_samples: int = Field(gt=0, strict=True)


class LanguageRuleSetDescriptor(LanguageContractRecord):
    """Exact versioned rule artifact used to produce the findings."""

    rule_set_id: Identifier
    rule_set_version: str = Field(min_length=1, max_length=128)
    rule_format_version: Literal["1.0"] = LANGUAGE_RULE_FORMAT_VERSION
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class AcceptedTranscriptReference(LanguageContractRecord):
    """Accepted-only A4.3 handoff without duplicating the transcript text."""

    segment_id: Identifier
    start_sample: int = Field(ge=0, strict=True)
    end_sample: int = Field(gt=0, strict=True)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    transcript_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    transcript_character_count: int = Field(gt=0, le=20_000, strict=True)
    transcript_utf8_bytes: int = Field(gt=0, le=80_000, strict=True)
    confidence_score: float = Field(ge=0, le=1)
    confidence_kind: TranscriptConfidenceKind
    transcript_reliability: Literal["accepted"] = "accepted"
    downstream_text_allowed: Literal[True] = True

    @model_validator(mode="after")
    def validate_span(self) -> "AcceptedTranscriptReference":
        if self.end_sample <= self.start_sample:
            raise ValueError("transcript end_sample must be greater than start_sample")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("transcript end_seconds must be greater than start_seconds")
        if self.transcript_utf8_bytes < self.transcript_character_count:
            raise ValueError("transcript_utf8_bytes cannot be below character count")
        return self


class LanguageRuleMatch(LanguageContractRecord):
    """One rule hit located in the accepted transcript without copying its text."""

    rule_id: Identifier
    rule_kind: LanguageRuleKind
    start_character: int = Field(ge=0, strict=True)
    end_character: int = Field(gt=0, strict=True)
    matched_text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_span(self) -> "LanguageRuleMatch":
        if self.end_character <= self.start_character:
            raise ValueError("rule-match end_character must be greater than start_character")
        return self


class LanguageFinding(LanguageContractRecord):
    """One explainable language category supported by ordered rule matches."""

    finding_id: Identifier
    category: LanguageCategory
    reason_codes: tuple[LanguageReasonCode, ...] = Field(min_length=1)
    matches: tuple[LanguageRuleMatch, ...] = ()

    @model_validator(mode="after")
    def validate_explanation(self) -> "LanguageFinding":
        reason_set = set(self.reason_codes)
        if len(reason_set) != len(self.reason_codes):
            raise ValueError("language reason_codes must be unique")
        if list(self.reason_codes) != sorted(
            self.reason_codes, key=lambda reason: _REASON_ORDER[reason]
        ):
            raise ValueError("language reason_codes must use canonical ordering")
        if list(self.matches) != sorted(
            self.matches,
            key=lambda match: (
                match.start_character,
                match.end_character,
                match.rule_id,
                match.rule_kind.value,
            ),
        ):
            raise ValueError("language rule matches must use deterministic ordering")

        match_kinds = {match.rule_kind for match in self.matches}
        has_primary_reason = bool(reason_set & _MATCH_REASONS)
        if LanguageReasonCode.KEYWORD_MATCH in reason_set:
            if LanguageRuleKind.KEYWORD not in match_kinds:
                raise ValueError("keyword_match reason requires a keyword rule match")
        if LanguageReasonCode.PHRASE_MATCH in reason_set:
            if LanguageRuleKind.PHRASE not in match_kinds:
                raise ValueError("phrase_match reason requires a phrase rule match")

        if self.category is LanguageCategory.NO_CONCERNING_MATCH:
            if reason_set != {LanguageReasonCode.NO_RULE_MATCH} or self.matches:
                raise ValueError(
                    "no_concerning_match requires only no_rule_match and no rule matches"
                )
            return self

        if LanguageReasonCode.NO_RULE_MATCH in reason_set:
            raise ValueError("no_rule_match is reserved for no_concerning_match")
        if not has_primary_reason or not self.matches:
            raise ValueError("language findings require a keyword or phrase match")

        if self.category in {
            LanguageCategory.DISTRESS,
            LanguageCategory.THREAT,
            LanguageCategory.WEAPON_REFERENCE,
        }:
            if reason_set - _MATCH_REASONS:
                raise ValueError("active language categories allow only match reason codes")
        elif self.category is LanguageCategory.AMBIGUOUS:
            if not reason_set & _AMBIGUITY_REASONS:
                raise ValueError("ambiguous category requires an ambiguity reason code")
            if reason_set - (_MATCH_REASONS | _AMBIGUITY_REASONS):
                raise ValueError("ambiguous category has an incompatible reason code")
        elif self.category is LanguageCategory.CONTEXT_SUPPRESSED:
            if not reason_set & _SUPPRESSION_REASONS:
                raise ValueError("context_suppressed requires a suppression reason code")
            if reason_set - (_MATCH_REASONS | _SUPPRESSION_REASONS):
                raise ValueError("context_suppressed has an incompatible reason code")
            if LanguageReasonCode.EXPLICIT_NEGATION in reason_set:
                if LanguageRuleKind.NEGATION not in match_kinds:
                    raise ValueError("explicit_negation requires a negation rule match")
        return self


class LanguageTranscriptAnalysis(LanguageContractRecord):
    """Complete language findings for one accepted downstream transcript."""

    transcript: AcceptedTranscriptReference
    findings: tuple[LanguageFinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_findings(self) -> "LanguageTranscriptAnalysis":
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("language finding IDs must be unique within a transcript")

        no_match = [
            finding
            for finding in self.findings
            if finding.category is LanguageCategory.NO_CONCERNING_MATCH
        ]
        if no_match and len(self.findings) != 1:
            raise ValueError("no_concerning_match must be the only transcript finding")

        for finding in self.findings:
            for match in finding.matches:
                if match.end_character > self.transcript.transcript_character_count:
                    raise ValueError("rule-match span cannot exceed the accepted transcript")

        def finding_key(finding: LanguageFinding) -> tuple[int, int, str]:
            first_character = (
                finding.matches[0].start_character
                if finding.matches
                else self.transcript.transcript_character_count
            )
            return first_character, _CATEGORY_ORDER[finding.category], finding.finding_id

        if list(self.findings) != sorted(self.findings, key=finding_key):
            raise ValueError("language findings must use deterministic ordering")
        return self


class LanguageEvidenceDocument(LanguageContractRecord):
    """Explainable text evidence; deliberately not an incident or risk contract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/language-evidence.schema.json"
        },
    )

    schema_version: Literal["1.0"] = LANGUAGE_EVIDENCE_SCHEMA_VERSION
    document_type: Literal["language_evidence"] = "language_evidence"
    evidence_id: Identifier
    created_at: datetime
    source: LanguageEvidenceSource
    analysis_method: Literal["versioned_rules"] = "versioned_rules"
    rule_set: LanguageRuleSetDescriptor
    input_transcript_count: int = Field(ge=0, strict=True)
    analyzed_transcript_count: int = Field(ge=0, strict=True)
    finding_count: int = Field(ge=0, strict=True)
    analyses: tuple[LanguageTranscriptAnalysis, ...]

    @model_validator(mode="after")
    def validate_document(self) -> "LanguageEvidenceDocument":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.input_transcript_count != len(self.analyses):
            raise ValueError("input_transcript_count must equal the accepted transcript inventory")
        if self.analyzed_transcript_count != len(self.analyses):
            raise ValueError("analyzed_transcript_count must equal the analysis inventory")
        if self.finding_count != sum(len(item.findings) for item in self.analyses):
            raise ValueError("finding_count must equal the complete finding inventory")

        segment_ids = [item.transcript.segment_id for item in self.analyses]
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("analyzed transcript segment IDs must be unique")
        if list(self.analyses) != sorted(
            self.analyses,
            key=lambda item: (
                item.transcript.start_sample,
                item.transcript.end_sample,
                item.transcript.segment_id,
            ),
        ):
            raise ValueError("transcript analyses must use deterministic temporal ordering")

        previous_end = 0
        for item in self.analyses:
            transcript = item.transcript
            if transcript.end_sample > self.source.num_samples:
                raise ValueError("transcript span cannot exceed the prepared source")
            if transcript.start_sample < previous_end:
                raise ValueError("accepted transcript spans must not overlap")
            previous_end = transcript.end_sample
            self._validate_seconds(transcript.start_sample, transcript.start_seconds)
            self._validate_seconds(transcript.end_sample, transcript.end_seconds)
        return self

    def _validate_seconds(self, sample: int, seconds: float) -> None:
        expected = sample / self.source.sample_rate_hz
        if not math.isclose(seconds, expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError("transcript seconds must equal sample offsets / sample_rate_hz")


def language_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public JSON Schema for the v1 language-evidence contract."""

    return {"language-evidence.schema.json": LanguageEvidenceDocument.model_json_schema()}


def write_language_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the public language contract for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in language_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
