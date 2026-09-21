"""A5.2 deterministic analysis of accepted A4.3 transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

from audio_sentinel.language_contracts import (
    AcceptedTranscriptReference,
    LanguageCategory,
    LanguageEvidenceDocument,
    LanguageEvidenceSource,
    LanguageFinding,
    LanguageReasonCode,
    LanguageRuleKind,
    LanguageRuleMatch,
    LanguageTranscriptAnalysis,
)
from audio_sentinel.language_rules import (
    MAX_LANGUAGE_RULE_SET_BYTES,
    LanguageRule,
    LanguageRuleSet,
    LoadedLanguageRuleSet,
    load_builtin_language_rule_set,
)
from audio_sentinel.speech_contracts import (
    SpeechEvidenceDocument,
    TranscriptReliability,
)
from audio_sentinel.speech_transcription import (
    DownstreamTranscript,
    SpeechTranscriptionResult,
)


_TOKEN_SCAN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_HARD_SCOPE_BOUNDARIES = frozenset(".!?;:\n")
_CATEGORY_ORDER = {category: index for index, category in enumerate(LanguageCategory)}


class LanguageAnalysisError(RuntimeError):
    """Stable analysis failure code plus a safe, non-sensitive explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class LanguageAnalysisSettings(BaseModel):
    """Resource limits for deterministic in-memory transcript matching."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_transcripts: int = Field(default=100_000, gt=0, strict=True)
    max_total_text_bytes: int = Field(default=1_048_576, gt=0, strict=True)
    max_total_tokens: int = Field(default=250_000, gt=0, strict=True)
    max_rule_checks: int = Field(default=10_000_000, gt=0, strict=True)
    max_findings: int = Field(default=100_000, gt=0, strict=True)


@dataclass(frozen=True)
class LanguageAnalysisResult:
    """Portable evidence plus the immutable settings used to create it."""

    evidence: LanguageEvidenceDocument
    settings: LanguageAnalysisSettings

    def to_summary(self) -> dict[str, object]:
        """Return JSON-ready evidence without duplicating transcript text."""

        return {
            "evidence": self.evidence.model_dump(mode="json"),
            "settings": self.settings.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class _Token:
    value: str
    start_character: int
    end_character: int


@dataclass(frozen=True)
class _Candidate:
    rule: LanguageRule
    start_token: int
    end_token: int
    match: LanguageRuleMatch


@dataclass
class _Budget:
    rule_checks: int = 0
    findings: int = 0


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise LanguageAnalysisError(
            "invalid_time", "Language analysis time must be timezone-aware."
        )
    return value.astimezone(UTC)


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _speech_identity_payload(evidence: SpeechEvidenceDocument) -> dict[str, object]:
    return {
        "source": evidence.source.model_dump(mode="json"),
        "reliability_policy": evidence.reliability_policy.model_dump(mode="json"),
        "vad_model": evidence.vad_model.model_dump(mode="json"),
        "transcription_model": (
            None
            if evidence.transcription_model is None
            else evidence.transcription_model.model_dump(mode="json")
        ),
        "input_window_count": evidence.input_window_count,
        "segment_count": evidence.segment_count,
        "transcribed_segment_count": evidence.transcribed_segment_count,
        "windows": [item.model_dump(mode="json") for item in evidence.windows],
        "segments": [item.model_dump(mode="json") for item in evidence.segments],
    }


def _validate_speech_input(
    speech: SpeechTranscriptionResult,
) -> tuple[SpeechEvidenceDocument, tuple[DownstreamTranscript, ...]]:
    if not isinstance(speech, SpeechTranscriptionResult):
        raise LanguageAnalysisError(
            "invalid_speech_input", "A4.3 speech transcription output is required."
        )
    try:
        evidence = SpeechEvidenceDocument.model_validate(
            speech.evidence.model_dump(mode="python")
        )
    except Exception as error:
        raise LanguageAnalysisError(
            "invalid_speech_input", "Speech evidence failed contract validation."
        ) from error
    if (
        evidence != speech.evidence
        or evidence.evidence_id != _canonical_hash(_speech_identity_payload(evidence))
    ):
        raise LanguageAnalysisError(
            "invalid_speech_input", "Speech evidence integrity validation failed."
        )

    expected = tuple(
        DownstreamTranscript(
            segment_id=segment.segment_id,
            start_sample=segment.start_sample,
            end_sample=segment.end_sample,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            text=segment.transcript.text,
            confidence_score=segment.transcript.confidence_score,
            confidence_kind=segment.transcript.confidence_kind,
        )
        for segment in evidence.segments
        if segment.assessment.reliability is TranscriptReliability.ACCEPTED
        and segment.assessment.downstream_text_allowed
        and segment.transcript is not None
    )
    if not isinstance(speech.downstream_transcripts, tuple) or any(
        not isinstance(item, DownstreamTranscript)
        for item in speech.downstream_transcripts
    ):
        raise LanguageAnalysisError(
            "invalid_speech_input", "Accepted transcript handoff has an invalid shape."
        )
    if speech.downstream_transcripts != expected:
        raise LanguageAnalysisError(
            "invalid_speech_input",
            "Accepted transcript handoff differs from verified speech evidence.",
        )
    return evidence, expected


def _normalized_tokens(text: str) -> tuple[_Token, ...]:
    normalized: list[str] = []
    source_spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        cluster = unicodedata.normalize("NFKC", text[index:end]).casefold()
        cluster = cluster.replace("\u2018", "'").replace("\u2019", "'")
        normalized.extend(cluster)
        source_spans.extend((index, end) for _ in cluster)
        index = end

    normalized_text = "".join(normalized)
    tokens: list[_Token] = []
    for match in _TOKEN_SCAN_RE.finditer(normalized_text):
        start, end = match.span()
        tokens.append(
            _Token(
                value=match.group(),
                start_character=source_spans[start][0],
                end_character=source_spans[end - 1][1],
            )
        )
    return tuple(tokens)


def _rule_index(
    loaded: LoadedLanguageRuleSet,
) -> dict[str, tuple[LanguageRule, ...]]:
    indexed: dict[str, list[LanguageRule]] = {}
    for rule in loaded.rule_set.rules:
        indexed.setdefault(rule.tokens[0], []).append(rule)
    return {
        token: tuple(
            sorted(
                rules,
                key=lambda rule: (-len(rule.tokens), rule.kind.value, rule.rule_id),
            )
        )
        for token, rules in indexed.items()
    }


def _validate_rule_set(loaded: LoadedLanguageRuleSet) -> LoadedLanguageRuleSet:
    if not isinstance(loaded, LoadedLanguageRuleSet):
        raise ValueError("loaded rule set has an invalid type")
    validated = LanguageRuleSet.model_validate(loaded.rule_set.model_dump(mode="python"))
    if (
        validated != loaded.rule_set
        or not re.fullmatch(r"[a-f0-9]{64}", loaded.artifact_sha256)
        or not isinstance(loaded.artifact_size_bytes, int)
        or isinstance(loaded.artifact_size_bytes, bool)
        or loaded.artifact_size_bytes <= 0
        or loaded.artifact_size_bytes > MAX_LANGUAGE_RULE_SET_BYTES
    ):
        raise ValueError("loaded rule set failed integrity validation")
    return LoadedLanguageRuleSet(
        rule_set=validated,
        artifact_sha256=loaded.artifact_sha256,
        artifact_size_bytes=loaded.artifact_size_bytes,
    )


def _match_candidates(
    text: str,
    tokens: tuple[_Token, ...],
    indexed_rules: dict[str, tuple[LanguageRule, ...]],
    settings: LanguageAnalysisSettings,
    budget: _Budget,
) -> tuple[tuple[_Candidate, ...], tuple[_Candidate, ...]]:
    active: list[_Candidate] = []
    negations: list[_Candidate] = []
    token_values = tuple(token.value for token in tokens)
    for start_token, token in enumerate(tokens):
        for rule in indexed_rules.get(token.value, ()):
            budget.rule_checks += 1
            if budget.rule_checks > settings.max_rule_checks:
                raise LanguageAnalysisError(
                    "analysis_too_large", "Language rule checks exceed the configured limit."
                )
            end_token = start_token + len(rule.tokens)
            if end_token > len(tokens) or token_values[start_token:end_token] != rule.tokens:
                continue
            start_character = token.start_character
            end_character = tokens[end_token - 1].end_character
            candidate = _Candidate(
                rule=rule,
                start_token=start_token,
                end_token=end_token,
                match=LanguageRuleMatch(
                    rule_id=rule.rule_id,
                    rule_kind=rule.kind,
                    start_character=start_character,
                    end_character=end_character,
                    matched_text_sha256=hashlib.sha256(
                        text[start_character:end_character].encode("utf-8")
                    ).hexdigest(),
                ),
            )
            if rule.kind is LanguageRuleKind.NEGATION:
                negations.append(candidate)
            else:
                active.append(candidate)
    return tuple(active), tuple(negations)


def _prefer_specific_matches(candidates: tuple[_Candidate, ...]) -> tuple[_Candidate, ...]:
    """Drop shorter same-category matches fully contained by a longer match."""

    preferred: list[_Candidate] = []
    ranked = sorted(
        candidates,
        key=lambda item: (
            -(item.end_token - item.start_token),
            item.start_token,
            item.end_token,
            item.rule.rule_id,
        ),
    )
    for candidate in ranked:
        if any(
            existing.rule.category is candidate.rule.category
            and existing.start_token <= candidate.start_token
            and existing.end_token >= candidate.end_token
            for existing in preferred
        ):
            continue
        preferred.append(candidate)
    return tuple(
        sorted(
            preferred,
            key=lambda item: (
                item.match.start_character,
                item.match.end_character,
                _CATEGORY_ORDER[item.rule.category],
                item.rule.rule_id,
            ),
        )
    )


def _applicable_negation(
    text: str,
    candidate: _Candidate,
    negations: tuple[_Candidate, ...],
    window: int,
) -> _Candidate | None:
    eligible = []
    for negation in negations:
        intervening_tokens = candidate.start_token - negation.end_token
        if intervening_tokens < 0 or intervening_tokens > window:
            continue
        between = text[
            negation.match.end_character : candidate.match.start_character
        ]
        if any(character in _HARD_SCOPE_BOUNDARIES for character in between):
            continue
        eligible.append(negation)
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            candidate.start_token - item.end_token,
            -(item.end_token - item.start_token),
            item.rule.rule_id,
        ),
    )


def _finding_id(
    segment_id: str,
    category: LanguageCategory,
    reason_codes: tuple[LanguageReasonCode, ...],
    matches: tuple[LanguageRuleMatch, ...],
) -> str:
    return _canonical_hash(
        {
            "segment_id": segment_id,
            "category": category.value,
            "reason_codes": [reason.value for reason in reason_codes],
            "matches": [match.model_dump(mode="json") for match in matches],
        }
    )


def _analyze_transcript(
    transcript: DownstreamTranscript,
    active: tuple[_Candidate, ...],
    negations: tuple[_Candidate, ...],
    loaded: LoadedLanguageRuleSet,
    settings: LanguageAnalysisSettings,
    budget: _Budget,
) -> LanguageTranscriptAnalysis:
    reference = AcceptedTranscriptReference(
        segment_id=transcript.segment_id,
        start_sample=transcript.start_sample,
        end_sample=transcript.end_sample,
        start_seconds=transcript.start_seconds,
        end_seconds=transcript.end_seconds,
        transcript_sha256=hashlib.sha256(transcript.text.encode("utf-8")).hexdigest(),
        transcript_character_count=len(transcript.text),
        transcript_utf8_bytes=len(transcript.text.encode("utf-8")),
        confidence_score=transcript.confidence_score,
        confidence_kind=transcript.confidence_kind,
    )
    findings: list[LanguageFinding] = []
    for candidate in _prefer_specific_matches(active):
        negation = _applicable_negation(
            transcript.text,
            candidate,
            negations,
            loaded.rule_set.negation_window_tokens,
        )
        primary_reason = candidate.rule.reason_code
        if negation is None:
            category = candidate.rule.category
            reason_codes = (primary_reason,)
            matches = (candidate.match,)
        else:
            category = LanguageCategory.CONTEXT_SUPPRESSED
            reason_codes = (primary_reason, LanguageReasonCode.EXPLICIT_NEGATION)
            matches = tuple(
                sorted(
                    (negation.match, candidate.match),
                    key=lambda match: (
                        match.start_character,
                        match.end_character,
                        match.rule_id,
                        match.rule_kind.value,
                    ),
                )
            )
        assert category is not None
        findings.append(
            LanguageFinding(
                finding_id=_finding_id(
                    transcript.segment_id, category, reason_codes, matches
                ),
                category=category,
                reason_codes=reason_codes,
                matches=matches,
            )
        )

    if not findings:
        reason_codes = (LanguageReasonCode.NO_RULE_MATCH,)
        findings.append(
            LanguageFinding(
                finding_id=_finding_id(
                    transcript.segment_id,
                    LanguageCategory.NO_CONCERNING_MATCH,
                    reason_codes,
                    (),
                ),
                category=LanguageCategory.NO_CONCERNING_MATCH,
                reason_codes=reason_codes,
            )
        )
    budget.findings += len(findings)
    if budget.findings > settings.max_findings:
        raise LanguageAnalysisError(
            "analysis_too_large", "Language findings exceed the configured limit."
        )
    ordered = tuple(
        sorted(
            findings,
            key=lambda finding: (
                finding.matches[0].start_character if finding.matches else len(transcript.text),
                _CATEGORY_ORDER[finding.category],
                finding.finding_id,
            ),
        )
    )
    return LanguageTranscriptAnalysis(transcript=reference, findings=ordered)


def analyze_accepted_transcripts(
    speech: SpeechTranscriptionResult,
    *,
    rule_set: LoadedLanguageRuleSet | None = None,
    settings: LanguageAnalysisSettings | None = None,
    now: datetime | None = None,
) -> LanguageAnalysisResult:
    """Analyze the exact accepted-only A4.3 handoff using versioned local rules."""

    created_at = _clock(now)
    analysis_settings = settings or LanguageAnalysisSettings()
    evidence, transcripts = _validate_speech_input(speech)
    if len(transcripts) > analysis_settings.max_transcripts:
        raise LanguageAnalysisError(
            "too_many_transcripts", "Accepted transcript count exceeds the configured limit."
        )
    total_text_bytes = sum(len(item.text.encode("utf-8")) for item in transcripts)
    if total_text_bytes > analysis_settings.max_total_text_bytes:
        raise LanguageAnalysisError(
            "input_too_large", "Accepted transcript text exceeds the configured byte limit."
        )

    try:
        loaded = _validate_rule_set(rule_set or load_builtin_language_rule_set())
        descriptor = loaded.as_descriptor()
    except Exception as error:
        raise LanguageAnalysisError(
            "invalid_rule_set", "Language rule data failed integrity validation."
        ) from error
    indexed_rules = _rule_index(loaded)
    budget = _Budget()
    analyses: list[LanguageTranscriptAnalysis] = []
    total_tokens = 0
    for transcript in transcripts:
        tokens = _normalized_tokens(transcript.text)
        total_tokens += len(tokens)
        if total_tokens > analysis_settings.max_total_tokens:
            raise LanguageAnalysisError(
                "input_too_large", "Accepted transcript tokens exceed the configured limit."
            )
        active, negations = _match_candidates(
            transcript.text,
            tokens,
            indexed_rules,
            analysis_settings,
            budget,
        )
        analyses.append(
            _analyze_transcript(
                transcript,
                active,
                negations,
                loaded,
                analysis_settings,
                budget,
            )
        )

    source = LanguageEvidenceSource(
        speech_evidence_id=evidence.evidence_id,
        speech_evidence_sha256=_canonical_hash(evidence.model_dump(mode="json")),
        clip_id=evidence.source.clip_id,
        consent_id=evidence.source.consent_id,
        sample_rate_hz=evidence.source.sample_rate_hz,
        num_samples=evidence.source.num_samples,
    )
    identity_payload = {
        "source": source.model_dump(mode="json"),
        "analysis_method": "versioned_rules",
        "rule_set": descriptor.model_dump(mode="json"),
        "input_transcript_count": len(analyses),
        "analyzed_transcript_count": len(analyses),
        "finding_count": budget.findings,
        "analyses": [item.model_dump(mode="json") for item in analyses],
    }
    language_evidence = LanguageEvidenceDocument(
        evidence_id=_canonical_hash(identity_payload),
        created_at=created_at,
        source=source,
        rule_set=descriptor,
        input_transcript_count=len(analyses),
        analyzed_transcript_count=len(analyses),
        finding_count=budget.findings,
        analyses=tuple(analyses),
    )
    return LanguageAnalysisResult(evidence=language_evidence, settings=analysis_settings)
