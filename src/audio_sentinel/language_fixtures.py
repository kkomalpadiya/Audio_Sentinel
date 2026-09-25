"""B5.2 versioned, labeled fixtures for language-analysis evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from importlib.resources import files
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.language_contracts import (
    LANGUAGE_EVIDENCE_SCHEMA_VERSION,
    LanguageCategory,
    LanguageReasonCode,
    LanguageRuleKind,
    LanguageRuleSetDescriptor,
)
from audio_sentinel.language_rules import (
    LoadedLanguageRuleSet,
    load_builtin_language_rule_set,
)
from audio_sentinel.preparation import Identifier


LANGUAGE_FIXTURE_SET_SCHEMA_VERSION = "1.0"
BUILTIN_LANGUAGE_FIXTURE_SET_ID = "audio-sentinel-language-fixtures-en-v1"
BUILTIN_LANGUAGE_FIXTURE_SET_VERSION = "1.0.0"
BUILTIN_LANGUAGE_FIXTURE_RESOURCE = "language-fixtures-en-v1.json"
BUILTIN_LANGUAGE_FIXTURE_SET_SHA256 = (
    "406c8e8506946eedd43c650e46e8dd9685ac7475847fd449987db457c45e78c0"
)
MAX_LANGUAGE_FIXTURE_SET_BYTES = 1_048_576


class LanguageFixtureKind(str, Enum):
    ACTIVE_INDICATOR = "active_indicator"
    NO_CONCERNING_MATCH = "no_concerning_match"
    CONTEXT_SUPPRESSED = "context_suppressed"
    AMBIGUOUS = "ambiguous"


_ACTIVE_CATEGORIES = frozenset(
    {
        LanguageCategory.DISTRESS,
        LanguageCategory.THREAT,
        LanguageCategory.WEAPON_REFERENCE,
    }
)
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


class LanguageFixtureRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class ExpectedLanguageFinding(LanguageFixtureRecord):
    """Expected portable labels and contributing rule IDs for one finding."""

    category: LanguageCategory
    reason_codes: tuple[LanguageReasonCode, ...] = Field(min_length=1)
    rule_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_expected_finding(self) -> "ExpectedLanguageFinding":
        reason_set = set(self.reason_codes)
        if len(reason_set) != len(self.reason_codes):
            raise ValueError("fixture reason_codes must be unique")
        if list(self.reason_codes) != sorted(
            self.reason_codes, key=lambda reason: _REASON_ORDER[reason]
        ):
            raise ValueError("fixture reason_codes must use canonical ordering")
        if len(set(self.rule_ids)) != len(self.rule_ids):
            raise ValueError("fixture rule_ids must be unique within a finding")

        has_match = bool(reason_set & _MATCH_REASONS)
        if self.category is LanguageCategory.NO_CONCERNING_MATCH:
            if reason_set != {LanguageReasonCode.NO_RULE_MATCH} or self.rule_ids:
                raise ValueError(
                    "no_concerning_match fixtures require no_rule_match and no rules"
                )
            return self
        if LanguageReasonCode.NO_RULE_MATCH in reason_set:
            raise ValueError("no_rule_match is reserved for no_concerning_match fixtures")
        if not has_match or not self.rule_ids:
            raise ValueError("non-empty fixture findings require a match reason and rule")

        if self.category in _ACTIVE_CATEGORIES:
            if reason_set - _MATCH_REASONS:
                raise ValueError("active fixture labels allow only match reasons")
        elif self.category is LanguageCategory.CONTEXT_SUPPRESSED:
            if not reason_set & _SUPPRESSION_REASONS:
                raise ValueError("suppressed fixtures require a suppression reason")
            if reason_set - (_MATCH_REASONS | _SUPPRESSION_REASONS):
                raise ValueError("suppressed fixture contains an incompatible reason")
        elif self.category is LanguageCategory.AMBIGUOUS:
            if not reason_set & _AMBIGUITY_REASONS:
                raise ValueError("ambiguous fixtures require an ambiguity reason")
            if reason_set - (_MATCH_REASONS | _AMBIGUITY_REASONS):
                raise ValueError("ambiguous fixture contains an incompatible reason")
        return self


class LanguageFixture(LanguageFixtureRecord):
    """One accepted English transcript and its expected language labels."""

    fixture_id: Identifier
    kind: LanguageFixtureKind
    text: str = Field(min_length=1, max_length=20_000)
    tags: tuple[Identifier, ...] = Field(min_length=1, max_length=16)
    expected_findings: tuple[ExpectedLanguageFinding, ...] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("fixture text must be nonblank and trimmed")
        if any(character in value for character in ("\x00", "\r")):
            raise ValueError("fixture text contains a disallowed control character")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or list(value) != sorted(value):
            raise ValueError("fixture tags must be unique and alphabetically ordered")
        return value

    @model_validator(mode="after")
    def validate_kind(self) -> "LanguageFixture":
        signatures = [
            (finding.category, finding.reason_codes, finding.rule_ids)
            for finding in self.expected_findings
        ]
        if len(set(signatures)) != len(signatures):
            raise ValueError("fixture expected findings must be unique")
        categories = {finding.category for finding in self.expected_findings}
        if self.kind is LanguageFixtureKind.NO_CONCERNING_MATCH:
            if categories != {LanguageCategory.NO_CONCERNING_MATCH} or len(
                self.expected_findings
            ) != 1:
                raise ValueError("no-match fixtures require one no-match finding")
        elif self.kind is LanguageFixtureKind.ACTIVE_INDICATOR:
            if not categories or categories - _ACTIVE_CATEGORIES:
                raise ValueError("active fixtures require only active categories")
        elif self.kind is LanguageFixtureKind.CONTEXT_SUPPRESSED:
            if categories != {LanguageCategory.CONTEXT_SUPPRESSED}:
                raise ValueError("suppressed fixtures require suppressed findings")
        elif self.kind is LanguageFixtureKind.AMBIGUOUS:
            if categories != {LanguageCategory.AMBIGUOUS}:
                raise ValueError("ambiguous fixtures require ambiguous findings")
        return self


class LanguageFixtureSet(LanguageFixtureRecord):
    """Complete versioned English fixture inventory for A5.3 evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/language-fixture-set.schema.json"
        },
    )

    schema_version: Literal["1.0"] = LANGUAGE_FIXTURE_SET_SCHEMA_VERSION
    fixture_set_id: Identifier
    fixture_set_version: str = Field(min_length=1, max_length=128)
    language: Literal["en"] = "en"
    label_contract_version: Literal["1.0"] = LANGUAGE_EVIDENCE_SCHEMA_VERSION
    rule_set: LanguageRuleSetDescriptor
    fixture_count: int = Field(ge=1, le=10_000, strict=True)
    expected_finding_count: int = Field(ge=1, le=100_000, strict=True)
    fixtures: tuple[LanguageFixture, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_inventory(self) -> "LanguageFixtureSet":
        if self.fixture_count != len(self.fixtures):
            raise ValueError("fixture_count must equal the fixture inventory")
        if self.expected_finding_count != sum(
            len(fixture.expected_findings) for fixture in self.fixtures
        ):
            raise ValueError("expected_finding_count must equal the finding inventory")
        fixture_ids = [fixture.fixture_id for fixture in self.fixtures]
        if len(set(fixture_ids)) != len(fixture_ids):
            raise ValueError("fixture IDs must be unique")
        if fixture_ids != sorted(fixture_ids):
            raise ValueError("fixtures must use canonical fixture-ID ordering")
        texts = [fixture.text for fixture in self.fixtures]
        if len(set(texts)) != len(texts):
            raise ValueError("fixture texts must be unique")
        if {fixture.kind for fixture in self.fixtures} != set(LanguageFixtureKind):
            raise ValueError("fixture inventory must cover every fixture kind")
        return self


@dataclass(frozen=True)
class LoadedLanguageFixtureSet:
    fixture_set: LanguageFixtureSet
    artifact_sha256: str
    artifact_size_bytes: int


def _validate_against_rules(
    fixture_set: LanguageFixtureSet,
    loaded_rules: LoadedLanguageRuleSet,
) -> None:
    if fixture_set.rule_set != loaded_rules.as_descriptor():
        raise ValueError("language fixture rule provenance differs from the loaded rules")
    rules = {rule.rule_id: rule for rule in loaded_rules.rule_set.rules}
    covered_active: set[str] = set()
    covered_negations: set[str] = set()
    for fixture in fixture_set.fixtures:
        for finding in fixture.expected_findings:
            referenced = []
            for rule_id in finding.rule_ids:
                rule = rules.get(rule_id)
                if rule is None:
                    raise ValueError("language fixture references an unknown rule")
                referenced.append(rule)
                if (
                    rule.kind is LanguageRuleKind.NEGATION
                    and LanguageReasonCode.EXPLICIT_NEGATION in finding.reason_codes
                ):
                    covered_negations.add(rule_id)
                elif fixture.kind is LanguageFixtureKind.ACTIVE_INDICATOR:
                    covered_active.add(rule_id)

            kinds = {rule.kind for rule in referenced}
            if (
                LanguageReasonCode.KEYWORD_MATCH in finding.reason_codes
                and LanguageRuleKind.KEYWORD not in kinds
            ):
                raise ValueError("keyword fixture reason requires a keyword rule")
            if (
                LanguageReasonCode.PHRASE_MATCH in finding.reason_codes
                and LanguageRuleKind.PHRASE not in kinds
            ):
                raise ValueError("phrase fixture reason requires a phrase rule")
            if (
                LanguageReasonCode.EXPLICIT_NEGATION in finding.reason_codes
                and LanguageRuleKind.NEGATION not in kinds
            ):
                raise ValueError("explicit-negation fixture requires a negation rule")
            if (
                LanguageRuleKind.NEGATION in kinds
                and LanguageReasonCode.EXPLICIT_NEGATION not in finding.reason_codes
            ):
                raise ValueError("negation rules require an explicit-negation fixture reason")
            primary = [rule for rule in referenced if rule.kind is not LanguageRuleKind.NEGATION]
            expected_primary_kinds = set()
            if LanguageReasonCode.KEYWORD_MATCH in finding.reason_codes:
                expected_primary_kinds.add(LanguageRuleKind.KEYWORD)
            if LanguageReasonCode.PHRASE_MATCH in finding.reason_codes:
                expected_primary_kinds.add(LanguageRuleKind.PHRASE)
            if {rule.kind for rule in primary} != expected_primary_kinds:
                raise ValueError("fixture match reasons differ from referenced active rule kinds")
            if finding.category in _ACTIVE_CATEGORIES and any(
                rule.category is not finding.category for rule in primary
            ):
                raise ValueError("active fixture rule category differs from its label")

    expected_active = {
        rule.rule_id
        for rule in loaded_rules.rule_set.rules
        if rule.kind is not LanguageRuleKind.NEGATION
    }
    expected_negations = {
        rule.rule_id
        for rule in loaded_rules.rule_set.rules
        if rule.kind is LanguageRuleKind.NEGATION
    }
    if covered_active != expected_active:
        raise ValueError("active fixture coverage does not include every active rule")
    if covered_negations != expected_negations:
        raise ValueError("negation fixture coverage does not include every negation rule")


def load_language_fixture_set_bytes(
    raw: bytes,
    *,
    expected_sha256: str | None = None,
    rule_set: LoadedLanguageRuleSet | None = None,
) -> LoadedLanguageFixtureSet:
    """Validate bounded JSON fixture bytes and their pinned rule references."""

    if not isinstance(raw, bytes):
        raise TypeError("language fixture artifact must be bytes")
    if not raw:
        raise ValueError("language fixture artifact must not be empty")
    if len(raw) > MAX_LANGUAGE_FIXTURE_SET_BYTES:
        raise ValueError("language fixture artifact exceeds the byte limit")
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and artifact_sha256 != expected_sha256:
        raise ValueError("language fixture artifact checksum mismatch")
    try:
        fixture_set = LanguageFixtureSet.model_validate_json(raw)
        _validate_against_rules(
            fixture_set, rule_set or load_builtin_language_rule_set()
        )
    except Exception as error:
        raise ValueError("language fixture artifact failed contract validation") from error
    return LoadedLanguageFixtureSet(
        fixture_set=fixture_set,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=len(raw),
    )


def load_builtin_language_fixture_set() -> LoadedLanguageFixtureSet:
    """Load the exact bundled English fixture artifact without network access."""

    raw = (
        files("audio_sentinel")
        .joinpath(f"resources/{BUILTIN_LANGUAGE_FIXTURE_RESOURCE}")
        .read_bytes()
    )
    loaded = load_language_fixture_set_bytes(
        raw,
        expected_sha256=BUILTIN_LANGUAGE_FIXTURE_SET_SHA256,
    )
    if (
        loaded.fixture_set.fixture_set_id != BUILTIN_LANGUAGE_FIXTURE_SET_ID
        or loaded.fixture_set.fixture_set_version
        != BUILTIN_LANGUAGE_FIXTURE_SET_VERSION
    ):
        raise ValueError("bundled language fixture identity differs from the pinned specification")
    return loaded


def language_fixture_schema_documents() -> dict[str, dict[str, object]]:
    return {
        "language-fixture-set.schema.json": LanguageFixtureSet.model_json_schema()
    }


def write_language_fixture_schemas(output_directory: Path) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in language_fixture_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
