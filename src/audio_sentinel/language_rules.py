"""B5.1 versioned language-rule data and integrity-checked loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import re
from pathlib import Path
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_sentinel.language_contracts import (
    LANGUAGE_RULE_FORMAT_VERSION,
    LanguageCategory,
    LanguageReasonCode,
    LanguageRuleKind,
    LanguageRuleSetDescriptor,
)
from audio_sentinel.preparation import Identifier


LANGUAGE_RULE_SET_SCHEMA_VERSION = "1.0"
BUILTIN_LANGUAGE_RULE_SET_ID = "audio-sentinel-en-v1"
BUILTIN_LANGUAGE_RULE_SET_VERSION = "1.0.0"
BUILTIN_LANGUAGE_RULE_RESOURCE = "language-rules-en-v1.json"
BUILTIN_LANGUAGE_RULE_SET_SHA256 = (
    "24794b01df242c37364f8c8c0740fa14f86c1d64d6d7f1076d03bec1c58d8488"
)
MAX_LANGUAGE_RULE_SET_BYTES = 1_048_576
TOKEN_PATTERN = r"^[a-z]+(?:'[a-z]+)?$"


_TOKEN_RE = re.compile(TOKEN_PATTERN)
_ACTIVE_CATEGORIES = frozenset(
    {
        LanguageCategory.DISTRESS,
        LanguageCategory.THREAT,
        LanguageCategory.WEAPON_REFERENCE,
    }
)
_RULE_KIND_ORDER = {kind: index for index, kind in enumerate(LanguageRuleKind)}
_CATEGORY_ORDER = {category: index for index, category in enumerate(LanguageCategory)}


class LanguageRuleRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class LanguageNormalization(LanguageRuleRecord):
    """Exact text normalization contract that A5.2 must implement."""

    unicode_form: Literal["NFKC"] = "NFKC"
    casefold: Literal[True] = True
    apostrophe_policy: Literal["curly_to_ascii_preserve"] = "curly_to_ascii_preserve"
    token_pattern: Literal[r"^[a-z]+(?:'[a-z]+)?$"] = TOKEN_PATTERN


class LanguageRule(LanguageRuleRecord):
    """One normalized keyword, phrase, or explicit-negation rule."""

    rule_id: Identifier
    kind: LanguageRuleKind
    tokens: tuple[str, ...] = Field(min_length=1, max_length=8)
    category: LanguageCategory | None = None
    reason_code: LanguageReasonCode

    @field_validator("tokens")
    @classmethod
    def validate_tokens(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for token in value:
            if (
                not _TOKEN_RE.fullmatch(token)
                or token != token.casefold()
                or token != unicodedata.normalize("NFKC", token)
            ):
                raise ValueError("language-rule tokens must be normalized lowercase words")
        return value

    @model_validator(mode="after")
    def validate_rule_shape(self) -> "LanguageRule":
        if self.kind is LanguageRuleKind.KEYWORD:
            if len(self.tokens) != 1:
                raise ValueError("keyword rules require exactly one token")
            if self.category not in _ACTIVE_CATEGORIES:
                raise ValueError("keyword rules require an active language category")
            if self.reason_code is not LanguageReasonCode.KEYWORD_MATCH:
                raise ValueError("keyword rules require keyword_match")
        elif self.kind is LanguageRuleKind.PHRASE:
            if len(self.tokens) < 2:
                raise ValueError("phrase rules require at least two tokens")
            if self.category not in _ACTIVE_CATEGORIES:
                raise ValueError("phrase rules require an active language category")
            if self.reason_code is not LanguageReasonCode.PHRASE_MATCH:
                raise ValueError("phrase rules require phrase_match")
        elif self.kind is LanguageRuleKind.NEGATION:
            if self.category is not None:
                raise ValueError("negation rules cannot assign a language category")
            if self.reason_code is not LanguageReasonCode.EXPLICIT_NEGATION:
                raise ValueError("negation rules require explicit_negation")
        return self


class LanguageRuleSet(LanguageRuleRecord):
    """Complete deterministic rule inventory for the first English baseline."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        json_schema_extra={
            "$id": "https://audio-sentinel.local/schemas/v1/language-rule-set.schema.json"
        },
    )

    schema_version: Literal["1.0"] = LANGUAGE_RULE_SET_SCHEMA_VERSION
    rule_format_version: Literal["1.0"] = LANGUAGE_RULE_FORMAT_VERSION
    rule_set_id: Identifier
    rule_set_version: str = Field(min_length=1, max_length=128)
    language: Literal["en"] = "en"
    normalization: LanguageNormalization
    negation_window_tokens: int = Field(default=3, ge=1, le=8, strict=True)
    rules: tuple[LanguageRule, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_inventory(self) -> "LanguageRuleSet":
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("language rule IDs must be unique")

        signatures = [(rule.kind, rule.tokens) for rule in self.rules]
        if len(set(signatures)) != len(signatures):
            raise ValueError("language rule patterns must be unique within each rule kind")

        if list(self.rules) != sorted(self.rules, key=_rule_sort_key):
            raise ValueError("language rules must use canonical deterministic ordering")

        if {rule.kind for rule in self.rules} != set(LanguageRuleKind):
            raise ValueError("rule set must contain keyword, phrase, and negation rules")
        for category in _ACTIVE_CATEGORIES:
            category_kinds = {
                rule.kind for rule in self.rules if rule.category is category
            }
            if category_kinds != {LanguageRuleKind.KEYWORD, LanguageRuleKind.PHRASE}:
                raise ValueError(
                    f"{category.value} requires both keyword and phrase rule coverage"
                )
        return self


def _rule_sort_key(rule: LanguageRule) -> tuple[int, int, tuple[str, ...], str]:
    category_index = (
        len(_CATEGORY_ORDER) if rule.category is None else _CATEGORY_ORDER[rule.category]
    )
    return _RULE_KIND_ORDER[rule.kind], category_index, rule.tokens, rule.rule_id


@dataclass(frozen=True)
class LoadedLanguageRuleSet:
    """Validated rule data plus the digest needed by language evidence."""

    rule_set: LanguageRuleSet
    artifact_sha256: str
    artifact_size_bytes: int

    def as_descriptor(self) -> LanguageRuleSetDescriptor:
        return LanguageRuleSetDescriptor(
            rule_set_id=self.rule_set.rule_set_id,
            rule_set_version=self.rule_set.rule_set_version,
            rule_format_version=self.rule_set.rule_format_version,
            artifact_sha256=self.artifact_sha256,
        )


def load_language_rule_set_bytes(
    raw: bytes,
    *,
    expected_sha256: str | None = None,
) -> LoadedLanguageRuleSet:
    """Validate bounded JSON bytes and optionally require an exact artifact digest."""

    if not isinstance(raw, bytes):
        raise TypeError("language rule artifact must be bytes")
    if not raw:
        raise ValueError("language rule artifact must not be empty")
    if len(raw) > MAX_LANGUAGE_RULE_SET_BYTES:
        raise ValueError("language rule artifact exceeds the byte limit")
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and artifact_sha256 != expected_sha256:
        raise ValueError("language rule artifact checksum mismatch")
    try:
        rule_set = LanguageRuleSet.model_validate_json(raw)
    except Exception as error:
        raise ValueError("language rule artifact failed contract validation") from error
    return LoadedLanguageRuleSet(
        rule_set=rule_set,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=len(raw),
    )


def load_builtin_language_rule_set() -> LoadedLanguageRuleSet:
    """Load the exact bundled English v1 rule artifact without network access."""

    raw = (
        files("audio_sentinel")
        .joinpath(f"resources/{BUILTIN_LANGUAGE_RULE_RESOURCE}")
        .read_bytes()
    )
    loaded = load_language_rule_set_bytes(
        raw,
        expected_sha256=BUILTIN_LANGUAGE_RULE_SET_SHA256,
    )
    if (
        loaded.rule_set.rule_set_id != BUILTIN_LANGUAGE_RULE_SET_ID
        or loaded.rule_set.rule_set_version != BUILTIN_LANGUAGE_RULE_SET_VERSION
    ):
        raise ValueError("bundled language rule identity differs from the pinned specification")
    return loaded


def language_rule_schema_documents() -> dict[str, dict[str, object]]:
    """Return the public JSON Schema for the v1 language-rule artifact."""

    return {"language-rule-set.schema.json": LanguageRuleSet.model_json_schema()}


def write_language_rule_schemas(output_directory: Path) -> dict[str, Path]:
    """Export the portable language-rule schema for non-Python consumers."""

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for filename, document in language_rule_schema_documents().items():
        destination = output_directory / filename
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported[filename] = destination
    return exported
