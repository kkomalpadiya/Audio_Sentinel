"""B5.1 tests for versioned keyword, phrase, and negation rule data."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from audio_sentinel import language_rules as rules
from audio_sentinel.language_contracts import (
    LanguageCategory,
    LanguageReasonCode,
    LanguageRuleKind,
)


@pytest.fixture
def rule_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (
            project_root
            / "src"
            / "audio_sentinel"
            / "resources"
            / rules.BUILTIN_LANGUAGE_RULE_RESOURCE
        ).read_text(encoding="utf-8")
    )


def test_builtin_rule_set_is_pinned_complete_and_portable(project_root: Path) -> None:
    loaded = rules.load_builtin_language_rule_set()
    artifact = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / rules.BUILTIN_LANGUAGE_RULE_RESOURCE
    ).read_bytes()

    assert loaded.rule_set.rule_set_id == rules.BUILTIN_LANGUAGE_RULE_SET_ID
    assert loaded.rule_set.rule_set_version == rules.BUILTIN_LANGUAGE_RULE_SET_VERSION
    assert loaded.rule_set.language == "en"
    assert loaded.rule_set.negation_window_tokens == 3
    assert loaded.artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert loaded.artifact_sha256 == rules.BUILTIN_LANGUAGE_RULE_SET_SHA256
    assert loaded.artifact_size_bytes == len(artifact)
    assert len(loaded.rule_set.rules) == 59


def test_git_preserves_the_pinned_rule_artifact_bytes(project_root: Path) -> None:
    attributes = (project_root / ".gitattributes").read_text(encoding="utf-8")

    assert (
        "src/audio_sentinel/resources/language-rules-en-v1.json -text" in attributes
    )


def test_exact_rule_kind_and_category_inventory() -> None:
    inventory = rules.load_builtin_language_rule_set().rule_set.rules

    assert sum(rule.kind is LanguageRuleKind.KEYWORD for rule in inventory) == 17
    assert sum(rule.kind is LanguageRuleKind.PHRASE for rule in inventory) == 23
    assert sum(rule.kind is LanguageRuleKind.NEGATION for rule in inventory) == 19
    assert sum(rule.category is LanguageCategory.DISTRESS for rule in inventory) == 14
    assert sum(rule.category is LanguageCategory.THREAT for rule in inventory) == 11
    assert sum(
        rule.category is LanguageCategory.WEAPON_REFERENCE for rule in inventory
    ) == 15
    assert sum(rule.category is None for rule in inventory) == 19


@pytest.mark.parametrize(
    "rule_id,tokens,kind,category",
    [
        ("distress-keyword-help", ("help",), "keyword", "distress"),
        ("distress-phrase-i-cannot-breathe", ("i", "cannot", "breathe"), "phrase", "distress"),
        ("threat-keyword-kill", ("kill",), "keyword", "threat"),
        ("threat-phrase-i-will-kill-you", ("i", "will", "kill", "you"), "phrase", "threat"),
        ("weapon-keyword-gun", ("gun",), "keyword", "weapon_reference"),
        ("weapon-phrase-has-a-gun", ("has", "a", "gun"), "phrase", "weapon_reference"),
        ("negation-do-not", ("do", "not"), "negation", None),
        ("negation-dont", ("don't",), "negation", None),
        ("negation-unarmed", ("unarmed",), "negation", None),
    ],
)
def test_representative_rules_are_exact(
    rule_id: str,
    tokens: tuple[str, ...],
    kind: str,
    category: str | None,
) -> None:
    inventory = {
        rule.rule_id: rule
        for rule in rules.load_builtin_language_rule_set().rule_set.rules
    }
    rule = inventory[rule_id]

    assert rule.tokens == tokens
    assert rule.kind.value == kind
    assert (None if rule.category is None else rule.category.value) == category


def test_reason_codes_follow_rule_kinds() -> None:
    inventory = rules.load_builtin_language_rule_set().rule_set.rules
    expected = {
        LanguageRuleKind.KEYWORD: LanguageReasonCode.KEYWORD_MATCH,
        LanguageRuleKind.PHRASE: LanguageReasonCode.PHRASE_MATCH,
        LanguageRuleKind.NEGATION: LanguageReasonCode.EXPLICIT_NEGATION,
    }

    assert all(rule.reason_code is expected[rule.kind] for rule in inventory)


def test_all_active_categories_have_keyword_and_phrase_coverage() -> None:
    inventory = rules.load_builtin_language_rule_set().rule_set.rules

    for category in (
        LanguageCategory.DISTRESS,
        LanguageCategory.THREAT,
        LanguageCategory.WEAPON_REFERENCE,
    ):
        assert {
            rule.kind for rule in inventory if rule.category is category
        } == {LanguageRuleKind.KEYWORD, LanguageRuleKind.PHRASE}


def test_descriptor_matches_a5_1_evidence_contract() -> None:
    loaded = rules.load_builtin_language_rule_set()

    assert loaded.as_descriptor().model_dump(mode="json") == {
        "rule_set_id": "audio-sentinel-en-v1",
        "rule_set_version": "1.0.0",
        "rule_format_version": "1.0",
        "artifact_sha256": rules.BUILTIN_LANGUAGE_RULE_SET_SHA256,
    }


def test_artifact_contains_no_risk_or_alert_decisions(project_root: Path) -> None:
    text = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / rules.BUILTIN_LANGUAGE_RULE_RESOURCE
    ).read_text(encoding="utf-8")

    for forbidden in ("risk_score", "severity", "incident", "alert", "probability"):
        assert forbidden not in text


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"kind": "keyword", "tokens": ["call", "police"], "category": "distress", "reason_code": "keyword_match"}, "exactly one"),
        ({"kind": "keyword", "tokens": ["help"], "category": "ambiguous", "reason_code": "keyword_match"}, "active language category"),
        ({"kind": "keyword", "tokens": ["help"], "category": "distress", "reason_code": "phrase_match"}, "keyword_match"),
        ({"kind": "phrase", "tokens": ["help"], "category": "distress", "reason_code": "phrase_match"}, "at least two"),
        ({"kind": "phrase", "tokens": ["call", "police"], "category": None, "reason_code": "phrase_match"}, "active language category"),
        ({"kind": "phrase", "tokens": ["call", "police"], "category": "distress", "reason_code": "keyword_match"}, "phrase_match"),
        ({"kind": "negation", "tokens": ["not"], "category": "threat", "reason_code": "explicit_negation"}, "cannot assign"),
        ({"kind": "negation", "tokens": ["not"], "category": None, "reason_code": "keyword_match"}, "explicit_negation"),
    ],
)
def test_rule_kind_shapes_are_strict(
    overrides: dict[str, object], message: str
) -> None:
    document = {"rule_id": "test-rule-001", **overrides}

    with pytest.raises(ValidationError, match=message):
        rules.LanguageRule.model_validate(document)


@pytest.mark.parametrize(
    "tokens",
    [
        ["Help"],
        ["help!"],
        ["two words"],
        ["don’t"],
        ["123"],
        [""],
    ],
)
def test_rule_tokens_must_already_match_normalization_contract(
    tokens: list[str],
) -> None:
    with pytest.raises(ValidationError, match="normalized lowercase"):
        rules.LanguageRule(
            rule_id="test-rule-001",
            kind="keyword",
            tokens=tokens,
            category="distress",
            reason_code="keyword_match",
        )


def test_duplicate_rule_ids_are_rejected(rule_data: dict[str, object]) -> None:
    rule_data["rules"][1]["rule_id"] = rule_data["rules"][0]["rule_id"]

    with pytest.raises(ValidationError, match="rule IDs must be unique"):
        rules.LanguageRuleSet.model_validate(rule_data)


def test_duplicate_patterns_are_rejected_even_with_different_ids(
    rule_data: dict[str, object],
) -> None:
    duplicate = copy.deepcopy(rule_data["rules"][0])
    duplicate["rule_id"] = "distress-keyword-duplicate"
    rule_data["rules"].insert(1, duplicate)

    with pytest.raises(ValidationError, match="patterns must be unique"):
        rules.LanguageRuleSet.model_validate(rule_data)


def test_rule_inventory_requires_canonical_order(rule_data: dict[str, object]) -> None:
    rule_data["rules"][0], rule_data["rules"][1] = (
        rule_data["rules"][1],
        rule_data["rules"][0],
    )

    with pytest.raises(ValidationError, match="canonical deterministic ordering"):
        rules.LanguageRuleSet.model_validate(rule_data)


@pytest.mark.parametrize("kind", ["keyword", "phrase", "negation"])
def test_rule_set_requires_every_rule_kind(
    rule_data: dict[str, object], kind: str
) -> None:
    rule_data["rules"] = [rule for rule in rule_data["rules"] if rule["kind"] != kind]

    with pytest.raises(ValidationError, match="keyword, phrase, and negation"):
        rules.LanguageRuleSet.model_validate(rule_data)


@pytest.mark.parametrize("category", ["distress", "threat", "weapon_reference"])
@pytest.mark.parametrize("kind", ["keyword", "phrase"])
def test_each_active_category_requires_both_match_kinds(
    rule_data: dict[str, object], category: str, kind: str
) -> None:
    rule_data["rules"] = [
        rule
        for rule in rule_data["rules"]
        if not (rule["category"] == category and rule["kind"] == kind)
    ]

    with pytest.raises(ValidationError, match=f"{category} requires both"):
        rules.LanguageRuleSet.model_validate(rule_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0"),
        ("rule_format_version", "2.0"),
        ("language", "fr"),
        ("negation_window_tokens", 0),
        ("negation_window_tokens", 9),
        ("unexpected", True),
    ],
)
def test_rule_set_metadata_is_closed_and_bounded(
    rule_data: dict[str, object], field: str, value: object
) -> None:
    rule_data[field] = value

    with pytest.raises(ValidationError):
        rules.LanguageRuleSet.model_validate(rule_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("unicode_form", "NFC"),
        ("casefold", False),
        ("apostrophe_policy", "drop"),
        ("token_pattern", ".*"),
        ("unexpected", True),
    ],
)
def test_normalization_recipe_is_exact(
    rule_data: dict[str, object], field: str, value: object
) -> None:
    rule_data["normalization"][field] = value

    with pytest.raises(ValidationError):
        rules.LanguageRuleSet.model_validate(rule_data)


def test_loader_rejects_changed_bytes(project_root: Path) -> None:
    artifact = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / rules.BUILTIN_LANGUAGE_RULE_RESOURCE
    ).read_bytes()

    with pytest.raises(ValueError, match="checksum mismatch"):
        rules.load_language_rule_set_bytes(
            artifact + b"\n",
            expected_sha256=rules.BUILTIN_LANGUAGE_RULE_SET_SHA256,
        )


@pytest.mark.parametrize(
    "payload,message",
    [
        (b"", "must not be empty"),
        (b"not-json", "contract validation"),
        (b"{}", "contract validation"),
    ],
)
def test_loader_rejects_empty_or_invalid_payload(payload: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rules.load_language_rule_set_bytes(payload)


def test_loader_rejects_wrong_type_and_oversized_payload() -> None:
    with pytest.raises(TypeError, match="must be bytes"):
        rules.load_language_rule_set_bytes("{}")
    with pytest.raises(ValueError, match="byte limit"):
        rules.load_language_rule_set_bytes(b"x" * (rules.MAX_LANGUAGE_RULE_SET_BYTES + 1))


def test_builtin_loader_rejects_changed_resource(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resource = tmp_path / "resources"
    resource.mkdir()
    (resource / rules.BUILTIN_LANGUAGE_RULE_RESOURCE).write_bytes(b"{}")
    monkeypatch.setattr(rules, "files", lambda package: tmp_path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        rules.load_builtin_language_rule_set()


def test_builtin_loader_rejects_identity_drift(
    monkeypatch: pytest.MonkeyPatch, rule_data: dict[str, object], tmp_path: Path
) -> None:
    rule_data["rule_set_version"] = "1.0.1"
    raw = (json.dumps(rule_data, indent=2) + "\n").encode("utf-8")
    resource = tmp_path / "resources"
    resource.mkdir()
    (resource / rules.BUILTIN_LANGUAGE_RULE_RESOURCE).write_bytes(raw)
    monkeypatch.setattr(rules, "files", lambda package: tmp_path)
    monkeypatch.setattr(rules, "BUILTIN_LANGUAGE_RULE_SET_SHA256", hashlib.sha256(raw).hexdigest())

    with pytest.raises(ValueError, match="identity differs"):
        rules.load_builtin_language_rule_set()


def test_models_and_loaded_result_are_immutable() -> None:
    loaded = rules.load_builtin_language_rule_set()

    with pytest.raises(ValidationError):
        loaded.rule_set.negation_window_tokens = 4
    with pytest.raises(AttributeError):
        loaded.artifact_sha256 = "0" * 64


def test_contract_loads_without_network_or_ml_runtime() -> None:
    code = """
import sys
class NoModelImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'onnxruntime', 'faster_whisper'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
def reject_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError('Unexpected network access')
sys.addaudithook(reject_network)
from audio_sentinel.language_rules import load_builtin_language_rule_set
assert len(load_builtin_language_rule_set().rule_set.rules) == 59
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        capture_output=True,
        timeout=30,
    )


def test_checked_in_schema_matches_generated_contract(project_root: Path) -> None:
    expected = rules.language_rule_schema_documents()["language-rule-set.schema.json"]
    checked_in = json.loads(
        (project_root / "docs" / "schemas" / "v1" / "language-rule-set.schema.json")
        .read_text(encoding="utf-8")
    )

    assert checked_in == expected
    assert checked_in["$id"].endswith("/language-rule-set.schema.json")
    assert "risk_score" not in json.dumps(checked_in)


def test_schema_export_writes_portable_json(tmp_path: Path) -> None:
    exported = rules.write_language_rule_schemas(tmp_path)
    destination = exported["language-rule-set.schema.json"]

    assert destination == tmp_path / "language-rule-set.schema.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        rules.language_rule_schema_documents()["language-rule-set.schema.json"]
    )
