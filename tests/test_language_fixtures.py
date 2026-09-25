"""B5.2 tests for versioned labeled language fixtures."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from audio_sentinel import language_fixtures as fixtures
from audio_sentinel.language_contracts import LanguageCategory, LanguageReasonCode
from audio_sentinel.language_rules import load_builtin_language_rule_set


@pytest.fixture
def fixture_data(project_root: Path) -> dict[str, object]:
    return json.loads(
        (
            project_root
            / "src"
            / "audio_sentinel"
            / "resources"
            / fixtures.BUILTIN_LANGUAGE_FIXTURE_RESOURCE
        ).read_text(encoding="utf-8")
    )


def _recount(document: dict[str, object]) -> None:
    document["fixture_count"] = len(document["fixtures"])
    document["expected_finding_count"] = sum(
        len(item["expected_findings"]) for item in document["fixtures"]
    )


def _load_document(document: dict[str, object]) -> fixtures.LoadedLanguageFixtureSet:
    return fixtures.load_language_fixture_set_bytes(
        (json.dumps(document, ensure_ascii=False) + "\n").encode("utf-8")
    )


def test_builtin_fixture_set_is_pinned_complete_and_portable(project_root: Path) -> None:
    loaded = fixtures.load_builtin_language_fixture_set()
    artifact = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / fixtures.BUILTIN_LANGUAGE_FIXTURE_RESOURCE
    ).read_bytes()

    assert loaded.fixture_set.fixture_set_id == fixtures.BUILTIN_LANGUAGE_FIXTURE_SET_ID
    assert loaded.fixture_set.fixture_set_version == fixtures.BUILTIN_LANGUAGE_FIXTURE_SET_VERSION
    assert loaded.fixture_set.language == "en"
    assert loaded.fixture_set.label_contract_version == "1.0"
    assert loaded.fixture_set.fixture_count == 74
    assert loaded.fixture_set.expected_finding_count == 75
    assert loaded.artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert loaded.artifact_sha256 == fixtures.BUILTIN_LANGUAGE_FIXTURE_SET_SHA256
    assert loaded.artifact_size_bytes == len(artifact)


def test_git_preserves_pinned_fixture_bytes(project_root: Path) -> None:
    attributes = (project_root / ".gitattributes").read_text(encoding="utf-8")

    assert (
        "src/audio_sentinel/resources/language-fixtures-en-v1.json -text"
        in attributes
    )


def test_exact_fixture_kind_inventory() -> None:
    inventory = fixtures.load_builtin_language_fixture_set().fixture_set.fixtures

    assert sum(item.kind is fixtures.LanguageFixtureKind.ACTIVE_INDICATOR for item in inventory) == 43
    assert sum(item.kind is fixtures.LanguageFixtureKind.NO_CONCERNING_MATCH for item in inventory) == 6
    assert sum(item.kind is fixtures.LanguageFixtureKind.CONTEXT_SUPPRESSED for item in inventory) == 23
    assert sum(item.kind is fixtures.LanguageFixtureKind.AMBIGUOUS for item in inventory) == 2


def test_every_versioned_active_and_negation_rule_has_fixture_coverage() -> None:
    fixture_set = fixtures.load_builtin_language_fixture_set().fixture_set
    rules = load_builtin_language_rule_set().rule_set.rules
    active_expected = {rule.rule_id for rule in rules if rule.category is not None}
    negation_expected = {rule.rule_id for rule in rules if rule.category is None}
    active_actual = {
        rule_id
        for fixture in fixture_set.fixtures
        if fixture.kind is fixtures.LanguageFixtureKind.ACTIVE_INDICATOR
        for finding in fixture.expected_findings
        for rule_id in finding.rule_ids
    }
    negation_actual = {
        rule_id
        for fixture in fixture_set.fixtures
        if LanguageReasonCode.EXPLICIT_NEGATION
        in fixture.expected_findings[0].reason_codes
        for finding in fixture.expected_findings
        for rule_id in finding.rule_ids
        if rule_id.startswith("negation-")
    }

    assert active_actual == active_expected
    assert negation_actual == negation_expected


def test_fixture_rule_descriptor_matches_pinned_rule_artifact() -> None:
    fixture_set = fixtures.load_builtin_language_fixture_set().fixture_set

    assert fixture_set.rule_set == load_builtin_language_rule_set().as_descriptor()


@pytest.mark.parametrize(
    ("fixture_id", "kind", "category", "reasons", "rule_ids"),
    [
        ("active-distress-keyword-help", "active_indicator", "distress", ("keyword_match",), ("distress-keyword-help",)),
        ("active-distress-phrase-i-cannot-breathe", "active_indicator", "distress", ("phrase_match",), ("distress-phrase-i-cannot-breathe",)),
        ("active-weapon-keyword-gun", "active_indicator", "weapon_reference", ("keyword_match",), ("weapon-keyword-gun",)),
        ("negated-do-not-shoot-you", "context_suppressed", "context_suppressed", ("phrase_match", "explicit_negation"), ("negation-do-not", "threat-phrase-shoot-you")),
        ("negated-apostrophe-cant-have-a-gun", "context_suppressed", "context_suppressed", ("phrase_match", "explicit_negation"), ("negation-cant", "weapon-phrase-have-a-gun")),
        ("boundary-negation-period", "active_indicator", "weapon_reference", ("keyword_match",), ("weapon-keyword-gun",)),
        ("harmless-helpdesk-token-boundary", "no_concerning_match", "no_concerning_match", ("no_rule_match",), ()),
        ("hypothetical-example-kill-you", "context_suppressed", "context_suppressed", ("phrase_match", "hypothetical_or_conditional"), ("threat-phrase-i-will-kill-you",)),
        ("quoted-script-i-will-hurt-you", "context_suppressed", "context_suppressed", ("phrase_match", "quoted_or_reported_speech"), ("threat-phrase-i-will-hurt-you",)),
        ("ambiguous-shoot-alone", "ambiguous", "ambiguous", ("keyword_match", "insufficient_context"), ("threat-keyword-shoot",)),
    ],
)
def test_representative_fixture_labels_are_exact(
    fixture_id: str,
    kind: str,
    category: str,
    reasons: tuple[str, ...],
    rule_ids: tuple[str, ...],
) -> None:
    inventory = {
        item.fixture_id: item
        for item in fixtures.load_builtin_language_fixture_set().fixture_set.fixtures
    }
    item = inventory[fixture_id]
    finding = item.expected_findings[0]

    assert item.kind.value == kind
    assert finding.category.value == category
    assert tuple(reason.value for reason in finding.reason_codes) == reasons
    assert finding.rule_ids == rule_ids


def test_multi_finding_fixture_preserves_expected_order() -> None:
    inventory = {
        item.fixture_id: item
        for item in fixtures.load_builtin_language_fixture_set().fixture_set.fixtures
    }
    item = inventory["multi-distress-and-weapon"]

    assert [finding.category for finding in item.expected_findings] == [
        LanguageCategory.DISTRESS,
        LanguageCategory.WEAPON_REFERENCE,
    ]
    assert [finding.rule_ids[0] for finding in item.expected_findings] == [
        "distress-phrase-please-help-me",
        "weapon-phrase-has-a-gun",
    ]


def test_explicit_negation_inventory_has_all_harmless_negation_tags() -> None:
    inventory = fixtures.load_builtin_language_fixture_set().fixture_set.fixtures
    explicit = [
        item
        for item in inventory
        if LanguageReasonCode.EXPLICIT_NEGATION in item.expected_findings[0].reason_codes
    ]

    assert len(explicit) == 19
    assert all("harmless-negation" in item.tags for item in explicit)
    assert all(item.kind is fixtures.LanguageFixtureKind.CONTEXT_SUPPRESSED for item in explicit)


def test_fixture_artifact_contains_no_risk_or_alert_labels(project_root: Path) -> None:
    text = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / fixtures.BUILTIN_LANGUAGE_FIXTURE_RESOURCE
    ).read_text(encoding="utf-8")

    for forbidden in ("risk_score", "severity", "incident", "alert", "probability"):
        assert forbidden not in text


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"category": "no_concerning_match", "reason_codes": ["keyword_match"], "rule_ids": ["distress-keyword-help"]}, "no_rule_match"),
        ({"category": "distress", "reason_codes": ["no_rule_match"], "rule_ids": []}, "reserved"),
        ({"category": "distress", "reason_codes": ["keyword_match"], "rule_ids": []}, "require a match"),
        ({"category": "threat", "reason_codes": ["keyword_match", "explicit_negation"], "rule_ids": ["threat-keyword-kill", "negation-not"]}, "only match"),
        ({"category": "context_suppressed", "reason_codes": ["keyword_match"], "rule_ids": ["threat-keyword-kill"]}, "suppression reason"),
        ({"category": "ambiguous", "reason_codes": ["keyword_match"], "rule_ids": ["threat-keyword-kill"]}, "ambiguity reason"),
        ({"category": "ambiguous", "reason_codes": ["keyword_match", "explicit_negation", "insufficient_context"], "rule_ids": ["threat-keyword-kill", "negation-not"]}, "incompatible"),
        ({"category": "distress", "reason_codes": ["keyword_match", "keyword_match"], "rule_ids": ["distress-keyword-help"]}, "unique"),
        ({"category": "distress", "reason_codes": ["explicit_negation", "keyword_match"], "rule_ids": ["distress-keyword-help", "negation-not"]}, "canonical"),
        ({"category": "distress", "reason_codes": ["keyword_match"], "rule_ids": ["distress-keyword-help", "distress-keyword-help"]}, "unique"),
    ],
)
def test_expected_finding_shapes_are_strict(
    document: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        fixtures.ExpectedLanguageFinding.model_validate(document)


@pytest.mark.parametrize(
    ("kind", "finding"),
    [
        ("no_concerning_match", {"category": "distress", "reason_codes": ["keyword_match"], "rule_ids": ["distress-keyword-help"]}),
        ("active_indicator", {"category": "context_suppressed", "reason_codes": ["keyword_match", "explicit_negation"], "rule_ids": ["negation-not", "distress-keyword-help"]}),
        ("context_suppressed", {"category": "threat", "reason_codes": ["keyword_match"], "rule_ids": ["threat-keyword-kill"]}),
        ("ambiguous", {"category": "distress", "reason_codes": ["keyword_match"], "rule_ids": ["distress-keyword-help"]}),
    ],
)
def test_fixture_kind_must_match_expected_categories(
    kind: str, finding: dict[str, object]
) -> None:
    document = {
        "fixture_id": "invalid-fixture",
        "kind": kind,
        "text": "example",
        "tags": ["test-tag"],
        "expected_findings": [finding],
    }

    with pytest.raises(ValidationError):
        fixtures.LanguageFixture.model_validate(document)


def test_duplicate_expected_findings_are_rejected() -> None:
    finding = {
        "category": "distress",
        "reason_codes": ["keyword_match"],
        "rule_ids": ["distress-keyword-help"],
    }

    with pytest.raises(ValidationError, match="expected findings must be unique"):
        fixtures.LanguageFixture(
            fixture_id="invalid-fixture",
            kind="active_indicator",
            text="help twice",
            tags=("test-tag",),
            expected_findings=(finding, finding),
        )


@pytest.mark.parametrize("text", ["", "   ", " leading", "trailing ", "bad\rtext", "bad\x00text"])
def test_fixture_text_is_bounded_trimmed_and_safe(text: str) -> None:
    with pytest.raises(ValidationError):
        fixtures.LanguageFixture(
            fixture_id="invalid-fixture",
            kind="no_concerning_match",
            text=text,
            tags=("test-tag",),
            expected_findings=(
                {"category": "no_concerning_match", "reason_codes": ("no_rule_match",)},
            ),
        )


@pytest.mark.parametrize("tags", [("z-tag", "a-tag"), ("same-tag", "same-tag")])
def test_fixture_tags_are_unique_and_ordered(tags: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="tags must be unique"):
        fixtures.LanguageFixture(
            fixture_id="invalid-fixture",
            kind="no_concerning_match",
            text="example",
            tags=tags,
            expected_findings=(
                {"category": "no_concerning_match", "reason_codes": ("no_rule_match",)},
            ),
        )


def test_duplicate_fixture_ids_are_rejected(fixture_data: dict[str, object]) -> None:
    fixture_data["fixtures"][1]["fixture_id"] = fixture_data["fixtures"][0]["fixture_id"]

    with pytest.raises(ValidationError, match="fixture IDs must be unique"):
        fixtures.LanguageFixtureSet.model_validate(fixture_data)


def test_duplicate_fixture_texts_are_rejected(fixture_data: dict[str, object]) -> None:
    fixture_data["fixtures"][1]["text"] = fixture_data["fixtures"][0]["text"]

    with pytest.raises(ValidationError, match="fixture texts must be unique"):
        fixtures.LanguageFixtureSet.model_validate(fixture_data)


def test_fixture_inventory_requires_canonical_order(fixture_data: dict[str, object]) -> None:
    fixture_data["fixtures"][0], fixture_data["fixtures"][1] = (
        fixture_data["fixtures"][1],
        fixture_data["fixtures"][0],
    )

    with pytest.raises(ValidationError, match="canonical fixture-ID ordering"):
        fixtures.LanguageFixtureSet.model_validate(fixture_data)


@pytest.mark.parametrize("field", ["fixture_count", "expected_finding_count"])
def test_inventory_counts_are_exact(fixture_data: dict[str, object], field: str) -> None:
    fixture_data[field] += 1

    with pytest.raises(ValidationError, match="count must equal"):
        fixtures.LanguageFixtureSet.model_validate(fixture_data)


@pytest.mark.parametrize("kind", list(fixtures.LanguageFixtureKind))
def test_fixture_set_requires_every_kind(
    fixture_data: dict[str, object], kind: fixtures.LanguageFixtureKind
) -> None:
    fixture_data["fixtures"] = [
        item for item in fixture_data["fixtures"] if item["kind"] != kind.value
    ]
    _recount(fixture_data)

    with pytest.raises(ValidationError, match="every fixture kind"):
        fixtures.LanguageFixtureSet.model_validate(fixture_data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "2.0"),
        ("fixture_set_version", ""),
        ("language", "fr"),
        ("label_contract_version", "2.0"),
        ("unexpected", True),
    ],
)
def test_fixture_metadata_is_closed_and_versioned(
    fixture_data: dict[str, object], field: str, value: object
) -> None:
    fixture_data[field] = value

    with pytest.raises(ValidationError):
        fixtures.LanguageFixtureSet.model_validate(fixture_data)


def test_loader_rejects_unknown_rule_reference(fixture_data: dict[str, object]) -> None:
    fixture_data["fixtures"][0]["expected_findings"][0]["rule_ids"] = ["unknown-rule"]

    with pytest.raises(ValueError, match="contract validation"):
        _load_document(fixture_data)


def test_loader_rejects_rule_provenance_drift(fixture_data: dict[str, object]) -> None:
    fixture_data["rule_set"]["artifact_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="contract validation"):
        _load_document(fixture_data)


def test_loader_rejects_incomplete_active_rule_coverage(
    fixture_data: dict[str, object]
) -> None:
    fixture_data["fixtures"] = [
        item
        for item in fixture_data["fixtures"]
        if item["fixture_id"] != "active-distress-keyword-bleeding"
    ]
    _recount(fixture_data)

    with pytest.raises(ValueError, match="contract validation"):
        _load_document(fixture_data)


def test_loader_rejects_incomplete_negation_rule_coverage(
    fixture_data: dict[str, object]
) -> None:
    target = next(
        item
        for item in fixture_data["fixtures"]
        if item["fixture_id"] == "negated-no-gun"
    )
    target["expected_findings"][0]["rule_ids"] = [
        "negation-not",
        "weapon-keyword-gun",
    ]

    with pytest.raises(ValueError, match="contract validation"):
        _load_document(fixture_data)


def test_loader_rejects_reason_and_rule_kind_mismatch(
    fixture_data: dict[str, object]
) -> None:
    target = fixture_data["fixtures"][0]["expected_findings"][0]
    target["rule_ids"] = ["distress-phrase-i-need-help"]

    with pytest.raises(ValueError, match="contract validation"):
        _load_document(fixture_data)


def test_loader_rejects_unexplained_extra_rule_kind(
    fixture_data: dict[str, object]
) -> None:
    target = fixture_data["fixtures"][0]["expected_findings"][0]
    target["rule_ids"] = [
        "distress-keyword-bleeding",
        "distress-phrase-i-am-bleeding",
    ]

    with pytest.raises(ValueError, match="contract validation"):
        _load_document(fixture_data)


def test_loader_rejects_active_category_mismatch(fixture_data: dict[str, object]) -> None:
    target = fixture_data["fixtures"][0]["expected_findings"][0]
    target["rule_ids"] = ["threat-keyword-kill"]

    with pytest.raises(ValueError, match="contract validation"):
        _load_document(fixture_data)


def test_loader_rejects_changed_bytes(project_root: Path) -> None:
    artifact = (
        project_root
        / "src"
        / "audio_sentinel"
        / "resources"
        / fixtures.BUILTIN_LANGUAGE_FIXTURE_RESOURCE
    ).read_bytes()

    with pytest.raises(ValueError, match="checksum mismatch"):
        fixtures.load_language_fixture_set_bytes(
            artifact + b"\n",
            expected_sha256=fixtures.BUILTIN_LANGUAGE_FIXTURE_SET_SHA256,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [(b"", "must not be empty"), (b"not-json", "contract validation"), (b"{}", "contract validation")],
)
def test_loader_rejects_empty_or_invalid_payload(payload: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fixtures.load_language_fixture_set_bytes(payload)


def test_loader_rejects_wrong_type_and_oversized_payload() -> None:
    with pytest.raises(TypeError, match="must be bytes"):
        fixtures.load_language_fixture_set_bytes("{}")
    with pytest.raises(ValueError, match="byte limit"):
        fixtures.load_language_fixture_set_bytes(
            b"x" * (fixtures.MAX_LANGUAGE_FIXTURE_SET_BYTES + 1)
        )


def test_builtin_loader_rejects_changed_resource(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resource = tmp_path / "resources"
    resource.mkdir()
    (resource / fixtures.BUILTIN_LANGUAGE_FIXTURE_RESOURCE).write_bytes(b"{}")
    monkeypatch.setattr(fixtures, "files", lambda package: tmp_path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        fixtures.load_builtin_language_fixture_set()


def test_builtin_loader_rejects_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    fixture_data: dict[str, object],
    tmp_path: Path,
) -> None:
    fixture_data["fixture_set_version"] = "1.0.1"
    raw = (json.dumps(fixture_data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    resource = tmp_path / "resources"
    resource.mkdir()
    (resource / fixtures.BUILTIN_LANGUAGE_FIXTURE_RESOURCE).write_bytes(raw)
    monkeypatch.setattr(fixtures, "files", lambda package: tmp_path)
    monkeypatch.setattr(
        fixtures, "BUILTIN_LANGUAGE_FIXTURE_SET_SHA256", hashlib.sha256(raw).hexdigest()
    )

    with pytest.raises(ValueError, match="identity differs"):
        fixtures.load_builtin_language_fixture_set()


def test_models_and_loaded_result_are_immutable() -> None:
    loaded = fixtures.load_builtin_language_fixture_set()

    with pytest.raises(ValidationError):
        loaded.fixture_set.fixture_count = 1
    with pytest.raises(AttributeError):
        loaded.artifact_sha256 = "0" * 64


def test_fixture_loader_is_local_only_and_independent_of_analysis_engine() -> None:
    code = """
import sys
class RestrictedImports:
    def find_spec(self, fullname, *args):
        if fullname == 'audio_sentinel.language_analysis':
            raise AssertionError('Fixture loading imported the analysis engine')
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'onnxruntime', 'faster_whisper'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RestrictedImports())
def reject_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError('Unexpected network access')
sys.addaudithook(reject_network)
from audio_sentinel.language_fixtures import load_builtin_language_fixture_set
assert load_builtin_language_fixture_set().fixture_set.fixture_count == 74
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
    expected = fixtures.language_fixture_schema_documents()[
        "language-fixture-set.schema.json"
    ]
    checked_in = json.loads(
        (
            project_root
            / "docs"
            / "schemas"
            / "v1"
            / "language-fixture-set.schema.json"
        ).read_text(encoding="utf-8")
    )

    assert checked_in == expected
    assert checked_in["$id"].endswith("/language-fixture-set.schema.json")
    assert "risk_score" not in json.dumps(checked_in)


def test_schema_export_writes_portable_json(tmp_path: Path) -> None:
    exported = fixtures.write_language_fixture_schemas(tmp_path)
    destination = exported["language-fixture-set.schema.json"]

    assert destination == tmp_path / "language-fixture-set.schema.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        fixtures.language_fixture_schema_documents()[
            "language-fixture-set.schema.json"
        ]
    )
