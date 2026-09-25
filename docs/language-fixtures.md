# Labeled language fixtures

## What B5.2 adds

B5.2 adds a versioned English evaluation artifact at
`src/audio_sentinel/resources/language-fixtures-en-v1.json`. It contains 74 short,
synthetic accepted-transcript examples with 75 expected findings. The corresponding
loader and validation contract are in `src/audio_sentinel/language_fixtures.py`, and
the portable JSON Schema is checked in at
`docs/schemas/v1/language-fixture-set.schema.json`.

The fixtures are evaluation data for A5.3. Loading them alone does not execute the
analysis engine, change its rules, assign risk, or send alerts. The A5.3 regression
test explicitly runs every fixture through the engine and compares complete output
signatures.

## Label structure

Every fixture has:

- a stable fixture ID and trimmed English transcript text;
- one of four purposes: active indicator, no concerning match, context suppressed,
  or ambiguous;
- alphabetically ordered tags for selecting focused test groups; and
- one or more expected findings with an A5.1 category, canonically ordered reason
  codes, and the exact B5.1 rule IDs that support the expectation.

The complete artifact pins both the A5.1 label-contract version and the exact B5.1
rule descriptor. Its fixture and expected-finding counts are validated against the
full inventories.

## Coverage

The 74 fixtures comprise:

| Fixture purpose | Count | What it covers |
| --- | ---: | --- |
| Active indicator | 43 | Every distress, threat, and weapon keyword/phrase rule, plus normalization, punctuation, sentence-boundary, and multi-finding cases |
| Context suppressed | 23 | All 19 explicit-negation rules, two hypothetical/conditional examples, and two quoted/reported examples |
| No concerning match | 6 | Harmless sentences and token-boundary examples such as `helpdesk`, `bombastic`, and `rifleman` |
| Ambiguous | 2 | Short concerning terms without enough context for an active interpretation |

Every one of the 40 active B5.1 rules appears in at least one active-indicator
fixture. Every one of the 19 negation rules appears in a harmless explicit-negation
fixture. The loader rejects incomplete rule coverage, unknown rule IDs, category or
rule-kind mismatches, rule provenance drift, duplicate IDs or text, unordered
inventories, invalid counts, unsupported metadata, and changed artifact bytes.

## Evaluation boundary

The fixture labels state desired A5.1 evidence behavior. A5.3 now verifies all 74
examples against the analyzer, including hypothetical, quoted/reported, and
insufficient-context cases. The comparison covers category, canonical reason-code
order, exact supporting rule IDs, and finding order. Additional tests verify that
context cues cannot cross hard sentence boundaries and that ambiguity handling does
not hide a question containing an object.

The text is synthetic and contains no recordings, personal data, speaker identity,
or real incident claims. No external dataset, model, network access, or download is
required.

## Integrity and reproducibility

The bundled bytes are pinned by SHA-256 and protected from line-ending conversion
in `.gitattributes`. The local-only loader limits the artifact to 1 MiB, validates
the closed Pydantic contract, confirms the complete rule inventory, and returns an
immutable result containing the artifact digest and size.

## Verification

Run the focused fixture and engine checks:

```powershell
python -m pytest tests/test_language_fixtures.py tests/test_language_analysis.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Beginner-friendly explanation

This artifact is a versioned answer key. It supplies examples of direct concerning
language, ordinary harmless text, explicit negations, quoted or hypothetical
language, and unclear short statements. The test suite now grades the engine against
that answer key without changing expectations to fit the output.
