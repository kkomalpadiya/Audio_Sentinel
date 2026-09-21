# Versioned language rule data

## What B5.1 adds

B5.1 provides the first offline English rule artifact for Phase 5. The bundled data
is `src/audio_sentinel/resources/language-rules-en-v1.json`. Its contract and
integrity-checked loader are in `src/audio_sentinel/language_rules.py`, and its
portable schema is `docs/schemas/v1/language-rule-set.schema.json`.

This task creates data and validation only. It does not scan transcripts or create
language findings. A5.2 will implement matching, negation scope, character offsets,
and the A5.1 evidence document.

## Pinned artifact

| Property | Value |
| --- | --- |
| Rule-set ID | `audio-sentinel-en-v1` |
| Rule-set version | `1.0.0` |
| Format/schema version | `1.0` |
| Language | English (`en`) |
| Artifact SHA-256 | `24794b01df242c37364f8c8c0740fa14f86c1d64d6d7f1076d03bec1c58d8488` |
| Total rules | 59 |
| Keywords | 17 |
| Phrases | 23 |
| Explicit negations | 19 |

The loader reads package data locally, limits the artifact to 1 MiB, verifies the
exact byte hash, validates every field, and checks the pinned ID and version. It
does not import an ML runtime or use the network.

## Rule meanings

Keyword and phrase rules can support only the three active A5.1 categories:
`distress`, `threat`, and `weapon_reference`. Every category has both keyword and
phrase coverage. Negation rules have no category; they can only supply the
`explicit_negation` reason used by a later `context_suppressed` finding.

The inventory includes examples such as:

- distress: `help`, `emergency`, `call an ambulance`, `i cannot breathe`;
- threat: `kill`, `shoot`, `i will kill you`, `you are going to die`;
- weapon reference: `gun`, `knife`, `bomb`, `has a gun`, `armed with`; and
- negation: `no`, `not`, `never`, `do not`, `don't`, `without`, `unarmed`.

These are candidate indicators, not conclusions about intent or events. Broad terms
such as `help`, `shoot`, `bomb`, and `knife` can appear in harmless discussion,
reporting, entertainment, idioms, or instructions. A rule hit must therefore stay
traceable and pass A5.2 context handling and A5.3 false-positive tests. It must not
be converted directly into risk or an alert.

## Normalization contract

The artifact fixes the input normalization that A5.2 must implement:

1. Unicode NFKC normalization.
2. Unicode case folding.
3. Curly apostrophes converted to the ASCII apostrophe and preserved inside words.
4. Tokens restricted to lowercase English letters with at most one internal
   apostrophe.

Patterns are arrays of normalized tokens rather than regular expressions. This
keeps matching inspectable and avoids regex execution from configuration data.
There is no stemming, fuzzy matching, synonym expansion, translation, or semantic
model in this baseline.

## Negation boundary

The versioned starting window is three tokens. A5.2 must apply it conservatively
and record both the concerning match and the exact negation match when suppression
occurs. The existence of a negation word elsewhere in the transcript must not erase
an unrelated finding. The window is an engineering baseline, not an evaluated
linguistic guarantee.

## Integrity rules

Validation rejects:

- duplicate rule IDs or duplicate patterns within the same kind;
- non-normalized or malformed tokens;
- unordered rule inventories;
- a keyword with multiple tokens or a phrase with fewer than two;
- a keyword/phrase without an active category;
- a negation that assigns a category;
- a kind with the wrong A5.1 reason code;
- missing keyword or phrase coverage for an active category; and
- unknown fields, versions, languages, or normalization settings.

## Verification

Run the focused checks:

```powershell
python -m pytest tests/test_language_rules.py -q
```

Run all project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

No download, model installation, or dataset is needed.

## Next task

A5.2 will implement the transcript-analysis engine over accepted A4.3 downstream
transcripts using this exact artifact and the A5.1 evidence contract.

## Beginner-friendly explanation

This file is a checked dictionary for the next analysis step. It lists concerning
words and phrases, plus words that may negate them. The checksum makes an unnoticed
dictionary change fail loudly. The dictionary itself does not decide that a threat
occurred; the next task will apply it with position and context rules.
