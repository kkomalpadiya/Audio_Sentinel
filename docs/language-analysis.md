# Transcript-analysis engine

## What A5.2 adds

A5.2 implements deterministic language analysis in
`src/audio_sentinel/language_analysis.py`. It connects the accepted-only A4.3
handoff to the B5.1 English rule artifact and produces the A5.1 portable language
evidence document.

The engine is intentionally a rule matcher, not a risk scorer. Its output records
candidate language indicators and context suppression. It does not infer speaker
identity or intent, decide that an incident occurred, assign severity or risk, or
send an alert.

## Verified Phase 4 boundary

`analyze_accepted_transcripts` accepts a complete `SpeechTranscriptionResult`. Before
matching, it:

1. revalidates the Phase 4 speech-evidence contract;
2. recomputes the semantic speech-evidence identifier;
3. reconstructs the downstream inventory from segments whose recorded reliability
   is `accepted` and whose policy allows downstream text; and
4. requires the supplied A4.3 `downstream_transcripts` tuple to match that inventory
   exactly, including order, text, sample span, timestamps, confidence score, and
   confidence kind.

Review-required, rejected, and untranscribed candidates are therefore excluded.
Removing, adding, reordering, or changing accepted text makes the whole operation
fail without returning partial evidence.

## Matching behavior

The engine follows the normalization recipe stored in the loaded rule artifact:

- Unicode NFKC normalization;
- Unicode case folding;
- curly apostrophes converted to ASCII apostrophes; and
- lowercase English-word tokens with at most one internal apostrophe.

Matching uses token arrays from configuration rather than executable regular
expressions. Original character positions are retained while text is normalized,
so each portable match points back to the exact accepted transcript substring and
stores that substring's SHA-256 digest. The evidence document stores the complete
transcript digest and counts, but does not duplicate the transcript text.

When a longer phrase fully contains a shorter same-category match, the phrase wins.
For example, `please help me` produces one phrase finding rather than both a phrase
and a contained `help` keyword finding. Separate occurrences remain separate and
are emitted in deterministic character order.

## Context safeguards

The rule artifact sets a three-token negation window. A negation can suppress a
concerning match only when it:

- ends before the concerning match begins;
- is no more than three intervening tokens away; and
- does not cross a hard sentence or clause boundary (`.`, `!`, `?`, `;`, `:`, or a
  newline).

The nearest eligible negation wins; a longer negation phrase wins a tie. A
suppressed finding records both exact matches and both the original keyword/phrase
reason and `explicit_negation`. A negation inside an active phrase does not suppress
that phrase, so `I cannot breathe` remains the configured distress phrase.

A5.3 adds three bounded safeguards after explicit-negation evaluation:

- modal speech framing such as `a character might say ...` and conditional speech
  framing such as `if I said ...` produce `context_suppressed` with
  `hypothetical_or_conditional`;
- report/statement framing and explicit quotation cues produce
  `context_suppressed` with `quoted_or_reported_speech`; and
- a keyword-only modal question with no object, or a keyword introduced only as a
  mentioned/referenced topic, produces `ambiguous` with `insufficient_context`.

These cues operate only inside the same hard-bounded clause as the match. Phrase
matches and keyword matches with meaningful trailing objects are not downgraded by
the ambiguity safeguard. Explicit negation takes precedence when more than one
context cue applies. These rules reduce known false positives; they do not claim to
infer intent or exhaust every form of quotation, hypothetical language, or
ambiguity.

## Determinism, provenance, and limits

The result pins the exact speech evidence by semantic ID and canonical SHA-256 and
pins the exact rule artifact by ID, version, format version, and artifact SHA-256.
Finding IDs and the language-evidence ID are content-derived and stable across
creation times. A fresh timezone-aware `created_at` remains an audit timestamp.

Default resource limits bound accepted transcript count, total UTF-8 bytes, total
tokens, indexed rule checks, and total findings. Invalid input, rule provenance,
time, or resource use raises a stable `LanguageAnalysisError` code with a safe
message that does not echo transcript text. The engine is local-only and needs no
network, model, dataset, or download.

## Verification

Run the focused Phase 5 checks, including the complete B5.2 fixture matrix:

```powershell
python -m pytest tests/test_language_analysis.py tests/test_language_contracts.py tests/test_language_rules.py tests/test_language_fixtures.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Evaluation fixtures

B5.2 supplies the versioned fixture matrix documented in
`docs/language-fixtures.md`. A5.3 executes all 74 fixtures against this engine and
compares every category, reason code, and supporting rule ID with the expected
labels.

## Beginner-friendly explanation

The project now takes only transcripts that the previous phase explicitly accepted,
checks them against the versioned dictionary, and creates a receipt for every match.
It remembers exactly where the words appeared and whether nearby wording such as
`do not`, `might say`, or `the report said` changes how they should be understood.
Short unclear uses are labeled for review instead of being treated as definite. A
match is still only evidence for later stages; it is not an emergency decision by
itself.
