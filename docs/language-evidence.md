# Language evidence contract

## What A5.1 adds

A5.1 defines the Phase 5 language categories, reason codes, and portable evidence
format before the rule data or analysis engine is implemented. The Python contract
is in `src/audio_sentinel/language_contracts.py`, the JSON Schema is in
`docs/schemas/v1/language-evidence.schema.json`, and the complete example is in
`docs/examples/language-evidence.json`.

This task defines evidence. It does not create language rules, analyze a transcript,
assign risk, decide that an incident occurred, or send an alert. B5.1 owns the rule
data and A5.2 owns the analysis engine.

## Accepted-only input boundary

Phase 5 may consume only A4.3 `downstream_transcripts`. Every transcript reference
therefore has the literal values `transcript_reliability: accepted` and
`downstream_text_allowed: true`. Review-required, rejected, and missing transcripts
cannot be represented as valid automatic language inputs.

The source record links the result to the complete Phase 4 evidence by ID and SHA-256
hash and retains the clip, consent, sample-rate, and length identifiers needed to
check provenance. The final A5.2 service must validate those links against the real
Phase 4 result; this schema does not turn a self-asserted reference into proof.

## Categories

Each finding uses exactly one of six categories:

| Category | Meaning |
| --- | --- |
| `no_concerning_match` | No versioned concerning-language rule matched. This is not proof that the speech or clip is safe. |
| `distress` | Accepted words support a distress-language indicator. |
| `threat` | Accepted words support a threatening-language indicator. |
| `weapon_reference` | Accepted words support a weapon-reference indicator. |
| `ambiguous` | A concerning rule matched, but the available text has insufficient or conflicting context. |
| `context_suppressed` | A concerning rule matched, but explicit negation, hypothetical/conditional wording, or quoted/reported context suppresses that match from becoming an active indicator. |

Categories are evidence labels, not intent, truth, speaker identity, incident type,
severity, or risk. Multiple active findings can occur in one transcript. A
`no_concerning_match` finding must be the transcript's only finding.

## Reason codes and matches

Reason codes make every category inspectable:

| Reason code | Use |
| --- | --- |
| `no_rule_match` | Required by `no_concerning_match`; no rule-match records may be present. |
| `keyword_match` | A versioned keyword rule supports the finding. |
| `phrase_match` | A versioned phrase rule supports the finding. |
| `explicit_negation` | A negation rule suppresses an otherwise concerning match. |
| `hypothetical_or_conditional` | Hypothetical or conditional context suppresses the match. |
| `quoted_or_reported_speech` | Quoted or reported wording suppresses the match. |
| `insufficient_context` | The accepted text is too limited to resolve the match. |
| `conflicting_signals` | Rule evidence points to incompatible interpretations. |

Distress, threat, and weapon-reference findings allow only keyword/phrase reasons.
Ambiguous findings require an ambiguity reason. Context-suppressed findings require
a suppression reason. Every non-empty match records the rule ID and kind, exact
character span, and SHA-256 of the matched text.

## Privacy and explainability

The language artifact does not duplicate complete transcript text. It retains the
accepted transcript hash, character and byte counts, and rule-match character spans.
This is enough to verify the evidence against the authorized Phase 4 result while
reducing repeated sensitive text. It also excludes raw audio, absolute paths,
speaker identity, embeddings, incident decisions, severity, risk scores, consensus
outcomes, and alerts. Unknown fields are forbidden.

## Integrity rules

The contract checks that:

- summary counts match the complete accepted-transcript and finding inventories;
- transcript IDs are unique and analyses are in deterministic chronological order;
- transcript spans are non-overlapping and stay within the prepared source;
- seconds exactly correspond to sample offsets and the recorded sample rate;
- findings, reason codes, and rule matches use deterministic ordering;
- rule-match spans stay inside their accepted transcript; and
- the checked category, reasons, and rule kinds are mutually compatible.

An empty accepted-transcript inventory is valid and records zero counts. It means
that Phase 4 supplied no accepted text; it does not mean that the recording was safe.

## Verification

Run the focused contract tests:

```powershell
python -m pytest tests/test_language_contracts.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Next task

B5.1 will create the versioned keyword, phrase, and negation rule artifact. It must
use these categories and reason-code boundaries without embedding severity or risk.

## Beginner-friendly explanation

This contract is a receipt for text analysis. It says which accepted transcript was
checked, which versioned rule matched, where it matched, and why the result received
its category. Negated or unclear language stays visibly suppressed or ambiguous
instead of being promoted to a threat. The receipt still does not decide whether an
incident happened.
