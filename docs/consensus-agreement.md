# Evidence agreement and conflict detection

## What B7.1 adds

B7.1 implements the deterministic agreement engine in
`src/audio_sentinel/consensus_rules.py`. It accepts one validated Phase 6
`RiskAssessmentDocument`, verifies it again, applies a SHA-256-pinned local rule
artifact, and returns a privacy-minimized `ConsensusAgreementEvaluation`.

The built-in rules are in
`src/audio_sentinel/resources/consensus-agreement-rules-v1.json`. The evaluation
and rule schemas are checked in as
`docs/schemas/v1/consensus-agreement.schema.json` and
`docs/schemas/v1/consensus-agreement-rule-set.schema.json`. A complete result over
the existing Phase 6 example is in `docs/examples/consensus-agreement.json`.

This task classifies agreement only. It does not choose `no_action`, `log`,
`review`, or `alert`; A7.2 owns that final decision.

## Branch classification

The engine returns acoustic, speech, and language evaluations in canonical order.
Each contains the Phase 6 input status, one A7.1 agreement state, and ordered reason
codes.

| Branch | Support rule | Neutral rule |
| --- | --- | --- |
| Acoustic | At least one risk-bearing v1 label has a peak score at or above 0.85. | There is no risk-bearing label, or every risk-bearing label is below 0.85. |
| Speech | Never supports risk by itself. | VAD speech presence or absence is internally consistent with the other branches. |
| Language | At least one `distress`, `threat`, or `weapon_reference` finding is present. | Findings are absent, safe, context-suppressed, or ambiguous. Ambiguity remains a separate Phase 6 review trigger. |

All nine risk-bearing acoustic labels are explicitly listed in the rule artifact.
All six language categories are partitioned between support and neutral lists.
Load-time validation rejects omissions, reordering, duplicates, unknown fields, or
attempts to disable required v1 conflict checks.

## Conflict rules

The built-in engine records a conflict when any enabled contradiction occurs:

1. High-confidence acoustic `no_speech` conflicts with one or more VAD speech
   segments. Language also conflicts when findings claim analyzed speech in that
   case.
2. A high-confidence acoustic speech-like label conflicts with zero VAD speech
   segments.
3. High-confidence `non_threatening_speech` conflicts with active distress, threat,
   or weapon-reference language.
4. Language findings conflict with a speech summary that has zero accepted
   transcripts.

A conflict overrides support or neutral for the involved branches and is retained
with its exact reason code. Non-present branches are never converted into a
conflict: `missing`, `not_permitted`, `not_applicable`, and `no_accepted_text` map
to their A7.1 states without inventing evidence.

## Provenance and determinism

Every result records:

- the Phase 6 assessment ID and canonical document SHA-256;
- the exact agreement rule ID, version, format version, and artifact SHA-256;
- the complete branch classifications and reason codes;
- ordered supporting and conflicting branch lists; and
- a deterministic evaluation ID derived from the assessment hash, rule descriptor,
  and branch results.

The evaluation timestamp is excluded from identity, so rerunning the same assessment
with the same rules produces the same evaluation ID. Changing evidence or the rule
artifact changes that identity.

## Privacy and limitations

The evaluator uses the privacy-minimized Phase 6 summary. It stores no raw audio,
transcript text, matched phrases, speaker identity, paths, alert recipient, or
delivery state. Failures return stable messages without echoing private input.

The 0.85 threshold is an explicit deterministic v1 policy value, aligned with the
Phase 6 high-confidence threshold. It is not an empirically calibrated probability
of an incident. Representative end-to-end accuracy is still evaluated later.

## Verification

Run the focused B7.1 and affected Phase 6 tests:

```powershell
python -m pytest tests/test_consensus_rules.py tests/test_risk_contracts.py -q
```

Run all project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Next task

A7.2 will verify the B7.1 evaluation provenance and apply the A7.1 outcome policy
to create the final consensus decision.

## Beginner-friendly explanation

This is the cross-checker. It asks whether the sound model, speech detector, and
language analyzer tell a compatible story. Strong sound and concerning language
can support risk, ordinary speech remains neutral, and contradictions are marked
for review instead of being averaged away.
