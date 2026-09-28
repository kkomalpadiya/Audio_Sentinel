# Versioned final evaluation report

B8.1 turns one completed `OfflineEvaluationResult` into a portable, versioned JSON
report. The report is a durable local evaluation record. It is not an alert message,
does not contact a recipient, and cannot authorize delivery.

## Public API

```python
from audio_sentinel.final_report import (
    build_final_report,
    load_final_report,
    save_final_report,
)

# Build an in-memory, validated document.
report = build_final_report(result)

# Persist atomically and verify by reloading it.
saved = save_final_report(settings.paths, result)

# Paths supplied to the loader are relative to data/processed.
loaded = load_final_report(
    settings.paths,
    f"final-reports/{saved.report.report_id}/report.json",
)
```

Production calls should omit `now`. A fixed timezone-aware value is available for
repeatable tests and controlled validation. Saving also reloads the current acoustic
evidence with the same clock, so its consent and source checks remain active.

## Contract

`FinalReportDocument` is strict, immutable, and fixed at schema and format version
`1.0`. Unknown fields, invalid enum values, naive timestamps, non-finite numbers,
misordered evidence receipts, and inconsistent embedded documents are rejected.

The document contains:

- a `report_id` and UTC `created_at` timestamp;
- privacy-minimized source identity, consent scope, sample counts, and SHA-256
  provenance hashes, but no local file path;
- one canonical acoustic, speech, and language evidence receipt, including the
  branch status and the evidence identity/hash when an artifact exists;
- a small derived summary of counts, score, severity, outcome, review state, and
  alert-candidate state;
- the complete trusted Phase 6 risk assessment;
- the complete B7.1 agreement evaluation; and
- the complete A7.2 final decision.

The report recomputes and checks the risk-assessment identity, assessment hash,
agreement reference, decision reference, decision identity, branch states, evidence
receipts, source identity, and derived summary. A modified nested score, status,
reason, outcome, or reference therefore fails validation unless every dependent
document is rebuilt through the trusted pipeline.

## Privacy and delivery boundary

The serializer deliberately excludes raw audio, prepared waveforms, tensor values,
transcript text, matched phrases, and absolute or relative local artifact paths. It
retains only the privacy-minimized evidence already approved for risk and consensus
processing.

Every report states:

```json
{
  "notification_delivery": "not_sent",
  "alert_delivery_authorized": false
}
```

An `alert` outcome remains a local candidate with human review required. B8.1 has no
network client, recipient field, notification publisher, or delivery side effect.
The A8.2 [local alert and audit layer](local-alert-audit.md) consumes only a saved,
revalidated report and preserves this no-delivery boundary.

## Identity and retry behavior

The `report_id` is `report-` followed by a SHA-256 digest of the canonical semantic
report content. It excludes only the final report's own ID and creation timestamp.
The embedded stage timestamps remain part of the identity. Rebuilding the same
evaluation at another report-creation time therefore produces the same ID, while a
meaningful evaluation change produces another ID.

Reports are stored at:

```text
data/processed/final-reports/<report_id>/report.json
```

Persistence writes a private staging directory, flushes the JSON file, validates a
readback, and atomically renames the completed bundle. A retry reuses an identical
existing report and never overwrites a conflicting bundle. Loading requires exactly
one `report.json` in the identity directory; linked paths, traversal, absolute paths,
unexpected files, oversized documents, wrong identity paths, and invalid JSON are
rejected.

SHA-256 makes content changes detectable; it is not a digital signature and does not
prove who created a report. Operating-system access control and a future signed audit
chain remain separate deployment responsibilities.

## Evidence status behavior

The receipt inventory always uses acoustic, speech, then language order.

| Status | Report behavior |
| --- | --- |
| `present` | Pins the completed evidence identity and SHA-256 hash. |
| `no_accepted_text` | Pins the completed language artifact even though no accepted transcript was available for language scoring. |
| `not_permitted` | Records the consent exclusion and carries no artifact identity or hash. |
| `missing` / `not_applicable` | Carries no artifact identity or hash; a supplied artifact would be rejected. |

For acoustic-only consent, speech and language are both `not_permitted`; their
summary counts are `null`, preserving the distinction between "not processed" and
an authorized branch that completed with zero findings.

## JSON Schema

The generated schema is checked in at
`docs/schemas/v1/final-report.schema.json`. Generate it from the Python contract with:

```powershell
python -c "from pathlib import Path; from audio_sentinel.final_report import write_final_report_schemas; write_final_report_schemas(Path('docs/schemas/v1'))"
```

The schema is useful for non-Python structural validation. The Python model performs
the additional cross-document hash, identity, ordering, and summary checks that JSON
Schema cannot express.

## Stable errors

`FinalReportError.code` is safe for callers and audit handling. Important codes
include `invalid_evaluation`, `invalid_time`, `source_mismatch`, `missing_evidence`,
`evidence_mismatch`, `invalid_output_path`, `invalid_path`, `file_too_large`,
`output_too_large`, `output_conflict`, `source_changed`, `invalid_document`,
`verification_failed`, `insufficient_memory`, and `write_failed`.

## Verification

Run the focused suite:

```powershell
python -m pytest tests/test_final_report.py -q
```

The suite covers the full and acoustic-only consent paths, no-accepted-text behavior,
semantic IDs, atomic save/load/reuse, size limits, unsafe paths, tampering, unexpected
bundle files, wrong identity directories, invalid evaluator handoffs, immutability,
privacy exclusions, and checked-in schema parity. Deterministic local test doubles
exercise the real evaluator and contracts without a model download or network call.
