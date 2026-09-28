# Local alert creation and audit records

A8.2 consumes one saved and revalidated B8.1 final report, records how it was
handled locally, and creates a local alert document only when the trusted consensus
outcome is `alert`.

This feature does not send a notification. It has no recipient, address book,
network client, webhook, message queue, or `AlertPublisher` implementation.

## Public API

```python
from audio_sentinel.alert_audit import (
    load_alert_audit,
    save_alert_audit,
)

saved = save_alert_audit(
    settings.paths,
    f"final-reports/{report_id}/report.json",
)

loaded = load_alert_audit(
    settings.paths,
    f"alert-audit/{saved.audit.audit_id}/audit.json",
)
```

Both paths are relative to `data/processed`. Production calls should omit `now`.
A fixed timezone-aware value is supported for tests and controlled validation; it
cannot predate the final report.

## Outcome behavior

Every verified final report receives one immutable `AlertAuditRecord`.

| Consensus outcome | Audit action | Local alert |
| --- | --- | --- |
| `alert` | `local_alert_created` | Created as local-only and pending review |
| `review` | `no_alert_created` | Not created |
| `log` | `no_alert_created` | Not created |
| `no_action` | `no_alert_created` | Not created |

A non-alert outcome cannot be relabeled as alert creation. An alert outcome must
produce exactly one local alert reference and retain `review_required=true`.

## Local alert contract

`LocalAlertDocument` is strict, immutable, and versioned at `1.0`. It contains:

- a deterministic semantic `alert_id`;
- a hash-pinned final-report reference;
- the clip, decision, and assessment identities from that report;
- the critical score and severity;
- the consensus decision reasons;
- `review_status=pending` and `review_required=true`; and
- permanent local-only and no-delivery flags.

It excludes audio, transcripts, matched phrases, tensors, local paths, recipient
data, and transport configuration.

## Audit contract

`AlertAuditRecord` records the final-report reference, decision outcome, local
action, action reason, review requirement, and optional local-alert reference. The
alert reference uses a semantic SHA-256 that excludes only the alert's own identity
and creation timestamp. The audit ID likewise excludes its own identity and
timestamp, so retrying the same report remains idempotent.

Every alert and audit document contains:

```json
{
  "notification_delivery": "not_sent",
  "alert_delivery_authorized": false
}
```

Creating a local record is therefore never equivalent to approving or delivering a
notification. A future review workflow must define any later state transition; A8.2
does not implement one.

## Storage and verification

Bundles are stored at:

```text
data/processed/alert-audit/<audit_id>/audit.json
data/processed/alert-audit/<audit_id>/alert.json  # alert outcomes only
```

Persistence:

1. reloads and fully verifies the referenced final report;
2. derives the audit and optional alert from that report;
3. writes a private staging bundle with bounded file sizes;
4. flushes and validates every staged document;
5. checks the documents against the final report again;
6. atomically publishes the completed directory; and
7. reloads the published bundle, including its referenced final report.

Identical retries reuse the existing immutable bundle. A conflict is never
overwritten. Loading rejects linked or escaping paths, malformed documents,
oversized files, changed identities, missing referenced reports, incorrect bundle
inventory, and cross-document mismatches.

The final report is found from its ID at the canonical B8.1 location; no filesystem
path is stored in the audit documents. SHA-256 detects content changes but is not a
digital signature or proof of authorship.

## JSON Schemas

The generated v1 schemas are checked in at:

- `docs/schemas/v1/alert-audit-record.schema.json`
- `docs/schemas/v1/local-alert.schema.json`

Regenerate them from the Python contracts with:

```powershell
python -c "from pathlib import Path; from audio_sentinel.alert_audit import write_alert_audit_schemas; write_alert_audit_schemas(Path('docs/schemas/v1'))"
```

The Python validators additionally enforce identities and cross-document alignment
that JSON Schema alone cannot express.

## Stable errors

`AlertAuditError.code` is safe for callers and future CLI/API mapping. Important
codes include `invalid_report`, `invalid_time`, `invalid_path`, `file_too_large`,
`output_too_large`, `invalid_output_path`, `output_conflict`, `bundle_mismatch`,
`report_unavailable`, `invalid_document`, `verification_failed`,
`insufficient_memory`, and `write_failed`.

## Verification

Run the focused suite:

```powershell
python -m pytest tests/test_alert_audit.py -q
```

The tests cover alert and non-alert outcomes, review state, privacy exclusions,
semantic identities, time ordering, atomic persistence, retry reuse, exact bundle
inventory, size limits, unsafe paths, tampering, wrong identity directories,
missing final reports, strict immutability, and checked-in schema parity. They use
deterministic local test doubles and make no network request.
