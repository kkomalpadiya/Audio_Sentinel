# Retention and deletion controls

## Scope

B9.2 adds an explicit, local cleanup boundary for the durable Phase 8 output
bundles in `data/processed`:

- `final-reports/<report_id>/`; and
- `alert-audit/<audit_id>/`, including its optional pending local alert.

The service does not delete raw recordings, prepared audio, Log-Mel arrays,
acoustic/speech/language evidence, model files, dataset files, or external data.
Its deletion receipt permanently records `raw_audio_deleted=false`,
`notification_delivery=not_sent`, and
`external_deletion_authorized=false`.

## Settings

`RetentionSettings` is strict, immutable, and disabled by default:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `enabled` | `false` | An applied deletion is rejected unless this is explicitly `true`. |
| `final_report_days` | `30` | Minimum stored age for a final report; `null` retains reports indefinitely. |
| `alert_audit_days` | `365` | Minimum stored age for an alert/audit bundle; `null` retains audits indefinitely. |
| `allow_pending_alert_deletion` | `false` | Pending local alerts remain protected unless explicitly enabled. |
| `max_delete_count` | `1000` | Maximum combined bundles in one plan; larger plans fail before changes. |

The checked-in [`configs/retention.example.json`](../configs/retention.example.json)
keeps deletion disabled. Copy it to an operator-controlled project-local file,
choose values required by the applicable consent and retention policy, review the
dry run, and only then set `enabled` to `true` if deletion is authorized.

Age uses each verified document's timezone-aware `created_at`, not filesystem
modification time. A bundle becomes eligible when its age is at least the configured
number of days.

## Dependency and safety rules

The planner verifies every managed final-report and alert-audit bundle through its
existing public loader before selecting anything. It rejects linked directories,
noncanonical bundle names, unexpected bundle contents, tampered documents, broken
references, and path escapes. One invalid bundle stops the entire operation before
deletion.

Alert/audit bundles are evaluated first. An expired final report remains protected
while any retained audit references it. This means a longer audit-retention setting
also extends the effective lifetime of its referenced report. A pending local alert
protects both its audit and report unless
`allow_pending_alert_deletion=true` was explicitly configured.

Callers provide no artifact path or identifier. The service discovers only the two
fixed managed directories, creates a bounded canonical plan, and deletes only the
verified identities in that plan. Raw data is outside the service's target set.

## Dry run and apply

Planning is the default and does not alter artifacts or create a deletion receipt:

```powershell
audio-sentinel retention `
  --project-root . `
  --retention-config configs/retention.example.json
```

The JSON response includes scanned counts, protected counts, and the ordered target
identities. It contains no absolute paths, audio names, transcript text, or consent
identifiers.

Application requires both `enabled=true` in the supplied settings and the explicit
`--apply` flag:

```powershell
audio-sentinel retention `
  --project-root . `
  --retention-config configs/retention.local.json `
  --apply
```

There is no scheduled or automatic cleanup. An operator or future authorized local
scheduler must invoke the command.

## Local deletion receipt

An applied plan writes one immutable JSON receipt at:

```text
data/processed/retention-audit/<retention_id>/audit.json
```

The receipt contains the exact policy snapshot, execution time, scanned and
protected counts, and each deleted bundle's opaque identity, kind, and original
creation time. It contains no filesystem path, audio, transcript, phrase match,
recipient, transport, or external-deletion authority.

Before deletion, selected bundles are moved into a private same-filesystem staging
directory. If the receipt cannot be written and verified, every moved bundle is
restored and the private directory is removed. After a verified receipt is
published, the private staged copies are removed. Retention audit receipts are not
targets of this cleanup service, so the record of deletion is not erased by the
same action it records.

The v1 JSON Schema is checked in at
`docs/schemas/v1/retention-audit.schema.json`. Reloading a receipt enforces its
size limit, strict schema, canonical content identity, canonical location, exact
one-file inventory, and path containment.

## Stable failures

Expected failures expose safe codes including `retention_disabled`,
`delete_limit_exceeded`, `invalid_inventory`, `inventory_changed`,
`invalid_output_path`, `output_too_large`, `output_conflict`,
`verification_failed`, `rollback_failed`, `cleanup_failed`, `delete_failed`, and
`invalid_document`. Unexpected exception details and local paths are not printed by
the CLI.

## Verification

```powershell
python -m pytest tests/test_retention.py tests/test_config.py tests/test_cli.py -q
```

The focused tests cover strict settings, project-local configuration, dry runs,
explicit authorization, age selection, report/audit dependency protection,
pending-alert protection and opt-in, bounded deletion, tamper failure, rollback,
audit-only deletion, immutable receipt loading, privacy exclusions, size limits,
schema parity, and CLI behavior.

## Limitations

- Policy values are technical settings, not legal advice. The project owner must
  choose them from applicable consent, contractual, and legal requirements.
- The command is local and single-process. Operators must stop concurrent writers
  before applying retention.
- Retention audit receipts are intentionally retained indefinitely in B9.2. A
  separately authorized audit-log archival policy would be needed to remove them.
- Cleanup currently covers Phase 8 final reports and alert/audit bundles only.
  Earlier derived artifacts need a dependency model before automatic deletion can
  be safe.
- Filesystem access control, encrypted storage, secure media erasure, backups, and
  deletion from external copies remain operating-system or deployment concerns.
