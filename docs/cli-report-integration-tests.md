# CLI and final-report integration tests

## What B8.3 verifies

B8.3 adds integration coverage across the B8.2 command boundary and the persisted
Phase 8 artifacts. The tests use the real preparation, acoustic inference,
aggregation, evidence persistence, speech orchestration, language analysis, risk,
consensus, final-report, and alert-audit code. Only the three large model runtimes
are replaced with deterministic in-process models that satisfy the real metadata
contracts.

This keeps the standard test suite offline and repeatable while exercising the
same orchestration and persistence code used by the installed CLI.

## Covered workflows

The eight integration tests cover:

1. An acoustic-only CLI run, followed by public report and audit reloads. Speech
   and language receipts must be `not_permitted`, and no local alert may exist.
2. A speech-authorized run that reaches an alert outcome. The report, audit, and
   local alert must agree, and the alert must remain pending, local-only, unsent,
   and unauthorized for delivery.
3. Two separate CLI runs over the same recording. Each run is a new time-stamped
   assessment, so the second produces new report and audit identities while the
   first immutable bundle remains byte-for-byte unchanged.
4. Report inspection in a fresh Python process through
   `python -m audio_sentinel.cli`, proving that the persisted report can be used
   after all in-memory evaluation objects are gone.
5. Tampered report content. Inspection must fail through the CLI's structured
   error channel without printing the damaged field or a local path.
6. An unexpected file added to a report bundle. Inspection must reject the bundle
   inventory and must not print that file's contents.
7. Relative traversal aimed outside `data/processed`. Inspection must reject the
   path before reading the outside file.
8. Portable output paths. Report, audit, and alert paths must be relative,
   forward-slash paths whose directory identities match their returned IDs.

## Privacy and safety assertions

The test matrix searches the CLI output and persisted Phase 8 documents for the
synthetic transcript and local project root. Neither may appear. It also verifies
that success uses standard output only, failures use standard error only, compact
inspection is valid one-line JSON, and no failure response echoes deliberately
placed private content.

These tests do not change thresholds, model behavior, risk weights, consensus
rules, alert eligibility, or notification policy. They test the completed workflow
rather than introducing another implementation.

## Why separate runs do not report reuse

Final reports embed a time-stamped risk assessment and decision. Repeating the CLI
command later is a new assessment, even for identical audio, consent, and policy.
It therefore receives new semantic report and audit identities. The persistence
layer's reuse behavior still applies when the same already-built typed result is
saved again; that lower-level behavior remains covered by the B8.1 and A8.2 tests.

This distinction prevents a later assessment from silently overwriting or being
mistaken for an earlier one.

## Run the tests

```powershell
python -m pytest tests/test_cli_report_integration.py -q
```

The tests require no network access or model download. They create isolated
temporary project roots and remove them through pytest's temporary-directory
lifecycle.

## Next task

A8.4 will add end-to-end API and evaluator tests, including the HTTP boundary added
in A8.3.

In plain language: B8.3 proves that the command line does more than print plausible
JSON. Its identifiers lead to real, independently reloadable records; later
processes can inspect those records; and path, tampering, inventory, privacy, and
immutability controls still hold across the complete workflow.

