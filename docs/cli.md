# Offline evaluation CLI

## What B8.2 adds

B8.2 provides two local commands around the existing Phase 8 services:

- `evaluate` runs one authorized recording through the A8.1 evaluator, saves the
  B8.1 final report, and then saves the A8.2 audit and optional local alert.
- `inspect-report` reloads a saved report through the B8.1 integrity checks and
  prints a privacy-minimized summary.

The CLI does not reimplement model inference, risk scoring, consensus, report
validation, or alert rules. It also does not send notifications. Every result still
records `notification_delivery=not_sent` and
`alert_delivery_authorized=false`.

## One-time setup

From the `Project_1` repository root, install the project so the console command is
available:

```powershell
pip install -e .[dev]
```

Evaluation uses the already selected local models. If they have not been installed
on this machine, run:

```powershell
.\scripts\setup_yamnet.ps1
.\scripts\setup_speech.ps1
```

The speech setup is needed only for `acoustic_and_speech`. The CLI never downloads
a model during evaluation.

## Evaluate a recording

Place a consent-covered WAV or FLAC file inside `data/raw`. Pass its path relative
to that directory, not an absolute path:

```powershell
audio-sentinel evaluate `
  --audio examples/authorized-recording.wav `
  --clip-id demo-clip-001 `
  --consent-id consent-demo-001 `
  --scope acoustic_only `
  --granted-at 2026-09-01T09:00:00+05:30 `
  --device-authorized `
  --source-dataset local-authorized-demo `
  --acoustic-threshold 0.5
```

`--device-authorized` is an explicit confirmation; omitting it stops evaluation.
The grant and optional expiry must include a timezone offset. The source dataset is
a documented collection name, not a person name.

The acoustic threshold is deliberately required because the research result from
A3.4 did not establish one production-calibrated threshold. Supplying a value makes
the experimental policy visible and auditable instead of hiding a default.
An optional `--audio-config` must also be a file inside the Project 1 repository;
absolute paths and paths that escape the repository are rejected.

For speech-authorized processing, use:

```powershell
--scope acoustic_and_speech
```

That scope loads both verified speech models. `acoustic_only` does not load or run
either speech model.

Successful output is JSON containing the outcome, risk score, report and audit
identities, and paths relative to `data/processed`. An alert outcome also includes
the optional local alert identity and path. Output never contains transcript text,
raw samples, tensors, or absolute filesystem paths. Add `--compact` for one-line
JSON.

You can always run the command without installing the console entry point:

```powershell
python -m audio_sentinel.cli evaluate --help
```

## Inspect a report

Use the relative `report_path` returned by `evaluate`:

```powershell
audio-sentinel inspect-report `
  final-reports/report-<sha256>/report.json
```

Inspection is not a raw file print. It uses `load_final_report`, so path containment,
size, schema, semantic identity, evidence references, decision alignment, and bundle
inventory are rechecked first. The JSON view includes source hashes, evidence
receipts, risk/decision identities, and the derived summary, but excludes sensitive
text and local paths.

## Errors and exit codes

Command or pipeline failures return a small JSON object on standard error:

```json
{"code":"file_not_found","error":"The source file or raw data directory does not exist."}
```

Expected domain failures retain their stable error code. Unexpected exceptions are
replaced with `unexpected_error` and a generic message so internal or sensitive
details are not printed. Runtime failures exit with status `1`; malformed command
syntax is handled by Typer and exits with status `2`.

## Verification

Run the focused CLI tests with:

```powershell
python -m pytest tests/test_cli.py -q
```

The tests cover command discovery, both consent scopes, speech-model gating, report
and audit persistence, local alert creation, report revalidation, privacy-minimized
output, compact JSON, explicit device authorization, timezone validation, unsafe
report and config paths, and unexpected-error redaction.

## Next task

A8.3 will expose the local evaluation boundary through a validated API endpoint.

In plain language: B8.2 gives an operator a safe front door to the pipeline. The
operator must state the consent facts and experimental threshold, while the existing
services retain control of every model, scoring, verification, and persistence rule.
