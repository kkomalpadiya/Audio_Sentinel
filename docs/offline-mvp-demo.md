# Offline MVP demonstration

## Goal and boundary

A9.3 provides one repeatable demonstration of the completed offline MVP. The
script evaluates one authorized recording through the public CLI, reloads the
saved final report through its integrity checks, and runs retention in planning
mode. It performs no notification, external action, model download, or deletion.

The demo is for a controlled local audience. It is not a live microphone demo,
an emergency simulation, or evidence that the system is ready for deployment.
Read [`authorized-use-and-limitations.md`](authorized-use-and-limitations.md)
before presenting it.

## Before the session

1. Choose a WAV or FLAC recording already covered by a current external consent
   record and place it under `data/raw/`. Do not use a person's name in the file,
   clip ID, consent ID, or source-dataset label.
2. Decide the least intrusive authorized scope. Use `acoustic_only` unless the
   consent specifically permits speech-content analysis.
3. Record the consent start and optional expiry with a timezone offset. Confirm
   that the recording device and intended demonstration are covered.
4. Select and record the experimental acoustic threshold. The A9.1 value `0.5`
   is reproducible but not production-calibrated.
5. Install the pinned local acoustic model once with
   `./scripts/setup_yamnet.ps1`. For `acoustic_and_speech`, also run
   `./scripts/setup_speech.ps1` and install the speech extra into the YAMNet
   environment as described in [`local-api.md`](local-api.md).
6. Run the full project verification before the presentation. Keep the machine
   offline during the demo if that is part of the presentation claim.

The script checks model directories and runtime packages but never downloads or
repairs them. A missing prerequisite stops before evaluation.

## Preflight command

From the `Project_1` root, choose an offset-aware time that reflects the actual
external consent record. This example uses five minutes before the current local
time only to show PowerShell syntax; do not invent a consent time for a real run.

```powershell
$grantedAt = [DateTimeOffset]::Now.AddMinutes(-5).ToString("o")

& .\.venv\yamnet\Scripts\python.exe .\scripts\run_offline_mvp_demo.py `
  --audio examples/authorized-recording.wav `
  --clip-id demo-clip-001 `
  --consent-id consent-demo-001 `
  --scope acoustic_only `
  --granted-at $grantedAt `
  --source-dataset authorized-local-demo `
  --acoustic-threshold 0.5 `
  --confirm-consent-reviewed `
  --confirm-authorized-device `
  --preflight-only
```

A successful preflight returns `status=preflight_complete`, the required local
model directories, `network_required=false`, and
`retention_apply_authorized=false`. It does not process audio or write a report.

## Run the demonstration

Remove only `--preflight-only` from the reviewed command:

```powershell
& .\.venv\yamnet\Scripts\python.exe .\scripts\run_offline_mvp_demo.py `
  --audio examples/authorized-recording.wav `
  --clip-id demo-clip-001 `
  --consent-id consent-demo-001 `
  --scope acoustic_only `
  --granted-at $grantedAt `
  --source-dataset authorized-local-demo `
  --acoustic-threshold 0.5 `
  --confirm-consent-reviewed `
  --confirm-authorized-device
```

Use forward slashes in `--audio`; it is relative to `data/raw`. For an authorized
speech demo, change the scope to `acoustic_and_speech`. Add `--expires-at` when
the consent has an expiry. The script refuses a future grant or an already expired
record before starting a model.

The terminal shows three bounded steps:

1. Evaluate the recording and atomically save the final report and local audit.
2. Reload the report through the public integrity checker and compare its outcome,
   score, severity, review state, and alert state with the evaluation response.
3. Verify the managed retention inventory and print a dry-run plan without
   supplying `--apply`.

The final JSON contains privacy-minimized artifact references, evidence-branch
statuses, the review requirement, retention counts, and permanent no-delivery
flags. It omits raw audio, transcript text, matched phrases, recipients, and
absolute paths.

## Five-minute speaking guide

### 1. Authorization and scope

Show the two explicit confirmations and explain that the script cannot establish
legal authority. `acoustic_only` prevents VAD, transcription, and language
analysis; the report records those branches as `not_permitted`, not missing.

### 2. Offline pipeline

Explain that the same shared service prepares the clip, runs the pinned local
model, builds acoustic evidence, calculates risk, checks branch agreement, and
chooses one outcome. No stage calls a network service during evaluation.

### 3. Outcome

Read the returned outcome and score without calling the score a probability.
`no_action` does not prove safety, `log` is not an incident, `review` requires a
person, and `alert` is only a pending local candidate.

### 4. Independent verification

Point out that the script does not trust its first response. It reloads the saved
report, rechecks hashes and cross-document identities, and stops if the verified
summary differs.

### 5. Human and delivery boundary

Show `policy_review_required` and the permanent values
`notification_delivery=not_sent`, `alert_delivery_authorized=false`, and
`external_action_authorized=false`. Any real-world decision belongs to the
separate authorized human procedure.

### 6. Data lifecycle

Show that retention was a dry run. It may identify eligible report/audit bundles,
but the demo cannot apply deletion and never targets raw audio. Applied retention
requires a separate reviewed policy and explicit command outside the demo.

## Expected variants

The recording and evidence determine the result; do not promise a particular
outcome before running it.

| Outcome | What to demonstrate |
| --- | --- |
| `no_action` | No configured evidence crossed the policy, but the result is not proof of safety. |
| `log` | The low score is stored locally without becoming an incident or alert. |
| `review` | The reason may be score, uncertainty, missing data, ambiguity, or conflict; a manual workflow is required. |
| `alert` | A local pending record exists, human review remains required, and no notification was sent or authorized. |

Rerunning the same authorized input may reuse content-addressed artifacts. The
`report_reused` and `audit_reused` fields make that behavior visible rather than
silently overwriting prior records.

## Failure and recovery guide

| Safe error | Action |
| --- | --- |
| `consent_confirmation_required` or `device_confirmation_required` | Stop and verify the external authorization. Do not add the flag until the fact is true. |
| `consent_not_yet_active` or `consent_expired` | Correct the record only from the authoritative source. Do not change timestamps merely to make the demo run. |
| `invalid_audio_path` | Use a forward-slash path to a regular WAV or FLAC file below `data/raw`. |
| `runtime_missing` or `model_not_found` | Run the documented setup before the session, then repeat preflight. Setup may need network access; the demo itself does not. |
| `output_conflict`, `invalid_document`, or verification failure | Preserve the files, stop the presentation, and investigate integrity. Do not delete or overwrite the conflicting bundle. |
| `delivery_boundary_failed`, `result_mismatch`, or `retention_boundary_failed` | Treat the demonstration as failed. Do not present or act on the output. |

CLI and stage errors remain structured and privacy-safe. Unexpected exceptions are
reduced to a generic error instead of printing local paths or sensitive content.

## After the session

- Record the actual command parameters, outcome, reviewer, and audience in the
  approved project record without copying sensitive audio or transcript text.
- Complete any required human review outside Audio Sentinel.
- Use the separate retention procedure only after reviewing its dry-run plan.
  The demo does not remove raw audio, intermediate evidence, backups, or exports.
- Do not describe the session as live monitoring, notification delivery, an
  emergency response, or production validation.

## Verification

The focused suite uses temporary files and contract-compatible JSON responses; it
does not load TensorFlow, ONNX Runtime, Whisper, or any model artifact:

```powershell
python -m pytest tests/test_offline_mvp_demo.py -q
```

It covers authorization confirmations, path containment, active timestamps,
scope-preserving commands, report revalidation, retention dry-run enforcement,
delivery-boundary failure, and model-free help output.
