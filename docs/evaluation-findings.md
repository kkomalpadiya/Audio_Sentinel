# A9.1 held-out end-to-end evaluation findings

## Purpose and decision rule

This task runs a predeclared, labeled collection through the same Phase 8
evaluation service used by the local CLI and API. It asks one narrow binary
question: for a clip whose source label is one of the configured target classes,
did the complete pipeline produce any non-`no_action` outcome? `log`, `review`,
and `alert` therefore count as positive outcomes. This is an end-to-end signal
detection check, not a claim that the exact risk level or consensus outcome was
correct.

The evaluation used a fixed `0.5` acoustic candidate threshold. It was selected
before this run and was not tuned against these 32 cases. Exact expected consensus
outcomes were deliberately omitted: ESC-50 and UrbanSound8K provide acoustic class
labels, not the human-reviewed action labels that would be needed to score exact
`no_action`/`log`/`review`/`alert` agreement.

## Reproducible inputs

| Item | Pinned value |
| --- | --- |
| A9.1 manifest ID | `evaluation-manifest-b643e0f3913f8223415cf662` |
| A9.1 manifest SHA-256 | `24da0b0fe6a352f00c1b79e9830db8ce79e8cd3f942776a6ea91be8c5dffefe9` |
| Selection seed | `audio-sentinel-a9.1-heldout-v1` |
| Source split | A3.4 holdout only |
| Source evaluation ID | `a3-4-92838b4c8910027e4ffc00ad` |
| Source report SHA-256 | `f1d805d0977c6d1931d907770d5fbd797dfb706e4e80aa16b4764e5cb8a5c5c9` |
| YAMNet artifact SHA-256 | `2aee541e6039364299c90cfe5a715d239097aafb38aa4ce50d805a5445993b82` |
| Evaluation run ID | `evaluation-run-33c6a7a6fd06d11f4722ef03` |
| Evaluation run SHA-256 | `a2fd00c159c50c7a320b79c979eaa9b6e7f20eb4f02b651fcfd8f59a123b7d33` |

The checked-in builder includes all 16 held-out target examples: four ESC-50
`glass_breaking`, four ESC-50 `siren`, four UrbanSound8K `gun_shot`, and four
UrbanSound8K `siren` clips. It then selects 16 negative examples without reading
model scores: eight distinct negative categories from each dataset, with categories
and clips ranked by the fixed seed and SHA-256. The resulting manifest is balanced
between target and background labels and between datasets. Rebuilding the manifest
must reproduce the pinned ID, case ordering, and file hash before a run is accepted.

## Results

All 32 cases completed. No case was hidden by an operational failure.

| Measure | Result | Explicit denominator |
| --- | ---: | ---: |
| True positives | 13 | 16 target clips |
| False negatives | 3 | 16 target clips |
| True negatives | 14 | 16 background clips |
| False positives | 2 | 16 background clips |
| Accuracy | 84.375% | 27 / 32 completed clips |
| Balanced accuracy | 84.375% | mean of recall and specificity |
| Precision | 86.667% | 13 / 15 positive predictions |
| Recall | 81.25% | 13 / 16 target clips |
| Specificity | 87.5% | 14 / 16 background clips |
| F1 | 83.871% | precision/recall harmonic mean |
| False-positive rate | 12.5% | 2 / 16 background clips |
| False-negative rate | 18.75% | 3 / 16 target clips |
| Completion rate | 100% | 32 / 32 manifest cases |

The observed policy outcomes were 17 `no_action`, 9 `log`, 6 `review`, and 0
`alert`. The six reviews include all four detected UrbanSound8K gunshot clips and
both detected ESC-50 glass-breaking clips. Seven detected siren clips produced
`log`. No alert was expected from this acoustic-only collection: the consensus
policy requires independent branch support before alert escalation, and the run
permanently records `notification_delivery=not_sent` and
`alert_delivery_authorized=false`.

Target-class recall by source category was:

| Source category | Detected | Missed | Observed positive outcome |
| --- | ---: | ---: | --- |
| ESC-50 `glass_breaking` | 2 / 4 | 2 | 2 `review` |
| ESC-50 `siren` | 4 / 4 | 0 | 4 `log` |
| UrbanSound8K `gun_shot` | 4 / 4 | 0 | 4 `review` |
| UrbanSound8K `siren` | 3 / 4 | 1 | 3 `log` |

## False-positive review

These cases are false positives against the available single class label. A public
dataset label may not exhaustively describe every sound audible in a clip, so the
evidence does not prove that a target-like sound is absent.

| Case | Source label | Observed result | Evidence |
| --- | --- | --- | --- |
| `a9-1-esc50-cow-c1a2d0a439a2` | `cow` | `log`, risk 10 (`low`) | One siren candidate, peak score `0.7208229303` |
| `a9-1-esc50-wind-e223b86ed2f5` | `wind` | `log`, risk 20 (`low`) | Two siren candidates, peak scores `0.9581786990` and `0.6662927866` |

Both errors are low-risk siren detections. The next model-quality investigation
should inspect the time-localized cow and wind segments, compare their YAMNet class
score neighborhoods, and test a siren-specific calibration or persistence rule on
a larger validation set. Thresholds must not be changed using only these two clips.

## False-negative review

| Case | Source label | Observed result | Evidence |
| --- | --- | --- | --- |
| `a9-1-esc50-glass_breaking-85c77e1a44d4` | `glass_breaking` | `no_action`, risk 0 | No acoustic event candidate crossed the fixed threshold |
| `a9-1-esc50-glass_breaking-8a8207337731` | `glass_breaking` | `no_action`, risk 0 | No acoustic event candidate crossed the fixed threshold |
| `a9-1-urbansound8k-siren-ccb1f3f0f174` | `siren` | `no_action`, risk 0 | No acoustic event candidate crossed the fixed threshold |

The two glass misses are the clearest observed weakness: recall was only 50% for
that four-clip category. The next evaluation should expand held-out glass examples
and inspect event duration, background masking, and YAMNet score distribution. The
single UrbanSound8K siren miss should be included in that analysis, but the present
sample is too small to choose a new global or class-specific threshold safely.

## Repeatability check

Running the same pinned manifest twice initially exposed a receipt-identity defect:
the Phase 8 service correctly created new immutable report, audit, and optional
alert receipt IDs on each execution, but B9.1 included those local receipt IDs in
the evaluation run's semantic identity. The evaluation code now retains those IDs
for traceability while excluding them from metric identity and idempotent reuse.
Predictions, risk results, safe failure codes, manifest receipt, metrics, and
notification state remain identity-bearing.

After the correction, an exact rerun returned the same
`evaluation-run-33c6a7a6fd06d11f4722ef03` identity and reported `reused=true`.
Regression tests also vary local receipt IDs while holding the measured result
constant, proving that a repeated measurement reuses the existing run rather than
creating a false conflict.

## Limits on interpretation

- This is a regression-style evaluation over clips already used as the A3.4
  holdout split, not untouched external validation.
- The balanced 16-positive/16-negative design does not represent real deployment
  prevalence, so precision and accuracy will not transfer directly to deployment.
- Only 16 target clips were measured: 4 glass-breaking, 4 gunshot, and 8 siren.
- The 16 negatives cover 16 deterministically selected categories, not all 232
  background examples in the A3.4 holdout set.
- Public environmental recordings do not reproduce the project's eventual rooms,
  microphones, distances, reverberation, or noise conditions.
- Single-label datasets can omit co-occurring sounds, limiting certainty about
  apparent false positives.
- The run is acoustic-only. It provides no measurement of speech transcription,
  language understanding, or cross-branch consensus accuracy.
- Exact policy outcome accuracy is unavailable without separately reviewed action
  labels.
- These results do not approve a production threshold, alert policy, or accuracy
  claim.

## Reproduce locally

The raw datasets and YAMNet model are intentionally ignored by Git and must already
be installed in their configured local directories. Their source licenses continue
to apply, including noncommercial restrictions where applicable.

```powershell
python scripts/build_a9_1_evaluation_manifest.py
$env:TF_CPP_MIN_LOG_LEVEL='2'
& '.venv\yamnet\Scripts\python.exe' scripts\run_evaluation_manifest.py `
  configs\a9-1-evaluation-manifest.json `
  --expected-sha256 24da0b0fe6a352f00c1b79e9830db8ce79e8cd3f942776a6ea91be8c5dffefe9 `
  --output outputs\a9_1_evaluation\evaluation-run.json
```

The builder refuses a source report whose evaluation identity or SHA-256 differs
from the pinned A3.4 evidence. The runner refuses a manifest whose content hash
differs from the value above.
