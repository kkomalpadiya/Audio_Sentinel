# Labeled acoustic evaluation

A3.4 evaluates the pinned YAMNet baseline against deterministic samples from the
approved local ESC-50 and UrbanSound8K copies. It measures clip-level recognition
quality. It does not validate incident decisions, risk levels, event timing, or
speech meaning.

## Scope and label coverage

The benchmark has direct positive categories for three project labels:

| Project label | Dataset categories |
| --- | --- |
| `siren` | ESC-50 `siren`; UrbanSound8K `siren` |
| `glass_break` | ESC-50 `glass_breaking` |
| `gunshot` | UrbanSound8K `gun_shot` |

`speech_present`, `smoke_alarm`, and `explosion` are not evaluated because the
local approved benchmark does not contain a direct positive category that matches
their A3.1 definitions. In particular, ESC-50 `fireworks` remains a negative
confounder rather than being relabeled as an explosion. Absence of false positives
on an unsupported label would not establish detection quality, so the runner does
not choose thresholds for those labels.

Both datasets are CC BY-NC 3.0 and require attribution. The runner requires an
explicit license acknowledgement, works locally, and records hashes of the
metadata, license files, and every selected audio file. It does not upload audio.

## Sampling and evaluation method

The runner selects clips independently within each dataset, split, and source
category by sorting a fixed SHA-256 rank. This produces the same selection from
unchanged dataset bytes without depending on filesystem order.

| Dataset | Calibration folds | Holdout folds | Calibration | Holdout |
| --- | --- | --- | ---: | ---: |
| ESC-50 | 1–4 | 5 | 8 per category, 400 total | 4 per category, 200 total |
| UrbanSound8K | 1–8 | 9–10 | 8 per category, 80 total | 4 per category, 40 total |

The 480 calibration clips and 240 holdout clips are disjoint. Each source clip is
decoded, converted to mono, and resampled to 16 kHz without loudness normalization
or noise reduction. Its project-label score is the maximum over all YAMNet patches
and all YAMNet classes mapped to that label. Every other selected category acts as
a negative for that label.

For each evaluated label, the runner selects a threshold that maximizes clip-level
F1 on calibration clips only. Ties prefer precision, then recall, then the higher
threshold. The untouched holdout clips are evaluated once with that threshold.
Average precision is threshold-independent and groups tied scores so input order
cannot change the value. Undefined rates are stored as JSON `null`, never as zero.

## Baseline results

The checked-in report was produced with the pinned YAMNet v1 artifact and 720
clips. These are small-sample research results, not production calibration.

| Label | Selected threshold | Holdout positives | TP | FP | FN | Precision | Recall | F1 | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `siren` | 0.690535 | 8 | 6 | 0 | 2 | 1.000 | 0.750 | 0.857 | 0.868 |
| `glass_break` | 0.084563 | 4 | 3 | 2 | 1 | 0.600 | 0.750 | 0.667 | 0.793 |
| `gunshot` | 0.917472 | 4 | 0 | 0 | 4 | undefined | 0.000 | 0.000 | 0.285 |

Siren generalized best in this bounded sample. Glass-break recall remained useful,
but its low selected threshold produced two holdout false positives. Gunshot did
not generalize at the calibration-selected threshold: it detected none of the four
holdout positives. The gunshot value must not be promoted into runtime defaults.
All three labels need broader positive coverage, deployment-like negatives,
confidence intervals, and operating-point review before production use.

The complete result is
`outputs/a3_4_evaluation/acoustic_detection_evaluation.json`. It contains the model
and mapping identity, split rules, dataset/license hashes, per-label calibration and
holdout metrics, per-dataset holdout metrics, and per-clip hashes, labels, audio
properties, truth values, and scores. This makes the selected samples and every
reported confusion count independently inspectable without committing raw audio.

## Run the benchmark

The B3.1 isolated runtime and model must already exist. The approved local dataset
copies must be present at `data/raw/esc50` and `data/raw/UrbanSound8K`.

```powershell
.\.venv\yamnet\Scripts\python.exe scripts\evaluate_acoustic_detection.py `
  --acknowledge-noncommercial-dataset-licenses `
  --output outputs/a3_4_evaluation/acoustic_detection_evaluation.json
```

The writer does not overwrite an existing report. Supply a different output path
for another run. The ordinary unit tests do not import TensorFlow or require the
datasets:

```powershell
python -m pytest tests/test_acoustic_evaluation.py -q
.\scripts\verify_project.ps1
```

## Interpretation limits

- The balanced category sample does not estimate real-world event prevalence.
- Environmental clips do not reproduce deployment microphones, rooms, noise, or
  distance.
- Single-label source metadata may omit background sounds, making some negatives
  noisy.
- Clip-level maximum scores do not measure timestamp or event-boundary accuracy.
- Only four holdout positives support glass-break and gunshot metrics, and eight
  support siren metrics.
- Threshold search on 480 calibration clips can overfit, as the gunshot result
  demonstrates.
- The benchmark does not authorize processing of private recordings. The normal
  consent and device-authorization controls still apply to project input audio.
