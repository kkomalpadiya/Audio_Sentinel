# Phase 3 completion report: Acoustic Detection

## 1. What this phase accomplished

Phase 3 built the part of Audio Sentinel that listens for sound patterns such as
sirens, breaking glass, gunfire, smoke alarms, explosions, and the presence of
speech. It starts with the clean audio windows produced in Phase 1, runs them
through a pretrained sound-recognition model, combines repeated detections from
overlapping windows, and saves an auditable evidence file.

The phase also tested the recognizer on labeled public recordings. That evaluation
showed that siren recognition is promising in the current small benchmark,
glass-break recognition needs improvement, and the tested gunshot threshold is
not reliable.

Phase 3 is complete as an engineering and research baseline. It is not a finished
emergency-detection product. The system can produce traceable sound evidence, but
it does not yet decide that an emergency happened or assign a risk level.

At the end of this phase:

- all seven Acoustic Detection tasks in the tracker are complete;
- 30 of the project's 72 tracked tasks are complete;
- the full project suite contains 717 passing tests;
- the exact local YAMNet model and its input/output contract are verified;
- raw model scores can be generated from prepared audio;
- repeated evidence from overlapping audio windows is merged safely;
- timestamped acoustic evidence can be saved and reloaded with provenance checks;
- labeled evaluation has been completed on 720 selected recordings; and
- the next task is A4.1, which begins the Speech and Transcription phase.

## 2. The complete flow in common terms

```text
Authorized recording
        |
        v
Prepared 16 kHz mono audio windows from Phase 1
        |
        v
Verified local YAMNet sound recognizer
        |
        v
Raw scores for 521 kinds of sound, patch by patch
        |
        v
Translation into Audio Sentinel labels
        |
        v
Repeated/overlapping evidence grouped into candidate time intervals
        |
        v
Timestamped evidence JSON with source, model, and rule history
        |
        v
Later phases may combine this evidence with speech and other information
```

The important word is **evidence**. A strong siren score means the model heard
something that resembles its learned siren examples. It does not prove that an
emergency exists. The code deliberately preserves this distinction.

## 3. Why a pretrained model was used

Training a sound recognizer from scratch would require a very large labeled audio
collection, significant computing power, and a long validation process. Instead,
Phase 3 selected YAMNet as the first baseline.

YAMNet is a Google sound-recognition model trained to identify 521 general sound
categories. Examples include speech, alarms, sirens, shattering sounds, gunfire,
music, vehicles, animals, and many ordinary environmental noises.

YAMNet was chosen because:

- it accepts 16 kHz mono waveforms, matching the prepared audio from Phase 1;
- it produces scores over short, repeated sections of audio, which supports
  timestamps;
- it has a documented 521-class vocabulary;
- it can run locally after a one-time model download; and
- its outputs are suitable for building a transparent baseline before considering
  a larger custom model.

PANNs Cnn14_16k was also considered. It remains a possible comparison model if
future testing shows that YAMNet is not good enough. YAMNet was selected first
because its short-time patch outputs fit the evidence pipeline more directly.

## 4. Why Phase 2 Log-Mel files are not fed into YAMNet

Phase 2 created general Log-Mel feature tables. They are useful for inspection and
for future models trained with that exact feature recipe. YAMNet, however, was
trained with its own internal audio-to-feature conversion.

Two feature tables can look similar while using different mathematical details,
just as two photographs can have the same dimensions but different color systems.
Feeding the Phase 2 tables into YAMNet would therefore produce invalid results.

For this baseline, the system gives YAMNet the verified 16 kHz mono waveform.
YAMNet creates its own internal features. The Phase 2 feature files remain intact
for later use and are not duplicated or discarded.

## 5. Task-by-task work completed

### A3.1 — Model selection and label mapping

This task chose the model and created a controlled translation dictionary between
YAMNet's vocabulary and Audio Sentinel's labels.

YAMNet understands 521 model classes, while Audio Sentinel uses a much smaller set
of project labels. Mapping version 1.0 connects the following labels:

| Audio Sentinel label | YAMNet evidence used | Common meaning |
| --- | --- | --- |
| `speech_present` | Speech, child speech, conversation, narration, synthesized speech, whispering | Speech-like sound exists; it says nothing about meaning or speaker identity. |
| `siren` | Police, ambulance, fire-engine, general, and civil-defense sirens | A siren-like sound is present. |
| `smoke_alarm` | Smoke detector and fire alarm | An alarm resembling these model classes is present. |
| `glass_break` | Shatter | A shattering sound is present; the material is not proven to be glass. |
| `gunshot` | Gunshot/gunfire and machine gun | A gunfire-like sound is present; recordings and lookalikes remain possible. |
| `explosion` | Explosion | An explosion-like sound is present. |

Seven other project labels were deliberately deferred: `ambient`, `no_speech`,
`non_threatening_speech`, `crowd_panic`, `distress_speech`,
`threatening_speech`, and `weapon_reference`.

They were deferred because sound alone cannot safely establish their meaning. For
example, crying does not automatically prove distress, a crowd does not
automatically prove panic, and hearing speech does not reveal whether the words
are threatening. Those conclusions need the later speech and language phases.

The complete official 521-class vocabulary is bundled with the package and
protected by a SHA-256 hash. In common terms, the hash is a digital fingerprint.
If even one byte changes, validation fails instead of silently using a different
dictionary.

The mapping uses the highest relevant model score for a project label. It never
adds related scores together. Adding them could make several weak guesses look
like one strong detection.

### B3.1 — Verified and isolated model loader

This task built the code that opens the local YAMNet model safely.

TensorFlow is a large optional dependency, so it lives in its own environment at
`.venv/yamnet`. The normal project and ordinary unit tests do not need to load it.
The model weights live under `models/yamnet/1` and are ignored by Git.

Before accepting the model, the loader checks:

- the model is inside the approved `models` directory;
- no link or junction redirects the loader somewhere unexpected;
- all required model files exist;
- no unexpected model files are present;
- the total model size is within the configured limit;
- the complete model fingerprint matches the pinned YAMNet v1 fingerprint;
- TensorFlow is the expected version;
- the 521-class vocabulary is the expected vocabulary; and
- the model's named input and three outputs have the expected shapes and data
  types.

The loader performs no network access. Downloading is a separate one-time setup
step. This means a production run cannot quietly replace the model from the
internet.

The returned metadata records which model, runtime, vocabulary, mapping, and
input/output contract were loaded. Absolute local paths are not exposed in the
portable metadata.

### A3.2 — Inference over prepared audio

This task built the part that actually sends prepared audio through YAMNet.

Each saved Phase 1 window is opened in manifest order. The code checks that it is
a 16 kHz, mono, PCM16 WAV matching the preparation record. It also checks that
permission for acoustic processing is still active.

Some prepared windows contain zero padding at the end so every saved window has a
fixed length. That padding is verified and removed before inference. YAMNet may
then add its own internal padding to form its final short analysis patch. Reported
timestamps are clipped to the real recorded audio, so padded silence is never
reported as evidence.

YAMNet returns three large outputs:

1. scores for 521 sound classes;
2. internal embeddings; and
3. YAMNet's internal Log-Mel representation.

The pipeline validates all three outputs, but it retains only the raw class-score
matrix needed by later steps. This avoids storing large unused arrays. Retained
scores are copied into read-only memory so another part of the program cannot
quietly alter the evidence.

Every result keeps the model identity, preparation-manifest fingerprint, raw-audio
fingerprint, prepared-window fingerprints, window positions, real input length,
and model patch positions.

Resource limits bound the number of windows, input sizes, decoded samples, and
model-output memory. A failure returns a stable project error instead of exposing
uncontrolled TensorFlow details.

### B3.2 — Combining overlapping evidence correctly

Phase 1 intentionally creates overlapping windows. That helps avoid missing a
sound that falls across a window boundary, but the same sound can appear in more
than one window.

Without careful handling, the system could count the same siren two or three times
and make the evidence look stronger than it is. B3.2 prevents that.

For each label, the aggregator:

- finds model patches whose scores meet an explicitly supplied threshold;
- joins patches that overlap, touch, or fall within an explicitly allowed gap;
- keeps the highest score instead of adding or averaging repeated scores;
- keeps different project labels separate;
- clips every interval to real recorded audio; and
- remembers every source window and patch that contributed.

The result is called an `AcousticEventCandidate`. It is intentionally not called
an incident. It has no risk level and makes no emergency decision.

The aggregator has no hidden production thresholds. A caller must supply all six
mapped-label thresholds. This prevented an untested number from quietly becoming a
system default.

### A3.3 — Timestamped acoustic-evidence JSON

This task turned the in-memory candidates into a durable evidence receipt.

The JSON document records:

- when it was created;
- a deterministic evidence ID;
- the raw recording and preparation-manifest fingerprints;
- the complete prepared-window and model-patch inventory;
- the exact model, vocabulary, runtime, and mapping versions;
- the thresholds and overlap-merging rules used;
- event and contribution positions in exact audio samples and readable seconds;
- the strongest score for each candidate; and
- every patch and YAMNet class that supported it.

Before creating the document, the builder recomputes aggregation from the paired
raw inference result. This prevents someone from accidentally joining scores from
one run with event intervals from another.

The evidence ID is based on the meaningful content, not the creation time. Running
the same computation again therefore produces the same identity.

Saving uses a private staging directory, verifies the written document, rechecks
the source, and publishes the final directory atomically. In common terms, other
code should see either the complete valid result or no result, not a half-written
file. Existing evidence is never overwritten. An identical result can be reused;
a conflicting or damaged result is rejected.

Reloading repeats the important checks. It verifies current permission, all source
and window fingerprints, the evidence identity, the directory contents, and
resource limits. The JSON stores no raw audio, embeddings, spectrograms, or full
521-class score matrices.

### B3.3 — Boundary and failure-case tests

This task expanded testing around the model loader and event aggregator. The two
focused files contain 96 tests, including 47 cases added in B3.3.

Loader tests cover ordinary loading plus missing files, extra files, changed model
bytes, linked paths, oversized artifacts, incomplete downloads, runtime changes,
vocabulary changes, model-signature changes, TensorFlow failures, and immutable
returned metadata.

Aggregation tests cover every mapped model class, exact threshold boundaries,
overlapping and adjacent patches, configured gaps, long chains of overlapping
evidence, tail clipping, reversed input order, duplicate windows, malformed score
arrays, invalid model metadata, contribution limits, deterministic ordering, and
proof that input arrays are not modified.

These tests check software behavior. They do not claim that the recognizer is
accurate. Accuracy was measured separately in A3.4.

### A3.4 — Evaluation against labeled recordings

This task created a reproducible benchmark using the approved local ESC-50 and
UrbanSound8K datasets.

The benchmark has direct positive examples for only three project labels:

- `siren` from both datasets;
- `glass_break` from ESC-50 `glass_breaking`; and
- `gunshot` from UrbanSound8K `gun_shot`.

The local data does not contain an approved direct positive category for
`speech_present`, `smoke_alarm`, or `explosion`. Fireworks were kept as a difficult
negative example rather than dishonestly relabeling them as explosions.

The evaluator selected a fixed, category-balanced sample using a SHA-256 ranking.
Selection therefore does not depend on the order in which Windows returns files.
It used official source folds to create two separate groups:

| Use | ESC-50 | UrbanSound8K | Total |
| --- | ---: | ---: | ---: |
| Threshold calibration | 400 | 80 | 480 |
| Final holdout check | 200 | 40 | 240 |
| Total | 600 | 120 | 720 |

The calibration group was used to choose one threshold per supported label. The
holdout group was kept separate and used only after threshold selection. This is
similar to studying with one set of questions and taking a test with different
questions.

For every clip, the label score was the highest relevant score seen anywhere in
the clip. Threshold selection maximized F1 on the calibration group. F1 balances
two concerns:

- **precision:** when the system reports the sound, how often is it right?
- **recall:** of the real examples, how many did it find?

When two thresholds had the same F1, the evaluator preferred precision, then
recall, then the higher threshold. The report stores every chosen file's hash,
source label, audio properties, truth values, and model scores so the final counts
can be independently checked.

## 6. Measured results

The final holdout results were:

| Label | Threshold | Real positives | Correctly found | False alarms | Missed | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Siren | 0.690535 | 8 | 6 | 0 | 2 | 1.000 | 0.750 | 0.857 |
| Glass break | 0.084563 | 4 | 3 | 2 | 1 | 0.600 | 0.750 | 0.667 |
| Gunshot | 0.917472 | 4 | 0 | 0 | 4 | Undefined | 0.000 | 0.000 |

### What the siren result means

The tested threshold found six of eight siren recordings and produced no false
alarms among the other 232 holdout clips. This is the strongest of the three
results, but eight positive examples are too few to approve a production setting.

### What the glass-break result means

The model found three of four glass-breaking recordings, but it also incorrectly
flagged two other clips. The very low selected threshold shows that the model's
`Shatter` output is relatively weak on these examples. It may still be useful as
one piece of evidence, but it should not independently trigger a high-risk action.

### What the gunshot result means

The selected threshold missed all four holdout gunshot recordings. Precision is
listed as undefined because the model made no positive predictions, so there were
no reported detections to judge as correct or incorrect.

This is a failed operating point. It is not hidden, converted into a passing
number, or installed as a runtime default. The result shows that the calibration
sample was too small or did not generalize well enough. Gunshot detection needs
more representative positive examples, difficult negative examples, threshold
review, and possibly a different or specialized model.

## 7. Safeguards built throughout the phase

### Local processing

The model runs locally. Loading and inference do not upload recordings or fetch a
new model from the internet. The one-time setup command performs the explicit
public model download.

### Permission checks

Normal project recordings still require active acoustic-processing consent and an
authorized device record. Permission is rechecked at processing and reload
boundaries. Public benchmark clips use their dataset licenses rather than
pretending to be private recordings with consent records.

### Exact identities

The pipeline records fingerprints for the model, vocabulary, raw audio,
preparation manifest, and prepared windows. Changed bytes cause a failure instead
of silently changing the meaning of old evidence.

### Safe paths and bounded resources

Managed files must remain inside approved project directories. Critical model and
evidence paths reject traversal, links, and junctions. Limits cover file sizes,
decoded audio, model outputs, window counts, evidence contributions, and benchmark
sample counts.

### No confidence inflation

Related model classes use their maximum score rather than a sum. Repeated patches
and overlapping windows also keep their maximum score rather than being added
together.

### No automatic incident claim

Outputs remain sound candidates and evidence. They contain no risk level and do
not claim that an emergency occurred. Later orchestration must combine multiple
forms of evidence under explicit decision rules.

## 8. Testing and verification completed

Phase 3 added 202 tests across its seven tasks:

| Task | Tests added | Main purpose |
| --- | ---: | --- |
| A3.1 | 22 | Model specification and mapping integrity |
| B3.1 | 20 | Verified local model loading |
| A3.2 | 33 | Inference and waveform/source checks |
| B3.2 | 37 | Overlap-safe aggregation |
| A3.3 | 22 | Evidence contract and persistence |
| B3.3 | 47 | Loader and aggregation boundary expansion |
| A3.4 | 21 | Sampling, metrics, reporting, and evaluation limits |
| **Total** | **202** | **Phase 3 additions** |

After A3.4, all 717 project tests passed. Compilation and the Phase 1 preparation
smoke test also passed.

Real-model smoke checks were completed for:

- loading the exact pinned YAMNet model;
- running inference on generated audio;
- combining overlapping predictions;
- saving and reloading acoustic-evidence JSON; and
- evaluating 720 real labeled dataset clips.

The real evaluation report's confusion counts were independently recomputed from
the stored per-clip scores and truth labels.

## 9. Important limitations

The following work is still required before acoustic results can be treated as
production-ready:

- Only three of the six mapped acoustic labels have direct positive evaluation
  examples in the current benchmark.
- Siren has only eight holdout positives. Glass break and gunshot have four each.
- The benchmark contains public environmental recordings, not recordings from the
  final device, room, microphone, distance, or noise conditions.
- The source datasets mostly provide one label per clip even when background
  sounds may also exist.
- The evaluation measures whether a sound appears somewhere in a clip. It does not
  measure exact start and end times.
- YAMNet scores are not incident probabilities.
- The evaluated thresholds cover only siren, glass break, and gunshot. The runtime
  aggregator still requires explicit thresholds for all six mapped labels, so no
  complete production threshold configuration was created.
- The gunshot operating point failed and must not be used as a production default.
- Smoke-alarm, explosion, and speech-presence accuracy remain unmeasured.
- Acoustic evidence alone cannot determine panic, distress, threatening language,
  weapon references, or overall risk.

## 10. Main files produced or changed

### Production code

- `src/audio_sentinel/acoustic_model.py` — pinned model specification and label
  translation dictionary.
- `src/audio_sentinel/acoustic_loader.py` — safe local model loader and metadata
  verification.
- `src/audio_sentinel/acoustic_inference.py` — prepared-waveform inference and raw
  score preservation.
- `src/audio_sentinel/acoustic_aggregation.py` — thresholding and overlap-safe
  candidate grouping.
- `src/audio_sentinel/acoustic_evidence.py` — evidence construction, persistence,
  and reload verification.
- `src/audio_sentinel/acoustic_evaluation.py` — deterministic labeled evaluation,
  threshold selection, and metrics.

### Model resources and schemas

- `src/audio_sentinel/resources/yamnet_class_map.csv` — official 521-class
  vocabulary.
- `src/audio_sentinel/resources/yamnet-LICENSE.txt` and
  `yamnet-NOTICE.txt` — model licensing and provenance.
- `docs/schemas/v1/acoustic-evidence.schema.json` — machine-readable evidence
  document contract.

### Scripts

- `scripts/setup_yamnet.ps1` — creates the isolated environment, installs the
  pinned runtime, downloads the model, and verifies it.
- `scripts/download_yamnet_model.py` — obtains and checks the official model.
- `scripts/smoke_test_acoustic_loader.py` — real loader check.
- `scripts/smoke_test_acoustic_inference.py` — real inference check.
- `scripts/smoke_test_acoustic_aggregation.py` — real aggregation check.
- `scripts/smoke_test_acoustic_evidence.py` — real persistence check.
- `scripts/evaluate_acoustic_detection.py` — real labeled benchmark runner.

### Results and documentation

- `outputs/a3_4_evaluation/acoustic_detection_evaluation.json` — complete
  per-clip evaluation record and metrics.
- `outputs/a3_4_tracker_update/Audio_Sentinel_Master_Task_List.xlsx` — tracker
  showing 30 of 72 tasks complete and A4.1 next.
- `docs/acoustic-model-selection.md`
- `docs/acoustic-model-loading.md`
- `docs/acoustic-inference.md`
- `docs/acoustic-aggregation.md`
- `docs/acoustic-evidence.md`
- `docs/acoustic-unit-tests.md`
- `docs/acoustic-evaluation.md`

## 11. Manual setup and rerun instructions

No manual step is needed to keep the completed Phase 3 work on the current
machine. The environment, model, and datasets used for the final checks are already
present locally.

On a fresh checkout or another machine, run the one-time model setup from the
project root:

```powershell
.\scripts\setup_yamnet.ps1
```

This creates the ignored `.venv/yamnet` environment and downloads the ignored
model weights into `models/yamnet/1`.

To repeat the labeled evaluation, approved local copies of ESC-50 and
UrbanSound8K must be placed at:

```text
data/raw/esc50
data/raw/UrbanSound8K
```

Both datasets are licensed CC BY-NC 3.0 and require attribution. The evaluation
runner therefore requires an explicit noncommercial-license acknowledgement:

```powershell
.\.venv\yamnet\Scripts\python.exe scripts\evaluate_acoustic_detection.py `
  --acknowledge-noncommercial-dataset-licenses `
  --output outputs/a3_4_evaluation/acoustic_detection_evaluation.json
```

The runner will not overwrite an existing report. A repeated run must use another
output path or intentionally remove the old generated report first.

## 12. What comes next

The next tracker task is **A4.1 — Define speech-evidence schema and reliability
rules**.

Phase 4 will build the separate speech path. Its first job is to define what speech
evidence means, what can and cannot be concluded from it, how confidence and
timestamps are represented, and how the system behaves when speech is unclear.

That separation is intentional. Phase 3 can say that a sound resembles speech,
a siren, shattering, or gunfire. It cannot determine what a person said or whether
their words were threatening. Phase 4 begins building that missing layer while
preserving the same consent, provenance, and evidence-first approach.
