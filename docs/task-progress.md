# Task progress

## Working arrangement

Continue both Person A and Person B tasks in this project while Person B is
unavailable. Keep task IDs so dependencies remain clear. Explain any manual steps
and add a short beginner-friendly explanation at the end of each completed task.

## A1.1 — Complete

Defined preprocessing settings and the prepared-audio manifest contract. Added
validation, JSON schemas, example JSON files, and the audio preparation guide.
The existing Phase 0 clip schema and preprocessing interface remain compatible.

Verification: all 69 tests pass (16 existing tests and 53 A1.1 cases), and the
project verification script passes. Updated tracker:
`outputs/a1_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 10 completed tasks out of 72 and assigns the 25 remaining Person B
tasks to Person A while preserving the original task IDs and completed ownership.

No manual setup, download, or audio processing is required for this task. Review
and commit this checkpoint before continuing if following the existing commit workflow.

Following task: B1.1 — Implement local audio loading and input validation.

## B1.1 — Complete

Implemented local WAV/FLAC loading with source metadata, original-file hashing,
permission checks, containment checks, input limits, and decoded-memory limits.
The loader returns float32 samples without changing sample rate, channels, or volume.
Added regression tests, a generated-tone smoke test, and `docs/audio-loading.md`.

Verification: all 118 tests pass with no skips. The project verification script and
the generated-tone smoke test pass. Tests include Windows junction containment,
truncated WAV/FLAC, invalid samples, and consent expiry before and after decoding.

Tracker at completion: `outputs/b1_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 11 completed tasks out of 72. No manual installation or download is
needed. Review and commit this checkpoint when ready.

Following task: A1.2 — Implement mono conversion, resampling, and loudness normalization.

The loader turns the audio file into a table of numbers, checks that the file and
permission are valid, and attaches a record of where those numbers came from.

## A1.2 — Complete

Implemented channel averaging, filtered resampling with the contracted frame count,
and RMS normalization with gain/peak limits. Added an in-memory helper connecting
these operations to loader output, normalization measurements, regression tests,
a synthetic smoke test, and `docs/audio-transforms.md`.

Verification: all 161 tests pass with no skips. The project verification script and
transform smoke test pass. Tests check pitch, anti-alias filtering, timing, stereo
balance, silence, gain/peak limits, memory limits, and original-data preservation.

Tracker at completion: `outputs/a1_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 12 completed tasks out of 72. No manual installation or download is
needed. Review and commit this checkpoint when ready.

Following task: B1.2 — Implement optional configurable noise reduction.

Mono conversion combines channels, resampling puts the recording on a common
timing grid, and normalization adjusts volume with limits. The result stays in
memory for the later noise-reduction, windowing, and persistence tasks.

## B1.2 — Complete

Implemented configurable stationary spectral gating with noise estimation from the
clip, smoothed attenuation, exact-length reconstruction, short/silent bypass,
diagnostics, and a spectral workspace budget. Integrated it after resampling and
before normalization. Disabled mode preserves A1.2 samples exactly. Updated schemas,
examples, regression tests, and the noise-reduction guide and synthetic smoke test.

Verification: all 199 tests pass with no skips. Project checks and the synthetic
smoke test pass. At strength 0.8, the fixed synthetic case improved SNR from 14.06
to 17.51 dB. Real-dataset effectiveness remains to be evaluated; default stays off.

Tracker at completion: `outputs/b1_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 13 completed tasks out of 72. No manual installation or download is
needed. Review and commit this checkpoint when ready.

Following task: B1.3 — Implement deterministic overlapping window segmentation.

Noise reduction estimates recurring background frequencies, turns down weaker
components near that estimate, and reconstructs the original-length recording.
It is optional because useful sustained sounds can also resemble background noise.

## B1.3 — Complete

Implemented deterministic overlapping windows with integer sample positions, both
padding/drop tail policies, stable IDs and planned paths, independent sample arrays,
one-window-at-a-time allocation, and permission checks while iterating. Records are
compatible with the existing prepared-audio manifest. Added tests, a synthetic
loader-to-segmentation smoke test, and `docs/audio-segmentation.md`.

Verification: all 229 tests pass with no skips. Project checks and the segmentation
smoke test pass. The 1.6-second sample produces three 1-second windows, one padded
5-second window, and one padded 10-second window. Prepared files are not written yet.

Tracker at completion: `outputs/b1_3_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 14 completed tasks out of 72. No manual installation or download is
needed. Review and commit this checkpoint when ready.

Following task: A1.3 — Implement prepared-output persistence and manifests.

Windows are short excerpts of the prepared recording. Overlap helps a sound near
one excerpt's edge appear centrally in another. Padding fills an incomplete final
excerpt with zeros while the metadata retains its actual recording boundaries.

## A1.3 — Complete

Implemented verified PCM16 WAV persistence for full clips and windows, validated
JSON manifests, deterministic bundle directories, staged publication, identical
output reuse, conflict protection, and per-save disk/window limits. Added failure
and permission regression tests, a synthetic save/reload smoke test, and
`docs/audio-persistence.md`. The result exposes the existing file-backed
`PreprocessedAudio` interface for the upcoming pipeline service.

Verification: all 266 tests pass with no skips. Project checks and the persistence
smoke test pass. The synthetic run verifies one full WAV, five window WAVs, the
manifest, repeat-save reuse, and original-file preservation. Failure checks cover
partial writes, manifest errors, corrupt readback, failed publication, conflicting
output, consent expiry, and Windows junctions.

Tracker at completion: `outputs/a1_3_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 15 completed tasks out of 72. No manual installation, recording, or
download is needed. Review and commit this checkpoint when ready.

Following task: A1.4 — Integrate preparation steps into one pipeline service.

Persistence turns the prepared numbers in memory into playable audio files. The
manifest catalogs those files and their origin/settings. Writing and checking a
temporary folder first keeps a failed save from exposing an incomplete result.

## A1.4 — Complete

Added `AudioPreparationService` to connect local loading, signal preparation,
windowing, and verified persistence in one call. The service returns the saved
bundle and implements the existing `AudioPreprocessor` interface. Per-call audio
settings apply consistently to every stage, without changing service defaults.
Paths, dataset attribution, annotations, output limits, and permission checks are
carried through to the result. Updated the project status response and added
`docs/audio-pipeline.md` with usage and a beginner-friendly explanation.

Verification: all 281 tests pass with no skips, including 15 new service checks.
Project compilation passes. Generated-recording tests verify noise reduction on
and off, actual encoded output, metadata, stereo/rate overrides, repeat saves,
protocol compatibility, failure propagation, limits, and permission expiry between
stages. No real recording or dataset download was needed.

Tracker at completion: `outputs/a1_4_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 16 completed tasks out of 72. No manual setup is needed. Review, commit,
and push this checkpoint when ready. B1.4 and A1.5 remain separate test tasks.

Following task: B1.4 — Add unit tests for loader, noise reduction, and segmentation.

The service connects the separate preparation tools in the right order. Give it
a local recording, permission, and a dataset name; it returns the saved audio and
manifest so later models can use them without repeating the preparation wiring.

## B1.4 — Complete

Reviewed all three component suites and added 22 checks for untested behavior:
WAV byte order and odd-chunk padding, decoder frame mismatches and resource errors,
late non-finite samples, permission-copy independence, noise profiling boundaries,
alternate FFT/rate combinations, amplitude scaling, read-only strided inputs,
silent-channel diagnostics, and allocation failures. Two parameterized tests also
check coverage and tail minimality across 80 seeded window grids.

Verification: all 137 tests in the three component suites pass. The full project
suite passes all 303 tests with no skips, and compilation and diff checks pass.
No audio-processing implementation changes were needed. Added
`docs/preparation-component-tests.md` and updated the project status next step.

Tracker at completion: `outputs/b1_4_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 17 completed tasks out of 72. No manual installation, recording, or
dataset download is needed. Review, commit, and push this checkpoint when ready.

Following task: A1.5 — Add preparation integration tests and sample-clip smoke test.

Unit tests check individual tools using known inputs and expected results. Some
also simulate broken input or failed allocations. Running them after future edits
helps catch regressions before those changes affect real recordings.

## A1.5 — Complete

Added 13 preparation integration checks, including an eight-case WAV/FLAC recipe
matrix, disk readback of every window against the full encoded clip, configuration
reload and repeat saves, silence, late-write recovery, metadata conflicts, and
annotation permission scope. Added `scripts/smoke_test_preparation.py`, runnable
without an editable install or `PYTHONPATH`, and a subprocess test from a different
working directory. Documented the workflow in `docs/preparation-integration-tests.md`.

Verification: all 316 project tests pass with no skips. The standard verification
script compiles the package, runs the suite, and passes the new smoke test. Python
exit codes are now explicitly checked so failures stop verification. The generated
1.6-second sample yields 16 kHz mono PCM16, five windows, and -20.0 dBFS RMS in the
report; repeat save reuses output, the source is unchanged, and temporary files are
removed. Diff checks pass. No audio-processing algorithm changes were needed.

Current tracker: `outputs/a1_5_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 18 completed tasks out of 72. No manual installation, recording, or
download is needed. Optionally run `python scripts/smoke_test_preparation.py` to see
the result yourself. Review, commit, and push this checkpoint when ready.

Next: A2.1 — Define Log-Mel feature contract and metadata schema.

Integration tests check that the separate preparation tools work together and
produce usable files. The smoke test is a quick complete run with one generated
sample, making it easy to check the main workflow in the current environment.

## A2.1 — Complete

Defined `LogMelSettings`, `LogMelSource`, and `LogMelFeatureMetadata`, two JSON
schemas, configuration/metadata examples, and `docs/log-mel-features.md`.
The recipe specifies mono input, FFT/window/hop settings, Slaney filters,
fixed-reference power dB, short-input padding, frame-tail handling, numeric
float32 arrays, source-window references and hashes, and software versions.
Settings load independently through `load_settings(log_mel_config_path=...)`.
Existing preparation options remain compatible. Metadata checks enforce shape,
rate, padding, payload limits, and linkage to a preparation manifest.

Verification: all 386 tests pass, including 70 new contract cases. The standard
verification script passes compilation, the full suite, and the unchanged
preparation smoke test. Updated the project-status expectation to B2.1.
No feature arrays have been generated: generation, storage, numerical tests,
and pipeline integration remain their separate Phase 2 tasks. The documented
default recipe is not yet certified for a pretrained acoustic model.

Current tracker: `outputs/a2_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 19 completed tasks out of 72. Use this project tracker; the standalone
Desktop workbook and the Documents/ChatGPT scaffold are older checkpoints.
No manual setup, recording, or downloads are needed. Review and commit this
checkpoint when ready.

Next: B2.1 — Implement Log-Mel spectrogram generation.

This task defines the format of the frequency-over-time numbers and the record
that ties them to their audio window. The next task will calculate those numbers
from the prepared audio.

## B2.1 — Complete

Implemented `generate_log_mel` and its in-memory `LogMelFeatures` result in
`src/audio_sentinel/log_mel.py`. The generator follows A2.1's periodic Hann,
Slaney Mel power, fixed-reference dB, frame-grid, and short-padding rules.
It returns owned contiguous float32 arrays and the installed librosa version,
preserves source samples, rejects incompatible input and empty filters, and
checks output and estimated workspace budgets before FFT allocations.
Float64 intermediates avoid overflow when squaring finite float32 extremes.

Added `docs/log-mel-generation.md`, 35 generator checks, and a standalone smoke
script that reloads all five saved Phase 1 windows of a generated tone. Tests
compare against independently calculated FFT frames, Mel triangles, and dB
values, and check frequency/amplitude behavior, silence, boundary lengths,
repeatability, input preservation, and errors. The smoke script verifies shapes
`(64, 97)` three times, `(64, 497)`, and `(64, 997)` without modifying source files.

Verification: all 421 tests pass with no skips. The standard verification script
passes compilation, the full suite (including the new smoke subprocess), and
the preparation smoke test. Diff checks pass. No setup or downloads were needed.

Current tracker: `outputs/b2_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 20 completed tasks out of 72. Feature storage remains A2.2, expanded
shape/range/determinism testing remains B2.2, and production integration remains
A2.3. Changes are ready for review and are uncommitted.

Next: A2.2 — Implement feature storage and source-window linkage.

Each audio window can now become a table of frequency energy over time. The
next task will save that table with a verifiable reference to its source window.

## A2.2 — Complete

Implemented `save_window_log_mel`, `load_log_mel`, and configurable persistence
limits in `src/audio_sentinel/feature_persistence.py`. One listed prepared window
is decoded, transformed, and saved as a verified `features.npy` and `metadata.json`
bundle under `data/processed/log-mel/`. Metadata includes exact feature/window/
manifest byte hashes and the original source identity inherited from preparation.
Reloads verify source linkage, current manifest consent, numeric format, shape,
finite values, and hashes. The raw source is not reopened or modified.

Publication uses a checked private staging directory. Repeated saves regenerate
and verify existing output while preserving its original bytes, timestamps, and
creation time. Corrupt/conflicting output is rejected. Source changes, invalid
paths, Windows junctions, oversized files/arrays, malformed NPY headers, denied
or expired consent, failed writes, and publication conflicts are covered.

Verification: all 461 project tests pass with no skips, including 40 new storage
tests. Compilation and the existing preparation smoke test also pass. Storage
tests use real saved Phase 1 windows and compare reloaded features with B2.1
generation. No manual setup, recording, or download was required. Added
`docs/feature-storage.md` with the API, limits, failure behavior, and explanation.

Current tracker: `outputs/a2_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 21 completed tasks out of 72. Changes remain uncommitted for review.
The full preparation-to-feature production service remains A2.3.

Next: B2.2 — Add feature shape, range, and determinism tests.

The frequency-over-time table can now be saved with a record identifying its
audio window and settings. Reloading checks that the table and its source still
match that record before returning the numbers.

## B2.2 — Complete

Reviewed the existing B2.1 numerical examples and A2.2 storage checks, then added
24 feature-property tests in `tests/test_feature_properties.py`. They cover 108
boundary/seeded length cases across four recipes at 8, 16, 44.1, and 48 kHz,
exact peak-relative clipping, fixed-reference shifts, configurable silence and
power floors, positive feature values, polarity/layout invariance, and output
independence after earlier arrays are modified. Two clean Python interpreters
with different extraction call orders reproduce the parent's exact NPY bytes
from the same saved input and recipe in the current environment.

Verification: all 485 project tests pass with no skips, including all 24 new
tests. Compilation and the preparation smoke test pass. No audio-processing
implementation changes were needed. Updated status and documentation, including
`docs/feature-property-tests.md`. No manual setup or downloads are required.
Exact cross-library/platform bitwise equality is not claimed.

Current tracker: `outputs/b2_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 22 completed tasks out of 72. Changes remain uncommitted for review.

Next: A2.3 — Integrate feature extraction with preparation pipeline.

These tests check that feature tables have the right dimensions, obey their
scaling rules, and reproduce the same numbers when the same audio is processed
again. The next task will connect the verified components into one service.

## A2.3 — Complete

Added `AudioFeatureService.prepare` and its file-backed `PreparedAudioFeatures`
result in `src/audio_sentinel/feature_pipeline.py`. One call prepares an authorized
recording, processes each saved window in manifest order, checks source consistency,
verifies the final feature inventory, and returns paths/metadata with a JSON-ready
summary. Per-call recipes leave defaults unchanged. Per-window limits and an
aggregate feature-output budget are checked; an explicit drop-tail empty inventory
returns zero features. Final verification releases one feature array at a time.

Failures propagate without returning partial success. Valid completed preparation
and feature bundles remain for checked reuse on retry. This is per-bundle
publication, not a transaction over all clip outputs. Existing preparation-only
behavior is unchanged; the integrated service requires mono conversion and
matching audio/feature rates before preparation begins.

Verification: all 515 project tests pass with no skips, including 30 integration
tests covering WAV/FLAC, 8/16 kHz, denoising, annotations, overrides, JSON defaults,
repeat runs, empty windows, budgets, retries, changed sources, earlier-output
corruption, and consent expiry. The new standalone
`scripts/smoke_test_feature_pipeline.py` passes through the suite: five expected
feature shapes, complete repeat reuse, unchanged generated source, and temporary
cleanup. Compilation, the preparation smoke test, and diff checks pass.

Added `docs/feature-pipeline.md`. No manual setup, recording, or downloads are
needed for this task. Changes remain uncommitted for review.

Current tracker: `outputs/a2_3_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 23 completed tasks out of 72. All five Phase 2 Log-Mel tasks are complete.

Next: A3.1 — Choose pretrained acoustic model and define label mapping.

The separate tools are now connected: one recording becomes prepared audio and
a verified list of saved frequency-over-time tables. The next phase adds a
pretrained model that interprets audio and maps its predictions to our event labels.

## A3.1 — Complete

Selected YAMNet TensorFlow Hub revision 1 after checking its official frontend,
vocabulary and runtime requirements against PANNs Cnn14_16k and the local project.
Added `src/audio_sentinel/acoustic_model.py` with a frozen model specification,
mapping version 1.0, six candidate acoustic-label mappings and explicit deferrals
for all seven remaining v1 labels. Exact index/MID/name triples are checked against
the bundled, checksummed official 521-class CSV. Its license/provenance are included;
package-data configuration and Git byte preservation keep the reference usable
after installation and checkout. No model weights are bundled.

Phase 1 and Phase 2 remain complete. YAMNet consumes prepared 16 kHz mono waveforms
through its own frontend, because our generic power-dB Log-Mel tables do not match
its trained magnitude/natural-log recipe. Documented this distinction, model
selection rationale, runtime observations, timing support, semantic limitations
and later score/aggregation responsibilities in `docs/acoustic-model-selection.md`.
Runtime loading, inference and measured accuracy remain later tasks.

Verification: all 537 project tests pass with no skips, including 22 new mapping
checks. Compilation, preparation smoke test and diff checks pass. Built a wheel
and successfully validated its vocabulary directly from the package, including
license/provenance resources. The vocabulary and reference license were downloaded
from the pinned official source; no model dependencies or weights were installed.

Current tracker: `outputs/a3_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 24 completed tasks out of 72. Changes remain uncommitted for review.

Next: B3.1 — Implement isolated acoustic-model loader with version metadata.

We have chosen the recognizer and defined its translation dictionary. The next
task loads it; subsequent tasks run predictions and test their accuracy.

## B3.1 — Complete

Added `load_yamnet` in `src/audio_sentinel/acoustic_loader.py` for local-only,
lazy TensorFlow loading. Before importing TensorFlow, it resolves the SavedModel
inside `models/`, rejects links/junctions and unexpected files, enforces a byte
budget, hashes the complete four-file payload, and compares it with the pinned
YAMNet v1 digest. It also reuses the verified 521-class vocabulary and mapping.

The loaded model must expose the exact `serving_default` waveform contract and
the expected 521-score, 1024-embedding, and 64-band model-owned Log-Mel outputs.
The result returns the callable plus JSON-ready metadata covering model identity,
relative local path, artifact/vocabulary hashes, label mapping, TensorFlow runtime,
SavedModel exporter versions, and tensor shapes/dtypes. B3.1 does not run inference.

Added a pinned `yamnet` optional dependency group, one-command isolated setup in
`scripts/setup_yamnet.ps1`, official Kaggle model download/verification, a real
loader smoke test, unit coverage, and `docs/acoustic-model-loading.md`. TensorFlow
2.21.0 and the model live under ignored `.venv/` and `models/` directories. The
normal project import and test path still does not require TensorFlow or network.

Verification: the real Google YAMNet v1 SavedModel loaded successfully under
Python 3.13 with TensorFlow 2.21.0. Its payload digest is
`2aee541e6039364299c90cfe5a715d239097aafb38aa4ce50d805a5445993b82`,
its vocabulary digest matches A3.1, its exporter reports TensorFlow 2.3.0, and its
signature matches the expected input and three outputs. The standard verification
script passes compilation, all 557 project tests, and the preparation smoke test.

Current tracker: `outputs/b3_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 25 completed tasks out of 72. Model inference remains A3.2.

Next: A3.2 — Implement inference orchestration over prepared waveforms.

The recognizer is now installed in a separate environment and its exact files and
input/output plugs are checked every time it loads. The next task will feed real
prepared audio through that verified callable and retain raw model evidence.

## A3.2 — Complete

Added `infer_prepared_audio` in `src/audio_sentinel/acoustic_inference.py` to run
the verified local YAMNet v1 callable over every listed preparation window in
manifest order. It reads 16 kHz mono PCM16 waveforms directly, verifies consent,
file containment and WAV properties, removes only verified preparation padding,
and preserves raw `(patches, 521)` sigmoid-score matrices as owned read-only
float32 arrays. The model's embeddings and internal spectrogram are checked for
their exact shapes/dtypes/finite values and then discarded.

Results include model version metadata, manifest/raw/window hashes, exact window
records, unpadded input lengths, output shapes, and padding-clipped absolute patch
support. Manifest/window snapshots are rechecked after inference. Configurable
limits bound manifest/window reads, decoded samples, total model output, and the
window count. Model/runtime failures are converted to stable safe errors. This task
does not choose thresholds, merge overlaps, emit events, or calculate risk.

Added focused fake-model regression coverage, a real generated-tone YAMNet smoke
test, and `docs/acoustic-inference.md`. The ordinary suite stays TensorFlow-free;
the real smoke uses the existing ignored `.venv/yamnet` runtime and local model.

Verification: all 590 project tests pass, including 33 A3.2 checks. The standard
verification script, preparation smoke test, real YAMNet inference smoke test,
compilation, and diff checks pass.

Current tracker: `outputs/a3_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 26 completed tasks out of 72. No new setup is required if B3.1 setup
was already run. On a clean machine, run `.\scripts\setup_yamnet.ps1` once.

Next: B3.2 — Implement event aggregation across overlapping windows.

In plain language: each saved audio excerpt now goes through the sound recognizer,
and we keep its original class scores with proof of the exact audio and model used.
The next task combines repeated evidence from overlapping excerpts without counting
the same sound twice.

## B3.2 — Complete

Added `aggregate_acoustic_events` in
`src/audio_sentinel/acoustic_aggregation.py`. For each patch and mapped project
label, it selects the maximum associated YAMNet class score rather than summing
related classes. Scores meeting an explicit caller-supplied threshold become patch
contributions. Overlapping, adjacent, or explicitly gap-bridged contributions for
the same label are unioned into one candidate interval whose score is the maximum
observed score, so repeated preparation windows cannot inflate confidence.

Each candidate retains its label, sample-exact union, peak uncalibrated score,
source window IDs/hashes, patch indexes, clipped supports, and winning YAMNet
classes. Labels remain separate. Output is chronologically deterministic and uses
`AcousticEventCandidate`, not `EventAnnotation`; it emits no risk level or incident
claim. Thresholds must explicitly cover all six mapped labels because A3.4 still
owns calibration and accuracy evaluation.

The aggregator revalidates pinned model metadata, patch/tensor shapes, score dtype,
finite range, window uniqueness, real input lengths, and contribution limits. It
returns no partial result on failure and never modifies raw inference arrays.
Added 37 focused tests, `docs/acoustic-aggregation.md`, and a real YAMNet structural
smoke test. The latter merges five patches from three overlapping generated-tone
windows into one interval per mapped label using a deliberately zero test threshold;
it makes no accuracy claim.

Verification: all 627 project tests pass. The standard verification script,
preparation smoke test, real YAMNet aggregation smoke test, compilation, and diff
checks pass.

Current tracker: `outputs/b3_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 27 completed tasks out of 72. No new setup is required if B3.1 setup
was already run. On a clean machine, run `.\scripts\setup_yamnet.ps1` once.

Next: A3.3 — Produce timestamped acoustic-evidence JSON.

In plain language: overlapping excerpts can show the recognizer the same sound
several times. This step groups that repeated evidence into one candidate interval,
keeps only the strongest score, and remembers every patch that supported it.

## A3.3 — Complete

Added `build_acoustic_evidence`, `save_acoustic_evidence`, and
`load_acoustic_evidence` in `src/audio_sentinel/acoustic_evidence.py`. The new v1
JSON contract records a UTC creation time, deterministic semantic evidence ID,
16 kHz sample-exact and second-based timestamps, explicit aggregation settings,
complete input window/patch inventory, every retained patch contribution, and the
full preparation/model/mapping/runtime provenance chain.

The builder recomputes B3.2 aggregation from the paired A3.2 inference snapshot
before accepting it. Contract validation checks the pinned model, event and patch
grids, timestamp conversions, ordering, thresholds, winning mapped classes, hashes,
counts, and evidence identity. The document is explicitly
`acoustic_event_candidates`; scores remain uncalibrated model evidence and the
schema has no incident, probability, or risk decision.

Persistence writes one JSON file under
`data/processed/acoustic-evidence/<evidence-id>/` using bounded staging, fsync,
readback, source rechecks, and atomic publication. Repeated equivalent saves reuse
the original document and creation time. Existing conflicts are never overwritten.
Reloading rechecks the bundle path/inventory, resource limits, current consent,
manifest/raw-audio identity, the complete prepared-window inventory, and every
window hash. It stores no raw audio, score matrices, embeddings, or spectrograms.

Added the checked-in JSON Schema, 22 focused tests, task documentation, and a real
isolated-YAMNet persistence smoke test. The smoke saved and reloaded evidence for
three overlapping windows, five YAMNet patches, and all six mapped labels using a
deliberately zero structural-test threshold. It makes no accuracy claim.

Verification: all 649 project tests pass. Compilation, diff checks, the real YAMNet
evidence smoke test, schema round-trip checks, repeat-save behavior, tamper/source
rejection, limits, and tracker formula/visual checks pass.

Current tracker: `outputs/a3_3_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 28 completed tasks out of 72. No new setup is required if B3.1 setup was
already run. On a clean machine, run `.\scripts\setup_yamnet.ps1` once.

Next: B3.3 — Add acoustic loader and aggregation unit tests.

In plain language: the recognizer's candidate sounds now have a durable receipt.
It says exactly when the model produced each piece of evidence and preserves proof
of the source audio, prepared windows, model version, mapping, and rules used.
