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

## B3.3 — Complete

Completed the boundary-focused unit-test matrix for the isolated YAMNet loader and
overlap-safe event aggregator. The focused pair now contains 96 tests, including
47 new cases added in this task. Production code did not require changes: the
existing B3.1/B3.2 implementation passed the stricter behavior and failure checks.

Loader additions cover valid/invalid download markers, model byte limits, linked
model directories and payloads, mid-hash file changes, full signature inventory,
input/output name and dtype drift, TensorFlow load/inspection failures, class-map
redirection, string asset paths, missing exporter metadata, and immutable returned
contracts. The tests continue to prove that the normal package import does not
require TensorFlow and that loading does not run inference.

Aggregation additions exercise every mapped class across all six project labels,
deterministic contribution ordering with reversed inputs, source-window
deduplication, tail-patch clipping, exact adjacency, transitive gap chains, strict
integer settings, exact contribution-limit behavior, complete model metadata drift,
malformed patch support, and immutable results. Existing checks continue to cover
maximum-not-sum scoring, threshold boundaries, overlap without score inflation,
finite score/tensor validation, empty inputs, and input preservation.

Added `docs/acoustic-unit-tests.md` with the test matrix, commands, and scope.
Software correctness tests do not measure recognition quality or approve thresholds;
A3.4 remains responsible for labeled evaluation.

Verification: all 696 project tests pass. The standard project verification script,
real pinned YAMNet loader smoke test, real overlap aggregation smoke test,
compilation, diff checks, and tracker formula/visual checks pass.

Current tracker: `outputs/b3_3_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 29 completed tasks out of 72. No new setup or manual action is required.

Next: A3.4 — Evaluate acoustic detection against labeled samples.

In plain language: the sound-model boundary now has tests for both ordinary use and
the failure cases most likely to corrupt provenance or double-count overlapping
audio. The next task checks how well the recognizer performs on labeled recordings.

## A3.4 — Complete

Added a reproducible, local-only labeled benchmark in
`src/audio_sentinel/acoustic_evaluation.py` and
`scripts/evaluate_acoustic_detection.py`. It creates disjoint calibration and
holdout samples from official dataset folds, balances selection by source category
with a fixed SHA-256 rank, verifies dataset/license/audio provenance, safely decodes
and resamples each clip, and reduces the pinned YAMNet output to the A3.1 project
label mapping. Thresholds maximize clip-level calibration F1; holdout clips never
participate in threshold selection.

The completed run evaluated 720 clips: 480 calibration and 240 holdout. Direct
positive coverage exists only for siren, glass break, and gunshot. Holdout results
were: siren F1 0.857 (6 TP, 0 FP, 2 FN), glass-break F1 0.667 (3 TP, 2 FP, 1 FN),
and gunshot F1 0.000 (0 TP, 0 FP, 4 FN). The failed gunshot operating point remains
visible and is explicitly rejected as a runtime default. Speech presence, smoke
alarm, and explosion remain unevaluated because the local datasets do not provide
direct positive categories under the approved mapping; fireworks remain a
confounder rather than an explosion proxy.

The checked-in JSON report preserves the model/mapping identity, dataset metadata
and license hashes, sampling contract, selected thresholds, calibration/holdout and
per-dataset metrics, limitations, and per-clip hashes, labels, audio properties,
truth, and scores. No raw dataset audio is committed or uploaded. The report labels
all results as a research baseline, not production calibration.

Added 21 TensorFlow-free unit tests for deterministic/disjoint sampling, category
mapping, bounded resource use, confusion metrics, tied-score average precision,
threshold selection, model/tensor identity, stable report identity, end-to-end
fake-model reporting, JSON safety, and non-overwriting persistence. Added
`docs/acoustic-evaluation.md` with methods, results, interpretation limits,
licensing, and rerun instructions.

Verification: the focused 21-test evaluation suite passes. The real pinned YAMNet
v1 run completed over all 720 selected clips and the report metrics reconcile to
their confusion counts. Full project verification and tracker checks are recorded
with this task.

Current tracker: `outputs/a3_4_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 30 completed tasks out of 72. No new setup is required when B3.1 and the
approved ESC-50/UrbanSound8K datasets are already present. A fresh machine must run
the prior model setup and place the licensed datasets locally before rerunning.

Next: A4.1 — Define speech-evidence schema and reliability rules.

In plain language: the recognizer now has an honest report card on labeled sounds.
Siren looks promising in this small sample, glass breaking needs refinement, and
gunshot is not reliable at the tested threshold. The next task defines the evidence
contract and reliability rules for the separate speech-processing branch.

## A4.1 — Complete

Defined the versioned speech-evidence contract and deterministic reliability rules
in `src/audio_sentinel/speech_contracts.py`. The contract links every result to an
authorized prepared-audio manifest, records model/runtime provenance, preserves the
complete input-window inventory, and expresses final VAD-positive segments with
sample-exact timestamps, optional transcript candidates, and verified handling
decisions. It deliberately contains no language classification, incident outcome,
severity, or risk score.

Reliability policy v1.0 records a VAD gate of 0.60, a transcript review boundary of
0.50, and an acceptance boundary of 0.80. Below-review candidates are rejected,
middle-band candidates require human review, and only accepted transcript text may
flow automatically to later language analysis. Missing transcription remains
`not_transcribed`; it is never interpreted as safe. Confidence values are typed as
model, derived, or calibrated signals so an ordinary normalized score is not
misrepresented as a probability.

The document validator recomputes each assessment and checks consent scope, model
provenance, counts, unique and deterministic inventories, safe relative paths,
source bounds, non-overlapping final segments, exact sample-to-second conversions,
VAD threshold compliance, and source-window links. Unknown fields, NaN/Infinity,
unbounded transcript text, and unapproved speech scope are rejected.

Added the checked-in JSON Schema, a complete two-segment example,
`docs/speech-evidence.md`, and 43 focused tests covering threshold boundaries,
tampering, VAD-only evidence, missing provenance, counts, timestamps, ordering,
overlap, source links, consent, transcript validation, immutability, schema export,
and privacy exclusions.

Current tracker: `outputs/a4_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 31 completed tasks out of 72. No model download or new manual setup is
required for this contract task.

Next: B4.1 — Implement pretrained voice-activity-detection wrapper.

In plain language: the speech branch now has a strict receipt format and a clear
traffic rule for uncertain words. Only sufficiently reliable text can continue
automatically; uncertain text stays blocked for review, and missing text is never
treated as proof that the audio was harmless.

## B4.1 — Complete

Implemented the verified Silero VAD v6 loader and raw frame-probability wrapper in
`src/audio_sentinel/vad.py`. The model comes from the pinned
`faster-whisper==1.2.1` distribution and runs through pinned
`onnxruntime==1.23.2` on the CPU execution provider. The loader verifies the exact
file inventory, installation marker, byte size, SHA-256 digest, runtime version,
provider, and complete ONNX input/output signature before returning the model. It
rejects linked paths and detects model changes during loading.

The wrapper scores independent 16 kHz mono float32 waveforms on Silero's 512-sample
(32 ms) frame grid. It supplies the required 64-sample preceding context, carries
the two recurrent states only across bounded batches within one waveform, and
resets both states for every new call. Incomplete final frames are right-padded,
but returned sample/time bounds stop at real audio and expose the padding count.
Outputs are finite raw float32 probabilities in `[0, 1]`; no speech segment,
transcript, language category, or risk decision is produced in this task.

Added strict input/output and memory limits, JSON-ready immutable result metadata,
the `speech` dependency extra, repeat-safe setup and model installation scripts,
the real pinned-model smoke test, MIT notices, and `docs/vad-wrapper.md`. Added 72
focused tests covering artifacts, markers, links, runtime and tensor drift, exact
frame/context construction, batching and state carry, state reset, tail padding,
resource limits, invalid model outputs, safe errors, and normal import isolation.

The real smoke test loaded the verified 1,245,151-byte model, scored 34 frames twice,
and matched the pinned Faster Whisper reference output exactly.

Current tracker: `outputs/b4_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 32 completed tasks out of 72.

Manual setup on a fresh machine: run `.\scripts\setup_speech.ps1` once. This machine
has already completed that setup and the model remains in ignored local directories.

Next: A4.2 — Implement speech-segment extraction and timestamp handling.

In plain language: the project can now ask a verified pretrained model how
speech-like each 32-millisecond piece of audio is. The answer is still raw evidence;
the next task will join qualifying pieces into reliable speech intervals with exact
timestamps.

## A4.2 — Complete

Implemented verified prepared-window speech extraction in
`src/audio_sentinel/speech_segments.py`. The service validates the pinned Silero
model before file access, reads a bounded preparation manifest, requires active
`acoustic_and_speech` consent plus 16 kHz mono audio, selects one complete prepared
window duration, verifies and hashes every PCM-16 window, strips only recorded zero
padding, and reruns source verification before returning evidence.

Window-relative 512-sample VAD frames are converted to absolute prepared-clip
samples. Frames meeting the recorded A4.1 VAD threshold are sorted and unioned
across overlapping windows. Adjacent support merges by default; larger gaps merge
only when explicitly configured. Segment scores use the maximum contribution, not
a sum, and timestamps are derived exactly from final sample bounds. Tail padding
cannot extend a segment beyond real audio.

The result contains immutable window hashes, frame contributions, extraction
settings, selected duration, resource counts, deterministic segment IDs, and a
transcript-free `SpeechEvidenceDocument`. Each public segment carries complete
source-window coverage, `transcript: null`, and the policy-derived
`not_transcribed` assessment. The semantic evidence ID is stable across creation
times for identical source, model, policy, and segment content.

Added 37 focused tests for absolute offset conversion, overlap deduplication,
threshold boundaries, exact gap behavior, minimum duration, tail clipping,
deterministic identity, evidence provenance, consent scope, model validation,
source mutation, malformed padding, duration selection, incomplete window
coverage, resource limits, safe errors, and immutability. The real-model smoke test
prepares a generated 1.6-second clip, scores 83 frames across three overlapping
windows twice, and confirms the exact `[0, 25600)` span under a structural
zero-threshold policy.

Manual setup on a fresh machine remains `.\scripts\setup_speech.ps1`; it now runs
both the B4.1 model check and the A4.2 segment-extraction check.

Current tracker: `outputs/a4_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 33 completed tasks out of 72.

Next: B4.2 — Implement offline transcription-model wrapper.

In plain language: possible speech pieces from overlapping windows now land on one
exact timeline and become clean, non-overlapping intervals. No words are invented;
the next task adds the offline transcription model that can propose text for those
intervals.

## B4.2 — Complete

Implemented a pinned, local-only Faster-Whisper `tiny.en` loader and single-segment
transcription wrapper in `src/audio_sentinel/transcription.py`. The model is fixed
to the official Systran English CTranslate2 conversion at repository revision
`0d3d19a32d3338f10357c0889762bd8d64bbdeba`. The loader verifies the exact four-file
inventory, per-file sizes and SHA-256 digests, installation marker, combined
inventory digest, runtime versions, CPU device, resolved `int8_float32` compute
type, and English-only capability. It rejects links, junctions, path traversal,
artifact drift, and model changes during loading; normal loading is offline.

The wrapper accepts one independent 16 kHz mono float32 segment. Decoding is fixed
to English transcription, one beam, temperature zero, no previous-text context,
no prompt or hotwords, no word timestamps, and no internal VAD. It validates and
copies the input, exhausts the lazy result generator, validates all returned text,
tokens, timestamps, probabilities, and model metadata, and enforces input, segment,
token, and output-byte limits. Stable safe errors replace runtime exceptions.

Nonempty output becomes an A4.1 `TranscriptCandidate`. Its normalized confidence is
the exponential of the token-count-weighted average log probability and is
explicitly typed `derived_score`, not a correctness probability. Empty or
whitespace-only output remains no candidate. This task does not apply reliability
thresholds, accept text downstream, classify language intent, or calculate risk;
A4.3 owns those decisions.

Added exact runtime pins, repeat-safe model installation, the real offline silence
smoke test, OpenAI Whisper's MIT notice, `docs/transcription-wrapper.md`, and 39
focused tests covering artifacts, runtime/capability drift, deterministic options,
confidence math, normalization, malformed output, limits, safe failures,
immutability, and normal import isolation.

The real smoke test loaded the verified 78 MB model and transcribed the same
one-second silence twice with the same empty result. The full project verification,
speech setup workflow, and tracker checks are recorded with this task.

Current tracker: `outputs/b4_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 34 completed tasks out of 72. On a fresh machine, run
`.\scripts\setup_speech.ps1` once; this machine already has the verified local model.

Next: A4.3 — Orchestrate transcription and low-confidence handling.

In plain language: the project can now turn one verified speech interval into an
offline English text candidate while keeping the model, settings, and uncertainty
honest. The next task connects those candidates to speech evidence and enforces the
review/accept/reject rules before any text can continue automatically.

## A4.3 — Complete

Implemented verified speech-transcription orchestration in
`src/audio_sentinel/speech_transcription.py`. The orchestrator validates the pinned
transcription model before file access and accepts only an intact, transcript-free
A4.2 result. It checks the semantic evidence identity and agreement between public
evidence and diagnostic model, window, frame-count, segment, timestamp, VAD-score,
and hash metadata.

The service reopens the bounded prepared manifest, revalidates current
`acoustic_and_speech` consent and all source metadata, and rereads each selected
PCM-16 window using the hashes captured by A4.2. Every final speech interval is
rebuilt on the absolute sample grid. Referenced windows must cover every sample, and
overlapping window samples must agree exactly. Segments are transcribed in separate
model calls without carrying audio or decoder context across boundaries. Manifest,
window, and consent state are checked again after inference; any change invalidates
the full result.

Every nonempty model hypothesis is retained as an A4.1 `TranscriptCandidate`, and
the recorded policy recomputes its handling outcome. Scores below 0.50 are rejected,
scores from 0.50 to below 0.80 require review, and scores at or above 0.80 are
accepted. Empty output remains `not_transcribed` and is never treated as safe.
A separate immutable `downstream_transcripts` inventory contains accepted text only;
rejected and review-required text cannot flow automatically to language analysis.

The completed `SpeechEvidenceDocument` preserves the original source, VAD, window,
segment, and policy provenance and adds the pinned transcription model, candidates,
assessments, and exact counts. Its semantic ID now includes transcription content
and remains stable across creation times. Configurable limits bound source bytes,
decoded memory, segment count, total transcription samples, and accepted-text bytes.
No waveform, absolute path, language-intent decision, incident, or risk score is
returned.

Added 37 focused tests covering all four reliability outcomes, exact threshold
boundaries, acceptance-only handoff, overlapping-window reconstruction, snapshot
tampering, source changes, consent expiry, empty segmentation, deterministic
identity, model failure, resource limits, JSON privacy, and immutability. Added
`docs/speech-transcription.md` and a real pinned-model orchestration smoke test.

The real test reconstructs one complete 1.6-second structural interval from three
overlapping windows and runs the pinned offline transcriber twice. Both runs produce
identical evidence; the generated tone yields no text and therefore remains visibly
`not_transcribed` with no downstream handoff.

Current tracker: `outputs/a4_3_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 35 completed tasks out of 72. On a fresh machine, run
`.\scripts\setup_speech.ps1` once; this machine already has both verified local models.

Next: B4.3 — Add VAD and transcription wrapper tests.

In plain language: detected speech intervals now receive offline text candidates,
and each candidate immediately goes through the recorded confidence rules. Only
accepted text gets a downstream pass; uncertain, rejected, or missing words remain
blocked and visible for audit.

## B4.3 — Complete

Completed the boundary-focused unit-test matrix for the verified Silero VAD and
Faster-Whisper transcription wrappers. The focused suite now contains 184 fast,
offline cases: 90 VAD cases and 94 transcription cases. All model runtimes and
inference outputs remain synthetic in this suite, so the tests require neither the
downloaded artifacts nor ONNX Runtime/Faster-Whisper imports.

Expanded VAD coverage now checks the artifact byte limit, changes while the runtime
loads, safe runtime-construction failures, uninspectable sessions, valid amplitude
and probability endpoints, exact input/output/batch boundaries, model-input
ownership, preparation memory failure, the callable session entry point, and every
pinned metadata field. Existing cases continue to cover path containment, links,
inventory and marker integrity, hashing, runtime and tensor-contract drift, context
construction, recurrent state, tail padding, malformed outputs, deterministic
repeated calls, safe errors, and immutable portable results.

Expanded transcription coverage now checks linked paths and payloads, all marker
failure modes, size-before-hash enforcement, changes during hashing or model load,
aggregate identity before runtime access, missing runtimes, safe model-load
failures, the complete fixed decoding option inventory, input ownership, exact
resource boundaries, UTF-8 byte accounting, whitespace resource accounting,
additional malformed segment fields, decreasing timestamps, lazy-iterator errors,
memory failures, the callable transcriber, and every pinned metadata field.
Existing cases continue to verify deterministic English CPU decoding, confidence
math, normalization, empty output, limits, safe failures, and JSON-ready immutable
results.

Added `docs/speech-wrapper-tests.md` with the complete matrix and commands. The real
Silero and Faster-Whisper smoke tests remain separate checks of the installed pinned
artifacts and do not claim model-quality evaluation.

Current tracker: `outputs/b4_3_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 36 completed tasks out of 72. On a fresh machine, run
`.\scripts\setup_speech.ps1` once; this machine already has both verified models.

Next: A4.4 — Add speech-branch integration tests.

In plain language: the two speech-model adapters now have explicit tests for normal
operation and for their dangerous edges. Wrong files, changed runtimes, malformed
model output, or exceeded limits must fail safely instead of producing evidence.

## A4.4 — Complete

Added `tests/test_speech_integration.py` with 16 offline integration cases for the
complete speech branch. These tests run the real loader, transforms, resampling,
mono conversion, preparation persistence, manifest and window verification, VAD
orchestration, segment merging, exact waveform reconstruction, transcription
orchestration, reliability policy, and final evidence construction. Only the two
pretrained runtime calls are replaced with contract-accurate fakes, keeping the
normal suite deterministic and independent of downloaded model artifacts.

The integration matrix covers a 48 kHz stereo overlapping-window path plus four
WAV/FLAC, 8–48 kHz, mono/stereo source combinations. It verifies all transcript
outcomes and acceptance-only handoff, no-speech behavior, final-window padding
removal, custom policy propagation, fresh-service determinism, operation from the
persisted bundle after raw-source removal, source and bundle immutability, VAD and
transcription failures, between-stage tampering, and portable JSON round trips.

The combined 274-case speech run also exposed an order-dependent defect in the VAD
import-isolation test. Reloading the production module inside pytest replaced its
module-level identities for later tests. The check now runs in a subprocess, still
proving that normal import does not load ONNX Runtime while leaving the shared test
process untouched.

Added `docs/speech-integration-tests.md` with the matrix, commands, and scope. The
existing pinned-model orchestration smoke test remains the real-runtime structural
check; neither the deterministic integration suite nor that smoke test claims
real-world accuracy evaluation.

Current tracker: `outputs/a4_4_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 37 completed tasks out of 72. On a fresh machine, run
`.\scripts\setup_speech.ps1` once for the optional real-model checks.

Next: A5.1 — Define language categories, reason codes, and evidence schema.

In plain language: Phase 4 now works as a tested chain. Authorized audio can move
from preparation through speech detection and transcription, while uncertain text,
changed files, broken models, and partial failures remain blocked and auditable.

## A5.1 — Complete

Defined the Phase 5 language-evidence boundary in
`src/audio_sentinel/language_contracts.py`. The contract has six closed categories:
no concerning rule match, distress, threat, weapon reference, ambiguous language,
and context-suppressed language. Eight typed reason codes distinguish keyword and
phrase support from explicit negation, hypothetical/conditional wording,
quoted/reported speech, insufficient context, conflicting signals, and no match.

Only A4.3 accepted downstream transcripts can be represented as automatic inputs.
The artifact links back to hashed Phase 4 evidence, records exact sample times and
versioned rule provenance, and uses transcript hashes plus character spans instead
of copying full text. Validators enforce compatible categories/reasons/rule kinds,
bounded spans, exact timestamps, unique deterministic inventories, and accurate
summary counts. The schema contains no incident, severity, risk score, consensus,
speaker identity, or alert output.

Added the checked-in JSON Schema, a complete example showing an active distress
phrase and a negated weapon reference, `docs/language-evidence.md`, and 64 focused
contract tests. Full verification passes all 1,098 project tests, compilation, and
the generated preparation smoke test. No model, rule artifact, dataset, download,
or manual setup is required for this contract task.

Current tracker: `outputs/a5_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 38 completed tasks out of 72 and B5.1 as the next task. Changes remain
uncommitted for review.

Next: B5.1 — Create versioned keyword, phrase, and negation rule data.

In plain language: this task defines a strict receipt for future text analysis. It
records which accepted transcript was checked, which rule matched and where, and
why the result received its category. Negated or unclear words stay visibly
suppressed or ambiguous, and no language finding is allowed to make the later risk
or alert decision by itself.

## B5.1 — Complete

Created the versioned English rule artifact at
`src/audio_sentinel/resources/language-rules-en-v1.json`. Its 59 deterministic
rules include 17 keywords, 23 phrases, and 19 explicit-negation forms. Distress,
threat, and weapon-reference categories each have keyword and phrase coverage;
negation rules assign no category and can only support later context suppression.

Added `src/audio_sentinel/language_rules.py` with immutable contracts, exact NFKC,
case-folding, apostrophe, and token rules, canonical inventory validation, a 1 MiB
artifact limit, and a local-only loader. The bundled bytes are pinned by SHA-256,
and the loader verifies the digest, rule-set identity, version, normalization,
unique IDs/patterns, reason-code compatibility, and category coverage before use.
The package configuration now includes JSON resources.

Added the checked-in rule-set JSON Schema, `docs/language-rules.md`, and 64 focused
tests for the complete inventory, representative patterns, malformed rules,
ordering, duplicates, missing coverage, metadata and normalization drift, artifact
tampering, resource limits, stable checkout bytes, schema export, immutability, and
operation without network or ML runtimes. Full verification passes all 1,162 project tests,
compilation, and the generated preparation smoke test.

This task does not match transcripts, apply negation scope, produce findings, or
assign risk. No model, dataset, download, or manual setup is required.

Current tracker: `outputs/b5_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 39 completed tasks out of 72 and A5.2 as the next task. Changes remain
uncommitted for review.

Next: A5.2 — Implement transcript-analysis engine.

In plain language: the project now has a checked dictionary of concerning words,
phrases, and negation forms. Its checksum makes unnoticed edits fail loudly. The
next task will apply this dictionary to accepted transcripts while retaining exact
positions and context instead of treating a word match as a risk decision.

## A5.2 — Complete

Implemented the deterministic transcript-analysis engine in
`src/audio_sentinel/language_analysis.py`. It revalidates the complete A4.3 speech
evidence, recomputes its semantic identifier, reconstructs the policy-accepted
transcript inventory, and requires the supplied downstream tuple to match exactly.
Review-required, rejected, missing, changed, reordered, or injected transcript text
cannot enter automatic language analysis.

The engine applies the exact B5.1 NFKC, case-folding, apostrophe, and English-token
normalization while retaining original character positions. It matches versioned
keywords and phrases, prefers a containing phrase over a shorter same-category
keyword, retains distinct occurrences, and applies the configured three-token
explicit-negation window without crossing hard sentence boundaries. Suppressed
findings retain both exact matches and both reason codes; `I cannot breathe` remains
an active configured distress phrase because its negation token is inside the
phrase rather than before it.

The result is an immutable A5.1 language-evidence document with hashes instead of
duplicated transcript text, exact source and rule provenance, deterministic finding
and evidence IDs, accurate counts, and no risk, severity, incident, or alert output.
Resource limits bound transcript count, text bytes, tokens, rule checks, and
findings. Stable safe errors return no partial evidence and do not echo private
transcript content.

Added `docs/language-analysis.md` and 44 focused engine tests covering normalization,
phrase specificity, exact original spans and hashes, explicit-negation scope,
multiple findings, accepted-only gating, tamper detection, deterministic identity,
privacy, immutability, invalid settings, safe errors, and resource limits. The
focused Phase 5 suite passes 172 tests. Full verification passes all 1,206 project
tests, compilation, and the generated preparation smoke test.

Current tracker: `outputs/a5_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 40 completed tasks out of 72 and B5.2 as the next task. Changes remain
uncommitted for review.

Next: B5.2 — Create labeled language fixtures including harmless negations.

In plain language: accepted text can now be checked against the pinned dictionary
without turning a word into an alarm. The output is a reproducible receipt showing
what matched, where it matched, and whether nearby explicit negation suppressed it.

## B5.2 — Complete

Created the versioned fixture artifact at
`src/audio_sentinel/resources/language-fixtures-en-v1.json`. It contains 74 short,
synthetic English transcript cases with 75 expected A5.1 findings: 43 active
indicator cases, 23 context-suppressed cases, six no-match cases, and two ambiguous
cases. Every one of the 40 B5.1 active rules appears in an active fixture, and every
one of the 19 negation rules appears in a harmless explicit-negation fixture.

The matrix also covers Unicode normalization, punctuation inside phrases, phrase
specificity, sentence-boundary behavior, multiple findings, token boundaries,
hypothetical or conditional language, quoted or reported language, and insufficient
context. The latter context cases define desired behavior for A5.3; B5.2 does not
change engine output or weaken labels to fit the current implementation.

Added `src/audio_sentinel/language_fixtures.py` with immutable fixture contracts, a
1 MiB limit, exact A5.1 and B5.1 provenance, canonical ordering and count checks,
complete active-rule and negation coverage validation, and a SHA-256-pinned local
loader. Added the checked-in JSON Schema, protected fixture bytes in
`.gitattributes`, `docs/language-fixtures.md`, and 73 focused tests covering exact
labels, complete coverage, malformed fixtures, metadata drift, rule mismatches,
artifact tampering, schema export, immutability, and operation without network,
model runtimes, or the analysis engine.

Full verification passes all 1,279 project tests, compilation, and the generated
preparation smoke test.

Current tracker: `outputs/b5_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 41 completed tasks out of 72 and A5.3 as the next task. Changes remain
uncommitted for review.

Next: A5.3 — Add language-analysis tests and false-positive safeguards.

In plain language: the project now has a checked answer key for language analysis.
It includes direct matches, ordinary harmless text, every negation form, and context
cases that the next task can use to measure and improve the engine consistently.

## A6.1 — Complete

Defined the Phase 6 risk-assessment contract in
`src/audio_sentinel/risk_contracts.py`. The contract summarizes hash-pinned Phase 3
acoustic evidence, Phase 4 speech evidence, and Phase 5 language evidence without
copying raw audio, transcript text, speaker identity, consensus decisions, or alert
details. Each branch now has an explicit status: present, missing, not permitted,
not applicable, or no accepted text.

The contract pins the risk score to a finite 0-100 range and validates the v1
severity bands: none at exactly 0, low above 0 through 24, medium from 25 through
49, high from 50 through 74, and critical from 75 through 100. It also defines the
missing-data policy: missing evidence is recorded and can require human review
instead of being interpreted as safety. Consent-limited branches are represented as
not permitted rather than as missing.

Added `docs/risk-assessment.md`, the checked-in
`docs/schemas/v1/risk-assessment.schema.json`, a complete example at
`docs/examples/risk-assessment.json`, and 36 focused tests covering score
boundaries, branch/evidence compatibility, missing-data review, privacy exclusions,
canonical reason ordering, exact acoustic timestamps, schema export, and immutable
public records.

Focused verification passes 36 tests in `tests/test_risk_contracts.py`. Full
project verification should be run before commit.

Local tracker note: this checkout still shows A5.3 as not started even though the
working instruction says Phase 5 has been completed and committed. Reconcile or
pull the A5.3 tracker/code state before updating the Excel tracker for Phase 6.

Next: B6.1 — Implement configurable risk-scoring rules engine.

In plain language: Phase 6 now has the receipt for risk scoring. It records what
evidence the scorer saw, what was missing, what score band the result belongs to,
and whether missing data needs a person to review it. The next task will calculate
the score.

## B6.1 — Complete

Implemented the configurable deterministic risk-scoring engine in
`src/audio_sentinel/risk_scoring.py`. The engine consumes a validated A6.1
`RiskInputSet`, calculates independently capped acoustic, speech, and language
contributions, caps the final score at 100, assigns the existing v1 severity band,
emits canonical reason codes, evaluates missing-data and uncertainty review rules,
and returns a validated `RiskAssessmentDocument`.

Added the versioned built-in scoring artifact at
`src/audio_sentinel/resources/risk-scoring-rules-v1.json`. The loader bounds the
artifact size, rejects unknown or malformed fields, validates complete canonical
label/category coverage, protects non-risk categories from nonzero weights, and
pins the exact built-in bytes with SHA-256. Risk assessments now include the rule
set ID, version, format version, and digest, and deterministic assessment IDs change
when either the inputs or scoring configuration changes.

The built-in rules reproduce the documented example score of 82: 50 acoustic
points for a high-confidence explosion, 7 speech points for speech plus one
review-required transcript, and 25 language points for distress. Missing branches
add no points but require review; consent-limited branches remain distinct from
missing data. Human review is also required at score 25 or above, for transcript
uncertainty, or for ambiguous language.

Added `docs/risk-scoring.md`, the checked-in
`docs/schemas/v1/risk-rule-set.schema.json`, scoring-rule provenance in the risk
assessment schema and example, and 25 focused scoring tests. The focused B6.1 and
A6.1 suite passes 61 tests. Full verification passes all 1,340 project tests,
compilation, and the generated preparation smoke test.

Local tracker note: the latest workbook in this checkout still records A5.3 and
A6.1 as not started, so it was not advanced for B6.1. Reconcile or pull the missing
tracker state before marking Phase 6 tasks complete in Excel.

Next: A6.2 — Integrate acoustic, speech, and language evidence into risk inputs.

In plain language: Phase 6 now has the calculator behind the risk receipt. Its
numbers come from a small versioned JSON rule file, no branch can contribute beyond
its cap, missing evidence stays visible, and every result records exactly which
rules produced it.

## A6.2 — Complete

Implemented `integrate_risk_inputs()` in
`src/audio_sentinel/risk_integration.py`. The adapter accepts a trusted A6.1 source
plus available acoustic, speech, and language evidence, revalidates each document,
checks that the branches belong to the same prepared source, and produces the
privacy-minimized `RiskInputSet` consumed by B6.1.

Acoustic events are converted to sample-exact risk signals with labels, peak
scores, and source indexes. Speech evidence is summarized into segment, accepted
transcript, review-required transcript, and maximum VAD counts. Language findings
become ordered signals containing only IDs, categories, reason codes, sample spans,
and match counts. Each present branch records the exact normalized evidence-document
hash without copying transcript text, matched text, raw audio, model tensors,
speaker data, or absolute paths.

Cross-branch verification now rejects mismatched clips, consent IDs, sample
metadata, prepared-manifest hashes, raw-audio hashes, speech evidence references,
or accepted-transcript provenance. Acoustic-only consent produces explicit
not-permitted speech and language branches. Missing expected evidence stays
missing, while a completed speech branch with zero accepted transcripts produces
the distinct `no_accepted_text` language state.

Added `docs/risk-integration.md` and 17 focused tests covering complete three-branch
integration through the scorer, exact artifact hashes, missing branches,
acoustic-only consent, no accepted text, empty language artifacts, cross-source and
transcript mismatches, invalid input types, and privacy exclusions.

The focused Phase 6 contract, scoring, and integration suite passes 78 tests. Full
verification passes all 1,357 project tests, compilation, and the generated
preparation smoke test.

Resumed Excel tracker updates and backfilled sequential A6.1 and B6.1 snapshots
before creating the A6.2 snapshot. The latest workbook records 44 of 72 tasks
complete and preserves all formulas, formatting, validation controls, and tabs.
A5.3 remains not started because no corresponding commit exists in this checkout.

Next: B6.2 — Create risk-scoring tests for normal and edge scenarios.

In plain language: Phase 6 can now take the three evidence reports, prove that they
belong together, remove private details, and hand a compact trustworthy summary to
the scoring calculator.

## A5.3 — Complete

Completed the language false-positive safeguards in
`src/audio_sentinel/language_analysis.py`. The analyzer now recognizes bounded
hypothetical or conditional speech framing, quoted or reported speech framing, and
two narrow insufficient-context constructions. Explicit negation remains the
highest-precedence context rule, and no context cue can cross a hard sentence or
clause boundary.

Converted the complete B5.2 answer key into executable regression coverage. All 74
fixtures now run through the real analyzer and compare category, canonical reason
codes, exact supporting rule IDs, and finding order. Additional cases verify that
the safeguards generalize within a clause, preserve direct questions with an
object, respect sentence boundaries, and keep explicit-negation precedence.

This closes the behavioral gap without changing any Phase 6 contract. A6.1 already
supports `ambiguous` and `context_suppressed`, A6.2 already transports those typed
findings, and B6.1 already scores context-suppressed language at zero while sending
ambiguous language to review. The focused analyzer and Phase 6 suite passes 202
tests. Full verification passes all 1,437 project tests, compilation, and the
generated preparation smoke test.

Current tracker: `outputs/a5_3_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 45 completed tasks out of 72 and B6.2 as the next task. Changes remain
uncommitted for review.

Next: B6.2 — Create risk-scoring tests for normal and edge scenarios.

In plain language: the analyzer now distinguishes direct concerning wording from
the same words used as an example, quotation, report, or unclear short question.
Phase 6 receives better evidence through the interfaces it already had, so its
completed work stays valid while false-positive scores become less likely.

## B6.2 — Complete

Added `tests/test_risk_scenarios.py` with 54 public-API scenarios for normal and
edge risk-scoring behavior. Representative evidence now exercises every severity
level, and complete parameter matrices verify all 13 acoustic labels, all six
language categories, speech review accumulation, ambiguity review, exact confidence
and severity boundaries, independent branch caps, and the final 100-point cap.

The tests also cover maximum acoustic aggregation, zero-weight safe and suppressed
evidence, each missing branch, nonmissing language absence, canonical combined
reasons, configurable fractional rounding, and exact review behavior immediately
below and at score 25.

The edge matrix identified one numeric robustness issue in B6.1: a contract-valid
review-required transcript count larger than floating-point range could overflow
before the speech cap was applied. `risk_scoring.py` now uses exact integer ratios
to check whether the count has reached the configured cap before multiplication.
The regression tests combine a count of `10**400` with both the built-in weight and
the smallest positive finite custom weight, confirming the branch deterministically
returns its 12-point cap and requires review.

Added `docs/risk-scoring-tests.md` and updated the scoring and project documentation.
The focused Phase 6 suite passes 132 tests. Full verification passes all 1,491
project tests, compilation, and the generated preparation smoke test.

Current tracker: `outputs/b6_2_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 46 completed tasks out of 72 and A6.3 as the next task. Changes remain
uncommitted for review.

Next: A6.3 — Add a trainable risk-model interface for future custom training.

In plain language: the scorer now has a complete worked-example test table, from
quiet input through critical combined evidence. It also stays bounded when a valid
counter is far larger than ordinary hardware can represent as a floating-point
number.

## A6.3 — Complete

Added the future-facing trainable risk-model interface in
`src/audio_sentinel/risk_model.py` without changing the deterministic B6.1 scorer.
The module defines immutable, versioned contracts for canonical feature vectors,
reviewed training targets and examples, reproducible model descriptors, and typed
predictions. Runtime-checkable trainer and predictor protocols leave the eventual
algorithm and model runtime replaceable while preserving one stable data boundary.

The feature extractor converts validated A6.1 inputs into ordered acoustic and
language inventories plus bounded speech workflow aggregates. It excludes source
identity, evidence IDs, transcript and matched text, speaker data, raw audio,
paths, and tensors. Canonical SHA-256 digests pin both source inputs and extracted
features, and validation rejects inventory drift, inconsistent totals or maxima,
non-present branches carrying data, altered hashes, mismatched severity bands, and
incomplete model provenance.

Added `docs/trainable-risk-model.md` and 34 focused tests covering deterministic and
content-sensitive hashes, privacy exclusions, missing and consent-limited branches,
all score-band boundaries, reviewed-label requirements, descriptor provenance,
prediction validation, protocol implementations, immutability, portable schema
export, and import without network or machine-learning runtime access.

The complete Phase 6 suite passes 166 tests. Full verification passes all 1,525
project tests, compilation, and the generated preparation smoke test.

Next: A6.4 — Validate risk-score behavior and thresholds.

In plain language: Phase 6 now has a carefully labeled socket for a future trained
model. The current rules calculator still makes today’s scores, while a later model
can receive the same compact evidence shape and return a score with enough version
and hash information to reproduce where it came from.

## A6.4 — Complete

Added the executable deterministic risk-policy acceptance boundary in
`src/audio_sentinel/risk_validation.py`. It loads a SHA-256-pinned 13-case suite,
expands each compact scenario through the public A6.1 `RiskInputSet` contract, runs
the public B6.1 scorer, and compares the exact score, severity, review decision,
reason codes, and missing branches with the expected result.

The checked-in suite validates all five severity levels, the inclusive 0.85
acoustic-confidence boundary, the score-25 human-review boundary, speech and
language branch caps, the total 100-point cap, transcript and ambiguity review,
context-suppressed zero weighting, missing-data review, consent-limited branches,
and the no-accepted-text state. The suite pins the exact built-in rule artifact, so
it cannot silently validate a different policy.

Generated `outputs/a6_4_validation/risk_score_validation.json` with 13 passed and
zero failed scenarios. Its validation ID is deterministic across run timestamps,
and the report records the suite hash, rule hash, threshold snapshot, complete
expected/observed results, and the explicit limitation that synthetic acceptance
validation is not empirical incident calibration.

Added `scripts/validate_risk_policy.py`, `docs/risk-score-validation.md`, and 19
focused tests covering suite integrity, required coverage, A6.1 expansion, exact
threshold metadata, passing and failing reports, rule mismatch, timestamp identity,
invalid scenarios, overwrite protection, CLI output, and offline operation.

The complete Phase 6 suite passes 185 tests. Full verification passes all 1,544
project tests, compilation, and the generated preparation smoke test.

Next: A7.1 — Define consensus policy and outcomes: no action, log, review, alert.

In plain language: Phase 6 now has a signed-off behavior checklist, not just a pile
of tests. The built-in scorer produces every required worked outcome and records
the exact thresholds it used, while clearly stating that real-world accuracy still
needs representative labeled evaluation later.

## A7.1 — Complete

Defined the Phase 7 consensus contract in
`src/audio_sentinel/consensus_contracts.py`. Every decision references one Phase 6
risk assessment by ID and canonical SHA-256, preserves its score, severity, reasons,
missing branches, and consent-limited branches, and records one typed agreement
state for each acoustic, speech, and language branch.

The v1 policy defines four mutually exclusive primary outcomes: `no_action` for a
complete zero-risk result, `log` for a low positive score without review triggers,
`review` for medium-or-higher risk or safety uncertainty, and `alert` for a local
alert candidate. Alert is permitted only for a critical score of at least 75 with
both acoustic and language support, no conflict, no missing evidence, and no Phase
6 transcript, ambiguity, or missing-data uncertainty reason. Speech presence alone
cannot count as risk support. Alert decisions retain the human-review requirement
instead of erasing it.

Added `docs/consensus-policy.md`, the checked-in
`docs/schemas/v1/consensus-decision.schema.json`, a complete example at
`docs/examples/consensus-decision.json`, and 30 focused tests. The tests cover all
four outcomes, exact policy thresholds, branch/status compatibility, canonical
ordering, risk-snapshot alignment, every alert gate, review-reason completeness,
privacy exclusions, immutable typed records, schema generation, and offline use.

Focused verification passes 30 tests. Full verification passes all 1,574 project
tests, compilation, and the generated preparation smoke test.

Current tracker: `outputs/a7_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 49 completed tasks out of 72 and B7.1 as the next task. Changes remain
uncommitted for review.

Next: B7.1 — Implement evidence-agreement and conflict-detection rules.

In plain language: the score is one opinion, and consensus is the safety gate that
decides what the system may do with it. Small clean signals can be logged,
uncertainty goes to a person, and only a clean critical result supported by both
sound and language can become a local alert candidate.
