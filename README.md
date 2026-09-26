# Project 1: Audio Sentinel

This repository contains the implementation plan and starter code for a multi-agent audio threat detection system.

## Confirmed build direction

- We are **not** building live Android streaming first.
- We **are** starting with offline audio clips and documented dataset sources.
- We will use **prebuilt agents** for speech recognition, acoustic event detection, and language understanding.
- We will design our **own risk assessment formula/model**.
- We will build a **verification and consensus layer** that combines the outputs of the other agents.

## First milestone

The first working milestone should be:

1. Load a recorded audio clip.
2. Preprocess it.
3. Run a baseline acoustic agent.
4. Run speech gating and transcription.
5. Produce a structured event summary and a risk score.

## Project layout

```text
configs/                 Dataset and runtime configuration
data/raw/                Downloaded source datasets (gitignored)
data/interim/            Resampled or sliced audio (gitignored)
data/processed/          Features and training-ready files (gitignored)
docs/                    Architecture, dataset notes, and panel material
models/                  Local model weights and exports (gitignored)
src/audio_sentinel/      Starter application package
tests/                   Tests
```

## Dataset strategy

The panel-facing dataset justification is documented in:

- `docs/dataset-sources.md`
- `docs/dataset-intake-checklist.md`

Do not download every dataset immediately. Start with the approved shortlist and record source, license, role, and target labels.

## Developer checks

Install the development dependencies once, then run the project verification script before committing:

```powershell
pip install -e .[dev]
.\scripts\verify_project.ps1
```

The test configuration intentionally collects only `tests/`, so third-party test files included inside downloaded datasets do not affect project verification.

## Audio preparation contract

A1.1 defines validated preprocessing settings and the prepared-audio manifest.
See [the preparation guide](docs/audio-preparation.md) for defaults, examples,
window rules, and a beginner-friendly explanation. B1.1 now implements
[local audio loading and input validation](docs/audio-loading.md), with a synthetic
smoke test. A1.2 implements [mono conversion, resampling, and volume normalization](docs/audio-transforms.md)
in memory. B1.2 adds [optional configurable noise reduction](docs/noise-reduction.md),
disabled by default. B1.3 adds [deterministic overlapping windows](docs/audio-segmentation.md)
with manifest-ready records. A1.3 adds [verified WAV persistence and manifests](docs/audio-persistence.md)
with safe repeat saves. A1.4 connects these components through a
[one-call preparation service](docs/audio-pipeline.md), compatible with the shared
preprocessor interface. B1.4 expands [component-test coverage](docs/preparation-component-tests.md)
for loading, noise reduction, and segmentation. A1.5 adds
[preparation integration tests and a sample-clip smoke test](docs/preparation-integration-tests.md).
Run `python scripts/smoke_test_preparation.py` for a generated-audio check; the standard
verification script also runs it. A2.1 defines the
[Log-Mel feature contract and metadata schema](docs/log-mel-features.md), including
validated settings, source-window linkage, shape rules, and JSON examples.
B2.1 implements [Log-Mel spectrogram generation](docs/log-mel-generation.md)
with numerical checks and a smoke test over saved preparation windows.
A2.2 adds [verified feature storage and source-window linkage](docs/feature-storage.md),
including checked reloads, repeat-save reuse, and source/consent checks.
B2.2 adds [feature shape, range, and determinism tests](docs/feature-property-tests.md),
including alternate recipes, exact dB relationships, and fresh-process reproducibility.
A2.3 provides a [one-call audio-to-feature service](docs/feature-pipeline.md),
with ordered verified output, configurable budgets, and retry reuse.
Run `python scripts/smoke_test_feature_pipeline.py` for the complete generated-sample check.
Phase 2 is complete. A3.1 selects [YAMNet and a versioned label mapping](docs/acoustic-model-selection.md).
B3.1 adds a [local, verified model loader](docs/acoustic-model-loading.md) with a
pinned artifact digest, isolated TensorFlow 2.21 runtime, vocabulary/signature
checks, and JSON-ready version metadata. The detector consumes prepared waveforms
through its own frontend; the saved generic Log-Mel recipe is not compatible with
YAMNet input features. Run `.\scripts\setup_yamnet.ps1` once to create the ignored
runtime and download the ignored model weights. A3.2 adds
[verified raw-score inference over prepared waveform windows](docs/acoustic-inference.md),
including padding-aware patch spans, source/model provenance, consent checks, and
resource limits. Run the real check with
`.\.venv\yamnet\Scripts\python.exe .\scripts\smoke_test_acoustic_inference.py`.
B3.2 adds [overlap-safe acoustic event aggregation](docs/acoustic-aggregation.md):
mapped classes use their maximum score, repeated patch support is unioned without
score inflation, and complete provenance is retained. Thresholds remain explicit
experimental inputs until evaluation. Run the real structural check with
`.\.venv\yamnet\Scripts\python.exe .\scripts\smoke_test_acoustic_aggregation.py`.
A3.3 adds [versioned timestamped acoustic-evidence JSON](docs/acoustic-evidence.md)
with sample-exact times, full source/model/window provenance, deterministic identity,
atomic repeat-safe persistence, and source/consent verification on reload. It remains
candidate evidence, not an incident or risk decision. Run the real persistence check
with `.\.venv\yamnet\Scripts\python.exe .\scripts\smoke_test_acoustic_evidence.py`.
B3.3 completes the [acoustic loader and aggregation unit-test matrix](docs/acoustic-unit-tests.md),
covering local artifact hardening, TensorFlow/signature drift, all mapped labels,
overlap/gap boundaries, deterministic ordering, limits, and malformed inputs.
A3.4 completes the [labeled acoustic evaluation](docs/acoustic-evaluation.md) with
separate calibration and holdout data, reproducible sampling, explicit coverage
gaps, and checked-in research-baseline metrics. Phase 3 is complete. A4.1 defines the
[speech-evidence contract and reliability rules](docs/speech-evidence.md), including
sample-exact provenance, consent scope, confidence typing, and deterministic
accepted/review/rejected handling. B4.1 adds the verified
[Silero VAD v6 wrapper](docs/vad-wrapper.md), using a pinned local ONNX artifact,
CPU-only runtime, recurrent-state isolation, bounded 32 ms frame scoring, and exact
tail-padding metadata. A4.2 adds [verified speech-segment extraction](docs/speech-segments.md):
it selects a complete prepared-window duration, maps relative VAD frames onto the
full clip sample grid, unions overlapping evidence without score inflation, and
emits transcript-free A4.1 evidence with exact timestamps. Run
`.\scripts\setup_speech.ps1` once on a fresh machine; the real extraction check is
`.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_speech_segments.py`. B4.2
adds a [verified offline transcription wrapper](docs/transcription-wrapper.md) for
the pinned English Faster-Whisper `tiny.en` model. It verifies every local model
file and exact runtime version, performs deterministic CPU decoding with bounded
inputs and outputs, and emits an explicitly derived—not calibrated—confidence
score. The real-model check is
`.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_transcription.py`. A4.3
adds the
[verified speech transcription orchestrator](docs/speech-transcription.md). It
reconstructs sample-exact segments from hash-checked prepared windows, transcribes
each independently, applies the recorded reject/review/accept policy, and exposes a
separate downstream handoff containing accepted text only. Run the real structural
check with
`.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_speech_transcription.py`.
B4.3 completes the
[VAD and transcription wrapper regression matrix](docs/speech-wrapper-tests.md)
with 184 offline cases covering artifact and runtime drift, exact resource
boundaries, input ownership, malformed model output, safe failures, and immutable
portable results. A4.4 completes the phase with 16
[speech-branch integration tests](docs/speech-integration-tests.md) spanning real
preparation and persistence through VAD segmentation, exact waveform reconstruction,
transcription policy handling, restart determinism, tamper rejection, and portable
final evidence. A5.1 adds the
[accepted-only language-evidence contract](docs/language-evidence.md) with six
bounded categories, typed reason codes, privacy-minimized transcript references,
exact rule-match spans, and a checked-in JSON Schema. It deliberately excludes
incidents, severity, risk scores, and alerts. B5.1 adds the
[versioned English language rules](docs/language-rules.md): 59 integrity-checked
keyword, phrase, and explicit-negation patterns with a fixed normalization recipe
and no risk or alert decisions. A5.2 adds the
[deterministic transcript-analysis engine](docs/language-analysis.md). It verifies
the exact accepted-only Phase 4 handoff, preserves original character spans through
normalization, prefers specific phrases over contained keywords, applies bounded
explicit-negation scope, and emits privacy-minimized A5.1 evidence. B5.2 adds the
[versioned English language fixtures](docs/language-fixtures.md): 74 synthetic
accepted-transcript cases with 75 expected findings, full active-rule and negation
coverage. A5.3 runs that complete matrix against the analyzer and adds bounded
hypothetical, quoted/reported, and insufficient-context safeguards while preserving
explicit-negation precedence and hard sentence boundaries.
A6.1 adds the [risk-assessment contract](docs/risk-assessment.md): hash-pinned
acoustic, speech, and language evidence summaries; explicit present, missing, and
not-permitted branch statuses; a fixed 0-100 score range; v1 severity bands; and a
missing-data policy that records absent evidence for review instead of treating it
as safe. B6.1 adds the [configurable risk-scoring engine](docs/risk-scoring.md):
SHA-256-pinned local score weights, bounded acoustic, speech, and language
contributions, deterministic assessment IDs, rule provenance, explicit reason
codes, and human-review triggers for missing or uncertain evidence. A6.2 adds the
[risk evidence integration boundary](docs/risk-integration.md), which verifies
cross-branch source and transcript provenance and converts Phase 3-5 evidence into
privacy-minimized scorer inputs with explicit missing and consent-limited states.
B6.2 adds the [risk-scoring scenario matrix](docs/risk-scoring-tests.md): 54 normal
and edge cases covering every built-in acoustic label and language category,
severity and review boundaries, branch caps, missing-data combinations, canonical
reasons, fractional scores, and extremely large valid counters. A6.3 adds a
[trainable risk-model interface](docs/trainable-risk-model.md) for future custom
training: canonical identity-free features, reviewed training targets, hash-pinned
examples, reproducible model descriptors, and typed trainer/predictor protocols.
It does not replace or change the deterministic B6.1 scorer. A6.4 completes the
phase with an [executable risk-policy validation](docs/risk-score-validation.md):
13 hash-pinned acceptance scenarios verify the built-in severity levels,
thresholds, caps, review triggers, missing-data behavior, and consent states while
explicitly avoiding any claim of empirical incident calibration.
A7.1 defines the [consensus policy and outcome contract](docs/consensus-policy.md):
one hash-pinned Phase 6 risk snapshot, typed acoustic/speech/language agreement
states, mutually exclusive no-action/log/review/alert outcomes, and conservative
alert gates requiring clean critical multi-branch support. An alert outcome is only
a local candidate for later Phase 8 handling; A7.1 sends no notification.
B7.1 implements the [evidence-agreement and conflict engine](docs/consensus-agreement.md)
with a SHA-256-pinned local rule artifact, explicit acoustic and language support
rules, neutral speech handling, four cross-branch contradiction checks,
deterministic evaluation identity, and privacy-minimized reason receipts. It
classifies evidence for A7.2 but does not select or send an alert outcome.
A7.2 implements the [final verification decision service](docs/consensus-service.md).
It binds the Phase 6 assessment to its B7.1 evaluation, recomputes the trusted
agreement result to reject stale or tampered handoffs, and applies the A7.1
precedence rules to select exactly one outcome. Critical evidence becomes a local
alert candidate only with clean acoustic and language support; every failed alert
gate remains reviewable, and the service performs no notification delivery.
B7.2 adds the [consensus safety test matrix](docs/consensus-safety-tests.md): 27
normal and edge-case checks covering all four outcomes, the exact acoustic
support boundary, every disagreement rule, all language categories, neutral
speech, missing and consent-limited evidence, and critical-score false-alert
blocks. It changes no production decision rule.
A7.3 defines the [trainable verification-model interface](docs/trainable-verification-model.md)
for future custom training: trusted Phase 6/B7.1 source validation, canonical
identity-free features, reviewed outcome targets, hash-pinned examples, reproducible
model descriptors, probability-bearing candidate predictions, and typed
trainer/predictor protocols. The interface is inactive and cannot replace A7.2 or
authorize an alert.
A7.4 completes Phase 7 with an
[executable end-to-end consensus validation](docs/consensus-validation.md): 13
SHA-256-pinned scenarios pass through the real Phase 6 scorer, B7.1 agreement
engine, and A7.2 final decision service. The generated report pins every policy
artifact and confirms all four outcomes, alert boundaries, conflicts, uncertainty,
missing evidence, consent limits, and safe-language behavior without claiming
empirical incident accuracy or sending notifications.
The detailed [Phase 7 completion report](docs/phase-7-verification-and-consensus-report.md)
explains the complete verification flow, algorithms, alternatives, technology
choices, safety controls, tests, examples, and limitations in common terms.

## Local panel demonstration

Run `python -m uvicorn audio_sentinel.main:app --app-dir src --host 127.0.0.1 --port 8000`
and open `http://127.0.0.1:8000/demo`. The frontend displays actual prepared audio
and saved/reloaded Log-Mel values for three generated signals. It includes audio
playback, window selection, padding boundaries, feature-value inspection, and PNG
export. See [the demo commands and speaking guide](docs/panel-frontend-demo.md).
No additional dependencies or model downloads are needed. Acoustic inference
is implemented in the backend but is not connected to this temporary frontend.
