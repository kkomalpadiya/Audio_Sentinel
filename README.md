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
verification script also runs it. Next is A2.1: the Log-Mel feature contract and metadata schema.
