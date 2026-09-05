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
smoke test. Next is A1.2: mono conversion, resampling, and volume normalization.
