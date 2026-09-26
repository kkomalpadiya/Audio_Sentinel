# Trainable risk-model interface

## Scope

A6.3 defines the stable data and protocol boundary for a future custom-trained risk
model. It does not train, load, or run a model. It also does not replace the B6.1
deterministic scorer, select a production model, combine multiple agents, or send
alerts.

The interface lives in `src/audio_sentinel/risk_model.py` and depends only on the
validated A6.1 risk inputs and existing score/severity semantics. Importing it does
not require a machine-learning runtime, a network connection, or model artifacts.

## Canonical features

`extract_risk_model_features()` converts a validated `RiskInputSet` into an
immutable `RiskModelFeatureVector`. The vector contains:

- explicit acoustic, speech, and language branch statuses;
- acoustic event totals, maximum confidence, and one ordered aggregate for every
  canonical event label;
- speech segment, accepted-transcript, and review-required-transcript counts plus
  the maximum VAD score; and
- language finding totals and one ordered count for every canonical category.

The feature contract rejects missing, duplicated, or reordered inventory entries,
inconsistent totals or maxima, and data attached to non-present branches. Feature
hashes use canonical JSON and SHA-256, so identical content has an identical digest
across runs.

Features intentionally exclude clip and consent identifiers, evidence IDs,
transcript text, matched text, speaker identity, raw audio, absolute paths, and
model tensors. Counts whose names mention transcripts describe workflow state only;
they do not contain transcript content.

## Training records

`RiskTrainingTarget` accepts a 0-100 score, its matching existing severity band,
and a label source of `human_reviewed` or `adjudicated`. Automatically generated or
unreviewed labels are outside this v1 interface.

`build_risk_training_example()` pairs that target with canonical features. Each
immutable `RiskTrainingExample` records:

- a bounded example identifier;
- a digest of the complete validated source risk inputs;
- a digest of the privacy-minimized feature vector; and
- the reviewed target.

The source digest supports provenance checks without copying source identity or
private content into the training record. The feature digest is rechecked whenever
the record is validated.

## Model and prediction records

`RiskModelDescriptor` records the interface and feature-schema versions, output
semantics, model and runtime versions, model-artifact digest, training-dataset
digest, and training/validation example counts. A concrete implementation must
provide this metadata before its output can be identified reproducibly.

`RiskModelPrediction` contains the descriptor, exact feature digest, 0-100 score,
matching severity, and optional 0-1 confidence. It is a candidate model output,
not a consensus decision, incident, or alert.

## Implementation protocols

Future implementations integrate through two runtime-checkable protocols:

- `RiskModelTrainer.train()` consumes reviewed training examples and optional
  validation examples, then returns a predictor.
- `RiskModelPredictor.predict()` consumes one validated feature vector and returns
  a provenance-bearing prediction. Its `descriptor` property identifies the model.

The protocols leave algorithm, artifact format, runtime, and deployment choices to
the future implementation while fixing the data boundary that those choices must
honor.

## Portable schemas

`risk_model_schema_documents()` returns JSON Schema documents for features,
training examples, model descriptors, and predictions. Call
`write_risk_model_schemas(output_directory)` when a non-Python consumer needs files.
Generated schema files are derived artifacts and are not checked into the project.

## Verification

Run the focused interface tests:

```powershell
python -m pytest tests/test_risk_model.py -q
```

Run all Phase 6 tests:

```powershell
python -m pytest tests/test_risk_contracts.py tests/test_risk_scoring.py tests/test_risk_integration.py tests/test_risk_scenarios.py tests/test_risk_model.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Next task

A6.4 will validate risk-score behavior and thresholds.
