# Trainable verification-model interface

## Scope

A7.3 defines the stable data and protocol boundary for a future custom-trained
verification model. It does not train, load, or run a model. It does not replace
the B7.1 agreement engine or A7.2 final verification service, change consensus
policy, create an incident, or send an alert.

The interface lives in `src/audio_sentinel/verification_model.py`. It depends only
on validated Phase 6 and Phase 7 contracts. Importing it requires no
machine-learning runtime, network connection, or model artifact.

## Trusted source boundary

`extract_verification_model_features()` accepts one `RiskAssessmentDocument` and
its `ConsensusAgreementEvaluation`. Before extracting features, it uses the A7.2
verification service to:

1. revalidate both complete source documents;
2. confirm the agreement assessment ID and SHA-256 match the supplied assessment;
3. rerun B7.1 with the trusted agreement-rule artifact; and
4. require the rerun to equal the supplied agreement exactly.

The built-in B7.1 rule artifact is trusted by default. A custom agreement can be
used only when the caller explicitly supplies the matching validated
`LoadedAgreementRuleSet`. Invalid, mismatched, or untrusted sources fail with the
safe `invalid_sources` interface code.

## Canonical features

The immutable `VerificationModelFeatureVector` contains:

- the Phase 6 risk score, severity, and human-review flag;
- one fixed-order presence indicator for every Phase 6 risk reason;
- one fixed-order acoustic, speech, and language branch record;
- each branch's input status and B7.1 agreement state;
- one fixed-order presence indicator for every B7.1 agreement reason on every
  branch; and
- derived counts for supporting, alert-eligible supporting, conflicting, missing,
  and not-permitted branches plus the conflict flag.

The feature contract revalidates each branch through the B7.1 branch-evaluation
contract and rejects missing, duplicated, reordered, or inconsistent inventories.
Feature hashes use canonical JSON and SHA-256, so identical features have the same
digest across runs.

Features intentionally exclude assessment, agreement, clip, consent, evidence,
and decision IDs; timestamps; transcript or matched text; speaker identity; raw
audio; paths; model tensors; and the A7.2 outcome. Excluding the current outcome
prevents direct target leakage into supervised examples.

## Training records

`VerificationTrainingTarget` accepts one A7.1 outcome and requires a label source
of `human_reviewed` or `adjudicated`. Generated or otherwise unreviewed labels are
outside this v1 interface.

`build_verification_training_example()` pairs that target with canonical features.
Each immutable `VerificationTrainingExample` records:

- a bounded example identifier;
- the SHA-256 of the complete validated Phase 6 assessment;
- the SHA-256 of the complete validated B7.1 agreement evaluation;
- the SHA-256 of the identity-free feature vector; and
- the reviewed target.

The source hashes preserve exact provenance without copying source identities or
private content into the feature vector or training record. The feature digest is
rechecked whenever the record is validated.

## Model and prediction records

`VerificationModelDescriptor` pins the interface, feature, Phase 6 assessment,
B7.1 agreement, and A7.1 policy versions. It also records the model and runtime
versions, model-artifact digest, training-dataset digest, and training/validation
example counts.

`VerificationModelPrediction` contains:

- the exact model descriptor and feature digest;
- one probability for every outcome in canonical order;
- the maximum-probability recommended outcome;
- a confidence value equal to that outcome's probability;
- `requires_deterministic_verification=true`; and
- `alert_is_local_candidate_only=true`.

Probabilities must be finite, lie between zero and one, and sum to one within a
small numeric tolerance. A prediction is only a candidate recommendation. Even an
`alert` recommendation is not an A7.2 decision and cannot authorize notification.

## Implementation protocols

Future implementations integrate through two runtime-checkable protocols:

- `VerificationModelTrainer.train()` consumes reviewed training examples and
  optional validation examples, then returns a predictor.
- `VerificationModelPredictor.predict()` consumes one validated feature vector
  and returns a provenance-bearing candidate prediction. Its `descriptor`
  property identifies the exact model.

The protocols leave the algorithm, artifact format, runtime, training process, and
deployment choice to a future implementation while fixing the boundary those
choices must honor.

## Portable schemas

`verification_model_schema_documents()` returns JSON Schema documents for feature
vectors, training examples, model descriptors, and predictions. Call
`write_verification_model_schemas(output_directory)` when a non-Python consumer
needs files. Generated schema files are derived artifacts and are not checked into
the project.

## Verification

Run the focused interface tests:

```powershell
python -m pytest tests/test_verification_model.py -q
```

Run the Phase 7 tests:

```powershell
python -m pytest tests/test_consensus_contracts.py tests/test_consensus_rules.py tests/test_consensus_service.py tests/test_consensus_safety.py tests/test_verification_model.py -q
```

Run the complete project suite:

```powershell
python -m pytest -q
```

## Next task

A7.4 will validate consensus outcomes across end-to-end scenarios. The trainable
interface remains inactive until a later implementation is explicitly selected,
trained on reviewed data, evaluated, and integrated behind the deterministic
verification boundary.
