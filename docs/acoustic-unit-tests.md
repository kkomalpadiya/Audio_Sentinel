# B3.3 — Acoustic loader and aggregation unit tests

## What this task verifies

B3.3 completes the focused unit-test matrix around the two deterministic acoustic
boundaries introduced in B3.1 and B3.2. These tests use fake TensorFlow metadata
and synthetic score arrays, so the normal suite remains fast, offline, and
independent of the optional TensorFlow runtime.

The focused suite now contains 96 tests across `test_acoustic_loader.py` and
`test_acoustic_aggregation.py`.

## Loader coverage

The loader tests verify successful local loading and JSON-ready version metadata,
plus failure behavior for:

- missing, traversing, absolute, linked, or junction-backed model paths;
- missing, unexpected, linked, oversized, or mid-hash-changing artifact files;
- valid and invalid Kaggle completion markers without including the marker in the
  model artifact identity;
- artifact digest changes before, during, and after TensorFlow loading;
- missing or wrong TensorFlow runtime versions;
- callable/signature inventory, input/output names, shapes, and dtypes;
- TensorFlow load and inspection exceptions;
- redirected vocabulary assets; and
- immutable returned metadata and tensor contracts.

The tests also prove that importing the production loader does not import
TensorFlow and that B3.1 loading does not run model inference.

## Aggregation coverage

The aggregation tests exercise every mapped class for all six project acoustic
labels. They verify maximum-per-label selection rather than score summation,
deterministic tie handling, exact threshold inclusion, and exclusion immediately
below a threshold.

Temporal cases cover overlapping windows, repeated patches from one window,
tail-patch clipping, exact adjacency, exact configured-gap boundaries, transitive
gap chains, separated evidence, reversed input order, source-window deduplication,
and deterministic event/contribution ordering. Repeated overlap never inflates the
peak score.

Validation cases cover the complete pinned model metadata, tensor dtype/shape and
finite score range, duplicate windows, real input lengths, malformed patch support,
threshold coverage/order, strict integer settings, exact contribution limits,
empty inputs, immutable outputs, and preservation of input score arrays.

## Commands

Run the fast focused suite:

```powershell
pytest -q tests/test_acoustic_loader.py tests/test_acoustic_aggregation.py
```

Run the real local artifact and aggregation checks separately:

```powershell
.\.venv\yamnet\Scripts\python.exe .\scripts\smoke_test_acoustic_loader.py
.\.venv\yamnet\Scripts\python.exe .\scripts\smoke_test_acoustic_aggregation.py
```

The real smoke checks validate the installed pinned model. They still do not
measure detection quality or approve thresholds; A3.4 owns labeled evaluation.

In plain language: the model loader and overlap logic now have explicit tests for
their normal behavior and their dangerous edge cases. This task verifies software
correctness, while the next task measures whether the recognizer is accurate.
