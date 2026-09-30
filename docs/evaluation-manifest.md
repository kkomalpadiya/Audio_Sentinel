# Repeatable evaluation manifests and metrics

B9.1 adds a versioned collection-evaluation boundary on top of the Phase 8
application service. It does not implement a second model pipeline. Every manifest
case carries the same `EvaluationRequest` accepted by the local CLI and API, and
the runner calls `run_evaluation()` once per case.

## Manifest contract

An `evaluation_manifest` contains:

- a content-addressed `manifest_id`;
- a timezone-aware creation time that does not affect that identity;
- an ordered, nonempty set of unique cases;
- the consensus outcomes counted as a positive result; and
- human-supplied truth for each case.

Each case has a unique case ID and clip ID, one authorized Phase 8 request, a
required binary truth label, an optional exact expected outcome, and up to 32
non-sensitive category tags. The default positive outcomes are `review` and
`alert`. Therefore, if an exact outcome is supplied, `review` and `alert` require
`expected_positive=true`; `no_action` and `log` require false. This mapping is
stored in the manifest instead of being hidden in metric code.

The manifest is strict and immutable after validation. Unknown fields, duplicate
identities, noncanonical positive-outcome ordering, inconsistent truth labels,
naive timestamps, unsafe audio paths, and invalid authorization are rejected.
Loading is bounded to 2 MiB, rejects symlinks and non-files, can verify a caller-
supplied SHA-256, and recalculates the semantic manifest identity.

`configs/evaluation-manifest.example.json` shows the wire format. Its recording
paths are examples only; the files are not included.

## Evaluation run

`run_evaluation_manifest()` processes cases in manifest order through the shared
Phase 8 service. A successful case records only its opaque case and clip IDs,
categories, expected and observed labels, risk result, and report/audit/alert
identities. It does not copy the audio path, consent ID, transcript, raw audio,
absolute path, or exception message into the run report.

A failed case remains in the report with a stable error code. Known safe pipeline
codes are retained. Unknown exception details become `unexpected_error`. The
runner continues so one bad recording does not erase visibility into the rest of
the collection. A report with any failed case has
`decision_status=incomplete_due_to_failures`.

Metrics use completed cases only, while total, completed, failed, and completion
rate fields make the operational denominator explicit. A failure is never silently
converted into a negative prediction.

## Metric definitions

For the manifest's declared positive outcomes:

- true positive: expected positive and observed positive;
- false positive: expected negative and observed positive;
- false negative: expected positive and observed negative; and
- true negative: expected negative and observed negative.

The report calculates accuracy, precision, recall, specificity, F1, balanced
accuracy, false-positive rate, and false-negative rate. A rate is JSON `null` when
its denominator is zero; it is not presented as a misleading zero. Exact-outcome
accuracy is calculated separately over completed cases that provide
`expected_outcome`.

The formulas are:

```text
accuracy            = (TP + TN) / (TP + FP + FN + TN)
precision           = TP / (TP + FP)
recall               = TP / (TP + FN)
specificity          = TN / (TN + FP)
F1                   = 2TP / (2TP + FP + FN)
balanced accuracy    = (recall + specificity) / 2
false-positive rate  = FP / (FP + TN)
false-negative rate  = FN / (FN + TP)
```

The run ID is content-addressed from the manifest receipt, expected and observed
labels, risk results, safe failure codes, and metrics. It excludes the run time and
the report, audit, and alert receipt IDs issued by each local execution. Those
receipt IDs remain in the saved report for traceability, but do not make identical
predictions look like a different metric result. Repeating the same semantic run
therefore produces the same ID. Saving reuses an existing semantically identical
report and never overwrites a conflicting file.

Run documents permanently state `notification_delivery=not_sent` and
`alert_delivery_authorized=false`. Evaluation may create the same pending local
alert artifacts as Phase 8, but the collection runner adds no delivery mechanism
or authority.

## Running a manifest

Install the model assets required by the processing scopes in the manifest, place
the authorized recordings under `data/raw`, then run:

```powershell
python scripts/run_evaluation_manifest.py path/to/manifest.json `
  --expected-sha256 <pinned-manifest-sha256>
```

By default the report is saved as
`outputs/b9_1_evaluations/<run_id>.json`. Use `--output` to choose another new
path. The command prints only a compact JSON summary. The full report conforms to
`docs/schemas/v1/evaluation-run.schema.json`.

## Interpretation limits

B9.1 defines reproducible measurement mechanics. It does not establish dataset
representativeness, production thresholds, acceptable error rates, or real-world
accuracy. A9.1 supplies an authorized labeled collection, runs this boundary, and
documents the observed false positives, false negatives, failures, and dataset
limitations.
