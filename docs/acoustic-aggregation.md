# B3.2 — Acoustic event aggregation across overlapping windows

## What this task adds

`audio_sentinel.acoustic_aggregation.aggregate_acoustic_events` converts A3.2's
raw YAMNet patch scores into deterministic candidate intervals. It performs two
different combinations:

1. For one project label in one patch, it takes the maximum score among that
   label's mapped YAMNet classes. It never sums parent, child, or proxy classes.
2. For one label over time, it unions thresholded patch support that overlaps,
   touches, or falls within an explicitly configured sample gap. It keeps the
   maximum score across that union. It never adds or averages repeated evidence.

This prevents preparation overlap from multiplying the apparent confidence of a
single sound. Every accepted patch remains attached as provenance, including its
window ID/hash, patch index, clipped sample support, mapped score, and winning
YAMNet class. Events are ordered chronologically with a deterministic label
tiebreaker. Separate labels are never merged with each other.

## Thresholds remain experimental

A3.4, not this task, will evaluate useful thresholds against labeled recordings.
`AcousticAggregationSettings` therefore requires one explicit threshold for each
of the six mapped acoustic labels. It has no hidden or claimed production default.
The tuple must cover exactly those labels, without duplicates, in stable label
order. `AcousticAggregationSettings.uniform(value)` is a convenience for tests and
experiments; it does not mean that one threshold is appropriate for every label.

Scores are independent, uncalibrated YAMNet sigmoid outputs. `peak_score` is model
evidence, not the probability that a real incident occurred. The output type is
`AcousticEventCandidate`, not the public `EventAnnotation`; it has no risk level.
Cross-label reasoning, incident verification, and risk scoring remain later tasks.

## Time behavior

All B3.2 boundaries use integer samples at 16 kHz. Patch support comes from A3.2
and is already clipped to real audio, so preparation padding and YAMNet's internal
padding cannot extend an event beyond recorded samples.

With the default `merge_gap_samples=0`, overlapping or exactly adjacent positive
patches for the same label become one interval. A positive gap can bridge brief
score dropouts, but the caller must choose it explicitly. These intervals describe
coarse model support. They are not exact sound onset/offset measurements.

## Safety and failure behavior

Before aggregation, the implementation revalidates the complete pinned YAMNet
metadata and each in-memory window's expected patch, embedding, and spectrogram
shapes. Scores must be finite float32 values within `[0, 1]`; window IDs must be
unique; real input lengths and patch spans must be consistent. A configurable
contribution limit prevents a permissive threshold from creating an unbounded
evidence inventory. A failure returns no partial aggregation result.

The input score arrays are read only and never modified. The result copies the
model, clip, manifest, and raw-audio identities from the A3.2 snapshot. A3.3 will
define the persistent timestamped JSON contract; B3.2's `to_summary()` is only a
diagnostic representation.

## Verification

The focused tests cover class mapping, threshold boundaries, overlap union,
non-overlap, exact gap behavior, chronological ordering, deterministic ties,
provenance, limits, malformed model metadata, malformed tensors, empty inputs,
and input preservation.

The real-model structural smoke test uses a generated 1.6-second tone, three
overlapping preparation windows, and five YAMNet patches. Its threshold is zero
only so every mapped label exercises the merge path; the result makes no accuracy
claim. Run it with:

```powershell
.\.venv\yamnet\Scripts\python.exe .\scripts\smoke_test_acoustic_aggregation.py
```

In plain language: several excerpts can contain the same sound. This step groups
their repeated model evidence into one time interval and keeps the strongest score,
while retaining every source patch so the grouping can be audited later.
