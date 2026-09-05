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

Current tracker: `outputs/b1_1_tracker_update/Audio_Sentinel_Master_Task_List.xlsx`.
It records 11 completed tasks out of 72. No manual installation or download is
needed. Review and commit this checkpoint when ready.

Next: A1.2 — Implement mono conversion, resampling, and loudness normalization.

The loader turns the audio file into a table of numbers, checks that the file and
permission are valid, and attaches a record of where those numbers came from.
