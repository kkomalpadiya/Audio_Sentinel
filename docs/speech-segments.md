# Speech-segment extraction and exact timestamps

## What A4.2 adds

A4.2 connects verified prepared-audio windows to the B4.1 Silero VAD wrapper and
turns threshold-qualified 32 ms frames into deterministic, non-overlapping speech
segments. The implementation is in `src/audio_sentinel/speech_segments.py`.

The result includes an in-memory, transcript-free `SpeechEvidenceDocument` that
satisfies the A4.1 contract. It is candidate speech evidence, not a transcript,
speaker identity, language decision, incident, or risk classification. B4.2 owns
offline transcription; A4.3 will orchestrate transcription and low-confidence
handling.

## Verified input path

`extract_prepared_speech_segments()` accepts a project `Paths` object, a prepared
manifest path relative to `data/interim`, and an already verified `LoadedVadModel`.
Before returning evidence it:

1. validates the pinned Silero model before reading source files;
2. reads and validates the bounded prepared-audio manifest;
3. requires active `acoustic_and_speech` consent, 16 kHz audio, and one channel;
4. selects one complete window-duration group;
5. verifies each listed PCM-16 WAV, removes only recorded zero padding, and hashes
   the exact bytes;
6. runs B4.1 VAD independently for every selected window;
7. re-reads the manifest and every selected window to reject mid-run changes; and
8. builds and revalidates the A4.1 evidence document.

The default window choice is the shortest configured duration in the manifest.
That avoids scoring the same clip once for every 1 s, 5 s, and 10 s view while
preserving the short-window timing resolution. A caller may request another
configured duration explicitly. The chosen group must cover the complete prepared
clip without a sample gap; incomplete `drop`-tail inputs fail instead of silently
omitting the end of a recording.

## Absolute timestamp rules

B4.1 frame offsets are relative to one prepared window. A4.2 converts them to the
full prepared-clip sample grid:

```text
absolute_start_sample = window.start_sample + frame.start_sample
absolute_end_sample   = window.start_sample + frame.end_sample
seconds               = absolute_sample / sample_rate_hz
```

The final incomplete VAD frame stops at the real window end. Neither VAD right
padding nor prepared-window padding can extend a segment beyond recorded audio.
Sample offsets are authoritative; seconds are derived only after the final absolute
span is known.

## Threshold and merge behavior

The extractor uses `SpeechReliabilityPolicy.vad_speech_threshold`, whose A4.1
default is 0.60. A frame exactly equal to the threshold is included; a lower frame
is excluded. This threshold remains an uncalibrated starting value until a later
labeled VAD evaluation task.

Qualified frame spans are sorted by absolute time. Overlapping or exactly adjacent
spans are unioned. `merge_gap_samples` may explicitly bridge a larger gap; its
default is zero, so silence is not hidden by an arbitrary hangover rule. The final
segment score is the maximum contributing frame probability, never a sum or average
that could be inflated by overlapping windows.

Each segment receives a deterministic ID such as `speech-0000`. Its
`source_window_ids` include the selected prepared windows that overlap and jointly
cover the complete segment. Contributions retain their window hash, frame index,
absolute span, and probability in the diagnostic result. The public A4.1 evidence
contains no waveform and records `transcript: null` with the required
`not_transcribed` reliability assessment.

## Settings and limits

`SpeechSegmentationSettings` records the chosen duration override, merge gap,
minimum retained duration, and limits for manifest bytes, individual window bytes,
total source bytes, decoded samples, windows, VAD frames, and final segments.
Limits are checked before model work where possible. Exceeding one fails the whole
operation; partial evidence is never returned.

The semantic evidence ID is a SHA-256 hash of source, policy, model, window, and
segment content. Creation time is excluded, so an identical rerun has the same ID.
The full extraction result additionally records the segmentation settings, selected
window duration, input frame count, raw per-window VAD results, and frame
contributions needed to inspect how each final span was formed.

## Verification

Run the focused tests:

```powershell
python -m pytest tests/test_speech_segments.py -q
```

Run the real pinned-model structural check:

```powershell
.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_speech_segments.py
```

The real check prepares a generated 1.6-second clip, selects three overlapping
1-second windows, scores 83 VAD frames twice, verifies stable semantic identity,
and confirms that a zero-threshold structural policy unions the exact sample range
`[0, 25600)`. The zero threshold is used only to guarantee deterministic structural
coverage for generated non-speech audio; production calls use the recorded policy.

Run all project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Beginner-friendly explanation

The VAD model reports speech-likeness for tiny pieces inside each saved audio
window. A4.2 places every piece back on the original clip's timeline, removes
duplicates caused by overlapping windows, and joins neighboring positive pieces
into clean intervals. It keeps sample numbers as the source of truth so rounding or
padding cannot make the timestamps drift.
