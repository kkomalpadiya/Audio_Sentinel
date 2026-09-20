# A4.4 — Speech-branch integration tests

## What this task verifies

A4.4 completes Phase 4 with 16 integration cases that exercise the full offline
speech branch across its component boundaries:

1. load, transform, window, and persist authorized source audio;
2. reopen and verify the prepared manifest and PCM16 windows;
3. score every selected window through the VAD wrapper contract;
4. merge frame evidence into sample-exact speech segments;
5. reconstruct each segment from its verified overlapping windows;
6. transcribe every segment independently; and
7. apply the reliability policy before exposing accepted text downstream.

Preparation, resampling, mono conversion, windowing, persistence, hashing, manifest
validation, segment merging, reconstruction, evidence construction, and policy
handling are real production code. The tests use contract-accurate fake VAD and
transcription runtimes so the ordinary suite remains deterministic, offline, and
independent of downloaded model artifacts.

## Integration matrix

| Scenario | Cross-stage behavior verified |
| --- | --- |
| 48 kHz stereo source with overlapping windows | Preparation produces 16 kHz mono audio; duplicate VAD support merges into one segment and one exact transcription call |
| WAV/FLAC, 8–48 kHz, mono/stereo matrix | Four source combinations reach accepted final evidence through resampling, persistence, VAD, reconstruction, and transcription |
| All transcript reliability outcomes | Rejected, review-required, accepted, and empty candidates retain segment order; only accepted text enters the downstream collection |
| No speech | Every window is checked, no segment is created, the transcriber is not called, and both model identities remain recorded |
| Padded final window | Verified zero padding is excluded before VAD and transcription; the final segment stops at the real clip boundary |
| Custom reliability policy | The same versioned policy controls both VAD admission and transcript handling without being replaced between stages |
| Fresh services and model objects | Reused persisted preparation and independent runs produce the same semantic segmentation and transcription identities |
| Raw source removed after preparation | Later speech stages use only the verified prepared bundle and do not depend on reopening the original recording |
| Input immutability | The raw source and every persisted bundle file keep the same hashes after a successful branch run |
| VAD failure | Segmentation stops with a safe stable error before transcription can start |
| Later transcription failure | The operation raises a safe error instead of returning partially completed final evidence |
| Window tampering between stages | A changed prepared window is rejected before the transcription model is called |
| Evidence round trip | Final evidence validates after JSON serialization and contains no waveform or absolute filesystem path |

## Test isolation correction

The combined speech-suite run exposed an order-dependent test defect in the earlier
VAD import-isolation check. It reloaded `audio_sentinel.vad` inside the pytest
process, replacing module-level dataclass and model-spec identities while later
tests still referenced the original objects. The check now runs in a child Python
process, matching the transcription import test. It still proves that a normal
import does not load ONNX Runtime, without changing shared process state.

## Commands

Run only the integration cases:

```powershell
python -m pytest -q tests/test_speech_integration.py
```

Run the complete isolated and integrated speech suite:

```powershell
python -m pytest -q tests/test_vad.py tests/test_speech_segments.py tests/test_transcription.py tests/test_speech_transcription.py tests/test_speech_integration.py
```

Run the existing real-model structural check separately:

```powershell
.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_speech_transcription.py
```

The real smoke test uses the pinned local Silero and Faster-Whisper artifacts. The
integration suite verifies software handoffs and failure behavior; neither test set
measures real-world speech detection or transcription accuracy.

In plain language: the speech components have now been tested as one connected
branch, starting with an authorized recording and ending at the acceptance-only
text handoff. The tests prove that timing, hashes, policies, and failures remain
consistent as data moves between the stages.
