# B1.4: Preparation component tests

Reviewed the existing loader, noise-reduction, and segmentation suites and added
22 checks for gaps. The three suites now contain 137 passing tests. The expanded
checks did not require changes to the audio-processing implementation.

## Behavior covered

| Component | Existing coverage | B1.4 additions |
| --- | --- | --- |
| Loader | WAV/FLAC formats, PCM scaling, source hash, containment, malformed files, limits, consent, changing sources | Independently constructed little/big-endian WAVs with odd metadata padding; short/extra decoded frames; decoder memory/I/O failures and retry; non-finite sample after the first read block; independent permission copy |
| Noise reduction | Background attenuation, tone/impulse preservation, disabled/short/silent behavior, channels, determinism, settings, memory and invalid data | Exact four-frame profiling boundary; alternate FFT sizes and sample rates; amplitude-scaling consistency; read-only strided input; silent-channel gain diagnostics; spectral allocation failure |
| Segmentation | Overlap and tails, sample positions, IDs, independent windows, lazy allocation, limits, consent expiry, manifest compatibility | 80 seeded window grids; permission before allocation; failed padding allocation; read-only strided input with independent writable outputs |

The window grid checks span both `pad` and `drop`, varied clip lengths, window
lengths, and hops. They verify actual sample slices, coverage, padding, unique IDs,
and whether another full window could fit. They do not reuse the implementation's
window-count formula as their expected result.

Decoder and allocation failures are injected with pytest's `monkeypatch` fixture.
This temporarily makes a dependency fail so the test can check the error code and
cleanup behavior without exhausting real memory or damaging an actual recording.
Pytest restores the original dependency after the test. Generated inputs use fixed
random seeds so a failure can be reproduced.

The two independent WAV fixtures are built from the container's bytes rather than
written through SoundFile. They check that integer decoding works with both byte
orders and that an odd-sized metadata chunk does not shift the audio samples.

## Running the checks

From the project root:

```powershell
python -m pytest tests/test_audio_loader.py tests/test_noise_reduction.py tests/test_segmentation.py -q
.\scripts\verify_project.ps1
```

The first command runs the three component suites. The second compiles the package
and runs the complete project suite. No new dependency, recording, or dataset
download is needed. Generated files live in pytest's temporary directories.

These are behavior checks, not a claim of exhaustive coverage or real-world noise
removal quality. A1.5 remains the broader preparation integration-test and
sample-clip smoke-test task. Review, commit, and push this checkpoint when ready.

## In plain language

A unit test gives one component a known input and checks its output. For example,
we know exactly which samples should appear in an overlapping window. Failure
tests also check what happens when input is broken or memory allocation fails.
Keeping these tests lets future changes reveal regressions automatically.
