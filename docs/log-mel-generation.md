# B2.1: Log-Mel spectrogram generation

`generate_log_mel` now computes the numeric features specified in
[A2.1](log-mel-features.md). It accepts a decoded prepared window as a float32
NumPy array shaped `(samples, 1)`, its sample rate, and `LogMelSettings`.

```python
import soundfile as sf
from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.log_mel import generate_log_mel

# window_path is a saved Phase 1 window whose source and permission were checked.
samples, rate = sf.read(window_path, dtype="float32", always_2d=True)
result = generate_log_mel(samples, rate, LogMelSettings())
print(result.values.shape)  # (64, 97) for a default 1-second window
```

The result contains an owned C-contiguous float32 array, the validated recipe,
input sample count, additional analysis padding, actual librosa version, and
implementation version. It does not fabricate file paths or hashes for unsaved
features. Generator results remain in memory; [A2.2](feature-storage.md) now saves
feature files and verifies source linkage,
and [A2.3 provides the integrated pipeline service](feature-pipeline.md).

## Behavior

- Reject empty/non-float32 audio, non-finite samples, stereo, and mismatched rates.
- Preserve source samples, including read-only or strided arrays.
- Right-pad inputs shorter than one FFT; otherwise keep complete FFT frames.
- Use a periodic Hann window, Slaney area-normalized Mel filters, squared FFT
  magnitudes, and fixed-reference power dB with the configured dynamic range.
- Reject empty/non-finite filters produced by unsuitable custom recipes.
- Compute with float64 intermediates, then return float32. This keeps finite
  float32 extremes and very small positive dB floors representable during computation.
- Complete silence produces -100 dB with the default recipe. The maximum is not
  forced to zero; this is a numeric feature matrix, not an image.

The function is a pure signal transform. It performs no file I/O, resampling,
normalization, model loading, or consent decisions. Callers must verify source
files and current permission, and keep the input stable during the call. The
smoke test uses generated audio with synthetic permission only.

## Memory and failures

The recipe's `max_feature_bytes` limits the returned payload. The independent
`max_working_bytes` argument defaults to 256 MiB and checks a conservative array
workspace estimate before signal/FFT/filter allocations. The estimate includes
signal conversion/padding, spectra, power, filter construction, Mel/log arrays,
output conversion, and allocation slack. It excludes caller-owned input, Python
and library loading/JIT overhead, native library caches, and unrelated processes;
it is not a hard process-memory cap. No hidden cross-call feature cache is used.

`LogMelError.code` distinguishes invalid samples/rates, mono requirements,
non-finite audio, invalid limits, oversized output/workspace, invalid filters or
features, and allocation failures. Invalid recipes raise Pydantic validation
errors before signal processing.

## Verification

```powershell
python -m pytest tests/test_log_mel.py -q
python scripts/smoke_test_log_mel.py
.\scripts\verify_project.ps1
```

The tests compare output to independent NumPy FFT/framing and explicit Slaney
triangles and dB calculations. They also check signal-level behavior, boundary
lengths, silence, extreme finite amplitudes, input preservation, repeatability,
invalid inputs, empty filters, and allocation guards/failures. [B2.2 adds expanded
feature shape/range/determinism coverage](feature-property-tests.md).

The smoke script prepares a generated 48 kHz stereo tone, reloads all five saved
mono PCM16 windows, computes features, and checks repeatability and unchanged raw
and prepared files. Expected shapes are three `(64, 97)` arrays, one `(64, 497)`,
and one `(64, 997)`. Temporary files are removed. No download or recording is needed.

## In plain language

Each prepared audio window now becomes a table showing how energy in different
frequency bands changes over time. The next task will save these tables and
record exactly which audio window produced each one.
