# B1.2: Optional configurable noise reduction

`src/audio_sentinel/noise_reduction.py` implements the stationary spectral gate
specified in A1.1. The order is mono conversion, resampling, optional noise reduction,
then RMS normalization. Noise reduction stays **disabled by default**. No new
libraries or model downloads are required on the current project machine.

## How the algorithm works

For each channel separately, SciPy's
[ShortTimeFFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.from_window.html)
splits the signal into frequency measurements using a periodic Hann window and 75%
overlap. The low temporal quantile of each frequency's magnitude estimates a
stationary background spectrum. Only frames fully inside the recording contribute
to that estimate, so padded edges cannot bias it.

The threshold is the estimated spectrum times `threshold_multiplier`. A bin below
the threshold receives maximum attenuation. At twice the threshold or above it
receives none; between these levels the decision changes linearly. A 3-by-3 average
smooths decisions over neighboring frequencies and times. Spectral gain is
`1 - reduction_strength * (1 - activity)`, where activity is in `[0, 1]`.
Frequencies with zero estimated background receive no direct suppression.

The original phase is retained. SciPy's
[inverse transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.istft.html)
reconstructs exactly the original number of frames. The module preserves channel
count, rate, timing, and input arrays. It does not crop or write audio. Spectral
gain is bounded by `[1 - strength, 1]`; this does not impose a peak ceiling on the
reconstructed waveform. The later normalization step handles that.

## Configuration

Options live under `AudioSettings.noise_reduction` and appear in the example JSON
configuration and regenerated schemas. Existing configurations inherit defaults.

| Option | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Exact copy when disabled; FFT is not called. |
| `method` | `stationary_spectral_gate` | Implemented algorithm. |
| `reduction_strength` | `0.5` | Range `(0, 1]`; higher allows more attenuation. |
| `fft_size` | `512` | Power of two, 64–8192. Default is 32 ms at 16 kHz. Hop is one quarter of this size. |
| `noise_quantile` | `0.2` | Range `(0, 0.5]`; temporal quantile of magnitudes for each frequency. |
| `threshold_multiplier` | `1.5` | Range `(0, 10]`; scales the background threshold. |
| `max_working_bytes` | `268435456` | 256 MiB budget for estimated per-channel spectral workspace. |

```python
from audio_sentinel.config import AudioSettings, NoiseReductionSettings
from audio_sentinel.audio_transforms import prepare_signal

settings = AudioSettings(noise_reduction=NoiseReductionSettings(
    enabled=True, reduction_strength=0.5,
))
# loaded is authorized B1.1 loader output.
prepared = prepare_signal(loaded, settings)
print(prepared.noise_reduction)
```

Direct numeric use is `reduce_noise(samples, sample_rate_hz, noise_settings,
max_decoded_bytes=...)`. It expects finite float32 `(frames, channels)` arrays.
The preparation helper continues to check permission before and after processing.

## Bypass, diagnostics, and limits

Disabled, all-zero, and insufficient-length recordings return independent exact
copies. Estimation requires four fully contained FFT frames. At the defaults, this
requires at least 896 samples, or 56 ms at 16 kHz. Shorter clips report `too_short`
and are not artificially extended for estimation.

`NoiseReductionStats` records `reason` (`disabled`, `too_short`, `silence`, or
`processed`), algorithm version `1.0`, profile-frame count, and average spectral
gain. Average gain is a diagnostic, not an SNR measurement. `PreparedSignal` now
includes it alongside separate normalization measurements.

The main signal still respects `max_decoded_bytes`. The separate workspace estimate
is `64 * frequency_bins * time_frames + 24 * signal_frames`, checked before STFT.
Channels are processed sequentially. Over-budget input raises
`noise_working_memory_exceeded`, rather than silently skipping a requested operation.
Long clips can exceed this budget even if loading succeeds. The estimate is not a
hard process memory cap: input/output arrays and library/application memory coexist.
Allocation failures report `insufficient_memory`.

This baseline assumes reasonably stable background noise. Persistent tones,
alarms, or speech can enter the noise estimate and be attenuated. Changing
backgrounds are also a poor fit. The default remains off pending dataset evaluation.
Stronger settings can remove useful sound, and subsequent normalization can raise
residual noise. Synthetic tests do not establish threat-detection accuracy.

## Run the synthetic check

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/smoke_test_noise_reduction.py
```

The fixed example uses an intermittent 1 kHz tone plus stationary Gaussian noise.
At strength 0.8, SNR improves from about 14.06 to 17.51 dB before normalization.
The script also verifies exact disabled behavior and unchanged frame count. It
uses no recordings, files, or downloads. Run `./scripts/verify_project.ps1` for all
checks, including channel independence, deterministic output, impulse timing,
short/silent inputs, invalid settings, memory limits, and processing order.

## In plain language

The module examines the frequencies in short pieces of audio and estimates what
background sound tends to be present. It turns down components near that estimate
and joins the pieces back together. Turning it off leaves the samples unchanged.
It cannot know which sounds matter, so the optional setting lets us compare results
before deciding whether to use it on real recordings.
