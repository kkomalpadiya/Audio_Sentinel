# B1.1: Local audio loading and input validation

The loader now reads an authorized local recording and returns its numeric samples
and source metadata. It does not resample, convert to mono, adjust volume, remove
noise, write prepared files, transcribe, or score threats. A1.2 now provides
[mono conversion, resampling, and normalization](audio-transforms.md) for its output.

## What happens when a file is loaded

1. Check the clip identifier and a fresh validated copy of its consent record.
   Permission must allow acoustic processing on an authorized device, already be
   effective, and not be expired. Timestamps must include a timezone. These checks
   run before accessing the source, and permission is checked again after decoding.
2. Resolve the path inside the configured `data/raw/` directory. Relative paths
   start at this root; absolute paths must remain inside it. Links resolving outside
   the root, missing files, and directories are rejected.
3. Open the file read-only and enforce its disk-size limit. Inspect the decoded
   container format, sample rate, channels, and duration before allocating samples.
   Check WAV chunk boundaries because some decoders tolerate truncated WAV files.
4. Decode in blocks to a float32 NumPy array shaped `(frames, channels)`. Mono still
   has a channel dimension: `(16000, 1)` is one second of mono audio at 16 kHz.
   Integer PCM is scaled into full-scale units. For 16-bit PCM, divide by 32,768:
   −32,768 becomes −1.0. Floating WAV values retain their amplitude, even above 1.0.
   Silence remains valid; NaN and infinity are rejected. Check actual decoded length.
5. Hash the original file bytes with SHA-256 and record format, sample rate, channel
   count, frame count, size, and a relative path in `SourceAudioMetadata`. Reject
   detectable size/timestamp changes while reading. The returned `LoadedAudio`
   carries the clip ID, consent, metadata, and samples for the next module.

The loader uses the existing SoundFile dependency. Its documentation explains
[sample arrays and float decoding](https://python-soundfile.readthedocs.io/en/latest/#soundfile.read).
No dependency installation is needed on the current project machine.

## Limits and supported input

The A1.1 settings still apply: WAV/FLAC, 8,000–192,000 Hz, up to 8 channels,
600 seconds, and 256 MiB on disk by default. WAV extensible headers are treated as
WAV. Container detection uses file contents, so renaming AIFF to `.wav` does not
make it acceptable. RAW, MP3, RF64, and other containers are outside this contract.

B1.1 adds `max_decoded_bytes` (256 MiB by default) for the main float32 sample
array. Compressed file size alone cannot bound this memory use. The loader checks
`frames * channels * 4` against this limit before allocation. Small temporary read
blocks and decoder overhead use additional memory. The value is configurable in
`AudioSettings` and the JSON example. Old configurations pick up the new default.
Clips too short to produce even one sample at the target rate are rejected.

The existing full-pipeline `AudioPreprocessor` interface is unchanged. This loader
is the input step that the future preprocessing service will call. It preserves
the source rate, channel count, and amplitude regardless of the conversion settings.

## Using the loader from Python

```python
from pathlib import Path
from audio_sentinel.audio_loader import AudioLoadError, load_audio
from audio_sentinel.config import load_settings
from audio_sentinel.contracts import ConsentRecord
from audio_sentinel.interfaces import InputAudio

settings = load_settings()
# Use actual approved permission evidence; do not invent a grant for real recordings.
consent = ConsentRecord.model_validate_json(Path("approved-consent.json").read_text())
clip = InputAudio("clip-001", Path("dataset/recording.wav"), consent)
try:
    loaded = load_audio(clip, settings)
    print(loaded.samples.shape, loaded.source.sample_rate_hz)
except AudioLoadError as error:
    print(error.code, str(error))
```

`AudioLoadError` is a `ValueError` with a stable `code`. Codes distinguish permission,
path, format, empty/corrupt audio, size/duration/channel/rate, non-finite samples,
source changes, and allocation failures. `now=` accepts an aware datetime for
deterministic tests; production callers should omit it to use the actual UTC clock.

This is an offline local loader. It does not establish real-world permission from
a file's presence, or defend against an adversary swapping filesystem entries while
the process runs. Use stable source files and real permission evidence. Hashes
identify the original bytes; they do not certify the file's origin or permission.

## Try it without a dataset

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/smoke_test_audio_loader.py
```

The script generates a 0.1-second tone in a temporary directory, loads it, checks
PCM scaling, prints a result, and removes the temporary audio. Its permission record
is explicitly for synthetic test audio. Nothing is recorded or downloaded.
For all project checks, run `./scripts/verify_project.ps1`.

## In plain language

A sound file stores measurements of air-pressure changes as numbers. The loader
unpacks those numbers into a table: each row is a moment in time, and each column
is an audio channel. It checks permission and file validity first, then returns the
table with a record of its source. A1.2 standardizes the channels, sample rate, and
volume so the analysis modules receive consistent audio.
