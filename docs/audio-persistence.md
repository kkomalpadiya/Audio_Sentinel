# A1.3: Prepared-output persistence and manifests

`save_prepared_audio` writes a prepared full clip, its overlapping windows, and a
validated JSON manifest. This completes the disk-saving component. A1.4 will join
loading, preparation, and saving behind one pipeline service.

## Output layout

All manifest audio paths are relative to the configured interim directory:

```text
data/interim/prepared/<key>/
    audio.wav
    windows/
        g0000-s000000000000.wav
        ...
    manifest.json
```

The key is shared with segmentation: SHA-256 over clip ID, original source hash,
the complete audio settings, and preprocessing version. Identical inputs and
settings select the same directory. Changes to the audio recipe select another.
The key identifies that combination; it is not a checksum of the encoded output.

The manifest uses the existing version 1.0 contract. It records source path/hash,
source dimensions, preprocessing settings, prepared dimensions and duration,
dataset, permission evidence, annotations, and the complete window inventory.
Window records retain actual start/end sample positions and zero-padding counts.
Annotations keep their full-clip times; saving does not relabel or infer events.

## Saving and repeating a save

The saver validates permission and input metadata, checks output limits, and
creates a private `.pending-*` directory beside the destination. It writes WAVs
there, flushes each file, then reads back every encoded sample and checks the rate,
channel count, frame count, format, and subtype. It writes the manifest last,
reloads it through the schema validator, and checks permission again before
renaming the complete directory into place.

Ordinary failures remove only that operation's staging directory. A process kill
or power loss may leave a hidden staging directory; consumers must use final
bundle directories and their manifests. Local rename publishes the folder as one
unit. This is not a power-loss recovery system, and OneDrive synchronization does
not provide the same all-at-once publication guarantee on another computer.

Repeating a save stages and verifies the requested result, then compares every
file byte and directory entry with existing output. A match returns `reused=True`
without changing existing files or timestamps. Missing, extra, or different
content raises `output_conflict`; it is never automatically overwritten. This
also applies if dataset, consent, or annotations change while the audio key stays
the same. Concurrent cooperating writers reuse the first matching bundle.
For a conflict, inspect or deliberately relocate the previous bundle before
retrying. A higher-level overwrite/versioning workflow is not part of this task.

## Encoding, limits, and permission

WAV output is little-endian PCM16. Each floating sample is clamped to the signed
16-bit range, multiplied by 32,768, and rounded to the nearest integer (ties to
even). Unsaturated samples differ by at most half a quantization step, 1/65,536.
Values outside [-1, 32767/32768] saturate; this matters if normalization was
disabled. The original in-memory float samples are unchanged.

`AudioSentinelSettings.persistence` supplies `PersistenceSettings`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `max_output_bytes` | 1,073,741,824 (1 GiB) | Maximum size of one completed bundle |
| `max_windows` | 10,000 | Maximum emitted window count |

A conservative preflight allowance includes PCM payloads, headers, and manifest
space. It can reject a near-limit bundle that might fit; actual written size is
also checked. Reuse temporarily needs space for another complete bundle. These
limits are per save, not a quota on accumulated output. The saver retains window
records and only one emitted window at a time; it writes audio in 65,536-frame
blocks. Existing decoded-array limits still apply. Persistence limits do not
change the audio recipe or output key and are passed separately to the API.

Output must remain under the project root, separate from raw data. Output
symlinks and Windows directory junctions are rejected. The caller must keep the
signal unchanged while saving, and the output directory should be controlled by
the application; these checks are not a defense against a hostile concurrent
filesystem writer.

The original source is neither copied nor deleted. Its
`raw_audio_retention_allowed` value remains unchanged in the manifest. Prepared
output uses the existing acoustic-processing permission contract; saving does
not grant new permission. Entry, window-generation, and final-boundary checks
enforce the supplied consent snapshot, including expiry. There is no live consent
registry lookup or automatic retention cleanup here; later retention work remains
separate. Production callers should omit the test-only `now=` override.

## API and verification

```python
from audio_sentinel.persistence import save_prepared_audio

# prepared = prepare_signal(loaded, settings.audio)
saved = save_prepared_audio(
    prepared, settings.paths, source_dataset="your-documented-dataset",
    policy=settings.persistence,
)
print(saved.manifest_path)
file_backed_audio = saved.audio  # PreprocessedAudio for later model stages
```

`PersistedAudio` returns the directory, manifest path, validated manifest, reuse
flag, and a `PreprocessedAudio` property. Operational errors carry
`AudioPersistenceError.code`: `invalid_output_path`, `too_many_windows`,
`output_too_large`, `output_conflict`, `verification_failed`, `write_failed`, or
`cleanup_failed`. Non-finite samples discovered during encoding use
`invalid_samples`. Existing permission, array, and model validation errors remain
their original exception types.

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/smoke_test_persistence.py
.\scripts\verify_project.ps1
```

The smoke test generates a temporary stereo tone, loads and prepares it, saves and
reloads one full clip plus five windows, and verifies a repeated save. It checks
the original file is unchanged, then removes its temporary files. No setup,
microphone recording, or dataset download is required. Audio output stays under
the repository's existing gitignored data directories.

## In plain language

Until now, the prepared audio existed only as numbers in memory. Persistence saves
those numbers as playable WAV files. The manifest is their catalog: it tells later
code where each file is, how it was made, and which part of the recording a window
covers. Checking everything before publishing the folder keeps later stages from
accidentally using a half-written result.
