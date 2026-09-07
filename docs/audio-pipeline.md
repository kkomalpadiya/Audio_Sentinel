# A1.4: One-call audio preparation service

`AudioPreparationService` in `src/audio_sentinel/pipeline.py` connects the completed
preparation components. One call accepts a local recording and returns its verified
saved bundle. Creating a service does not open files or create directories.

## What runs

1. The loader checks permission, source containment, WAV/FLAC validity, and input
   limits, then decodes the recording and records its source hash.
2. Preparation combines channels when configured, resamples, optionally reduces
   noise, and normalizes volume. Noise reduction remains disabled by default.
3. Persistence generates windows one at a time, writes the full clip and windows,
   verifies the WAVs and JSON manifest, then publishes the completed bundle.

Every stage uses the same audio recipe. Paths and persistence limits come from
the service's project settings. The source array is released after transformation;
the returned object contains paths and metadata rather than in-memory audio arrays.
The service keeps no last-result state, so it can prepare multiple clips with
the same configuration and dataset attribution.

## Usage

```python
from pathlib import Path

from audio_sentinel.config import load_settings
from audio_sentinel.contracts import ConsentRecord
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.pipeline import AudioPreparationService

settings = load_settings()
# This must be genuine permission evidence supplied for this recording.
consent = ConsentRecord.model_validate_json(Path("consent.json").read_text())
service = AudioPreparationService(settings, source_dataset="your-documented-dataset")
clip = InputAudio("clip-001", Path("recordings/example.wav"), consent)
result = service.prepare(clip)
print(result.manifest_path)
print(result.audio.audio_path)
print(result.reused)
```

The relative input above resolves under `settings.paths.raw_data`, normally
`data/raw/recordings/example.wav`. Absolute paths are also accepted when contained
in that raw directory. Dataset attribution is required; the service does not guess
it from a filename. It must be a nonblank name of at most 128 characters.

`prepare` returns the existing `PersistedAudio`: `directory`, `manifest_path`,
validated `manifest`, `reused`, and the `audio` property. Optional
`annotations=[...]` preserves existing event annotations in full-clip seconds.
The manifest validator checks their bounds and permission scope.

An optional `audio_settings=AudioSettings(...)` override applies to loading,
transformation, segmentation, and the saved manifest for that call. It leaves the
service's defaults unchanged. For example, restricting accepted formats to FLAC
also affects loading, rather than only the saved metadata. Change the top-level
settings before constructing the service to use different paths or output limits.

## Compatibility with later model stages

```python
from audio_sentinel.interfaces import AudioPreprocessor

preprocessor: AudioPreprocessor = service
audio = preprocessor.preprocess(clip, settings.audio)
# A later acoustic detector can consume this PreprocessedAudio.
```

`preprocess(clip, settings)` implements the existing protocol and uses the supplied
audio settings for the complete run. It saves the same full bundle, but returns
only `PreprocessedAudio`. Use `prepare` when you need the manifest, reuse flag,
annotations, or a test clock. The older `WindowPlan` export remains compatible;
actual pipeline window behavior is controlled by `AudioSettings.window_seconds`.

## Errors and repeat runs

Stage errors propagate with their existing types and codes: `AudioLoadError`,
`AudioTransformError`, `AudioPersistenceError`, and Pydantic validation errors.
The service stops at the first failure. Rejected input does not reach transformation
or saving, and a failed transform does not create prepared output. Persistence
retains its staged-write cleanup and conflict protection.

No success result is returned on failure. A repeated successful request verifies
and reuses identical output; differing content or metadata in the same destination
raises a conflict. See [the persistence guide](audio-persistence.md) for storage,
quantization, concurrency, and recovery limits.

Permission is checked at the existing processing boundaries. Production calls
omit `now=` so each stage reads the current clock; permission can expire during a
run. Tests can pass an aware timestamp to make boundary checks deterministic.
This remains validation of the supplied permission snapshot, not a live consent
registry lookup. The service neither copies nor deletes the raw source.

## Verification and manual steps

No installation, microphone access, or dataset download is needed for this task.
The focused service tests use generated recordings and temporary directories:

```powershell
python -m pytest tests/test_pipeline.py -q
.\scripts\verify_project.ps1
```

They check real saved output with noise reduction on/off, stereo/rate overrides,
manifest metadata, repeat saves, protocol compatibility, output limits, failures,
and permission expiry between stages. B1.4 adds [component-test coverage](preparation-component-tests.md).
A1.5 adds [broader integration tests and a runnable sample-clip check](preparation-integration-tests.md).
The `/project/status` response now reflects the completed preparation service.

Review, commit, and push this checkpoint when ready.

## In plain language

Earlier tasks built separate tools for opening, cleaning, slicing, and saving
audio. This service connects them in the right order. You give it a recording,
permission, and a dataset name; it returns the saved files and their catalog.
Later model code can use that result without managing the preparation steps itself.
