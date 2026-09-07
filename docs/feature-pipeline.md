# A2.3: Integrated audio-to-feature pipeline

`AudioFeatureService.prepare` now runs the completed preparation and Log-Mel
components in one call. It prepares one authorized recording, processes every
saved window in manifest order, verifies the final feature inventory, and returns
file-backed results. Acoustic inference belongs to Phase 3.

```python
from pathlib import Path
from audio_sentinel.config import load_settings
from audio_sentinel.feature_pipeline import AudioFeatureService
from audio_sentinel.interfaces import InputAudio

settings = load_settings(
    audio_config_path=Path("configs/preprocessing.example.json"),
    log_mel_config_path=Path("configs/log-mel.example.json"),
)
service = AudioFeatureService(settings, source_dataset="your-registered-dataset")
# consent is the actual validated ConsentRecord for this recording.
result = service.prepare(InputAudio("clip-001", Path("dataset/clip.wav"), consent))
print(result.to_summary())
```

The input path is relative to the configured raw-data directory, as in Phase 1.
The result's `prepared` field is the existing `PersistedAudio` bundle. `features`
is an ordered tuple of `SavedLogMel` bundles, one per preparation window, with
matching window records. `to_summary()` provides a JSON-serializable inventory of
paths, shapes, window IDs, counts, and reuse flags. The summary is returned to the
caller; no additional combined index file is written.

## Configuration

The service uses `settings.audio` and `settings.log_mel`. Optional per-call
`audio_settings` and `log_mel_settings` overrides apply throughout that invocation
without changing defaults. Recipes are revalidated before processing. The
integrated service requires mono conversion enabled and equal target/feature
sample rates; incompatible settings fail before preparation. The separate
`AudioPreparationService` retains its existing stereo-preserving options.

Pass a `FeaturePersistenceSettings` instance as `feature_policy` when constructing
the service to control A2.2's per-file, decode, output, and workspace limits. The
service also has `max_total_feature_bytes`, defaulting to 1 GiB for one recording's
feature inventory. It estimates output requirements from the actual windows
after preparation, before writing features. This includes payload and conservative
header/metadata allowances, so a budget very close to actual size may be rejected.
It also checks actual file totals as bundles finish. Preparation has its own
existing persistence policy and is outside the feature-byte total. These are
operation budgets, not a process-memory limit or a project-wide disk quota.

Default 1/5/10 second preparation windows of a 1.6-second sample produce five
feature bundles: three `(64, 97)`, one `(64, 497)`, and one `(64, 997)`. If an
explicit drop-tail configuration produces no windows, the service returns an
empty feature tuple and `feature_count: 0`; no feature directory is created.
This is successful preparation with no feature windows available for inference.

## Integrity and failure handling

Features are generated from saved PCM16 window files through A2.2. Each result
must match this run's preparation manifest, window record, and manifest byte
hash. After all windows finish, the service verifies each bundle through the
checked reload API, releasing each array before checking the next one. Only
paths and metadata remain in the returned result. Consent and the manifest hash
are checked again before return. Omit the optional `now` argument for live clock
checks; it exists for reproducible tests.

The service propagates preparation, schema, consent, generation, and storage
failures and returns no partial success result. `FeaturePipelineError.code`
identifies integration-specific configuration, inventory budget, and consistency
failures. Valid preparation and completed feature bundles remain on disk after
a later failure. Retrying verifies and reuses them, while creating missing bundles.
This is recoverable per-bundle publication, not an atomic transaction over the
whole recording. Corrupt bundles still fail verification and are not overwritten
automatically. As in A2.2, files are assumed to be application-owned local data;
verification is not an adversarial concurrent-filesystem guarantee.

`result.reused` is true only when preparation and every returned feature bundle
were reused. Per-bundle flags distinguish reused output from newly completed
windows on a retry. An empty inventory follows the preparation reuse flag.

## Run and verify

```powershell
python scripts/smoke_test_feature_pipeline.py
python -m pytest tests/test_feature_pipeline.py -q
.\scripts\verify_project.ps1
```

The standalone smoke script loads both JSON recipes, creates a synthetic stereo
recording, runs the integrated service twice, checks all five shapes and complete
reuse, verifies the raw source is unchanged, and removes temporary files.
The standard suite includes this script as a subprocess test.

Integration tests exercise WAV/FLAC inputs, 8/16 kHz recipes, denoising on/off,
metadata and annotations, per-call and JSON defaults, repeat runs, empty
inventories, limits, partial-failure retries, earlier-bundle corruption, changes
between windows, and consent expiry. No manual recording, download, or setup is
required for these generated-sample tests.

## In plain language

Give the service one recording and its permission record. It prepares the audio,
splits it into windows, saves the frequency-over-time tables, and returns the
verified file list. Phase 2 is complete. Next is A3.1: choose a pretrained acoustic
model and define how its labels map to the project's event labels.
