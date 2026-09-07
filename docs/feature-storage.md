# A2.2: Feature storage and source-window linkage

`save_window_log_mel` reads one window listed in a saved preparation manifest,
generates B2.1 features, and publishes a verified two-file bundle:

```text
data/processed/log-mel/<identity>/
    features.npy
    metadata.json
```

The array is stored as little-endian float32 in C order without pickle. The JSON
uses A2.1's `LogMelFeatureMetadata`: recipe, shape, analysis padding, software
versions, creation time, feature-file hash, source-window record, window WAV hash,
preparation manifest path/hash, clip ID, and original raw-file hash.

## Use

```python
from audio_sentinel.config import load_settings
from audio_sentinel.feature_persistence import save_window_log_mel, load_log_mel

settings = load_settings()
# Obtain these from an existing AudioPreparationService result.
manifest_relative = prepared.manifest_path.relative_to(settings.paths.interim_data)
window_id = prepared.manifest.windows[0].window_id
saved = save_window_log_mel(
    settings.paths, manifest_relative, window_id, settings.log_mel,
)
loaded = load_log_mel(
    settings.paths, saved.metadata_path.relative_to(settings.paths.processed_data),
)
print(saved.reused, loaded.values.shape)
```

Manifest input paths are relative to `data/interim`. Reload metadata paths are
relative to `data/processed`. Use `Path.relative_to` as above for absolute paths.
Source and output paths reject traversal, symlinks, and Windows junctions.
Output directories must remain inside the project and separate from raw/interim
data. The public API generates from the exact decoded source bytes rather than
accepting an arbitrary array paired with a claimed source record.

## Verification and repeat saves

Saving validates the preparation manifest, current acoustic-processing consent,
window membership, actual WAV/PCM16 properties, and zero tail padding. Byte hashes
are calculated from the manifest and window that were actually read. Generation
uses those decoded samples. The original raw-file hash is inherited from the
preparation manifest; storage does not reopen raw audio or independently certify
that original recording. Local manifests are the authority for consent and source
history; this is not a signed provenance or external consent-registry system.

The directory identity includes source references/hashes, recipe, librosa version,
and implementation version. Creation time is excluded. A repeated request
regenerates features and verifies the existing bundle, preserving its original
creation time, bytes, and modification times. A changed source, recipe, or backend
version gets a distinct identity. A differing or corrupt bundle at the same
identity is rejected without overwriting it.

New output is written into a private staging directory, flushed and read back,
then published by directory rename. Failed writes remove only that staging
directory. A cooperating writer that publishes the same output first is handled
as verified reuse. This assumes application-owned local directories; it does not
promise race-free behavior against hostile processes replacing files concurrently.

`load_log_mel` verifies metadata/path identity, the two-file inventory, current
source hashes and consent, and NPY header/shape/dtype/order/size/hash/finite values.
NPY headers are checked before allocating the claimed shape; pickle arrays,
truncation, trailing data, and unexpected shapes are rejected. Source and consent
are rechecked after feature readback and before new publication. The optional
`now` argument is for reproducible tests; omit it for real processing.

## Limits and errors

Pass a `FeaturePersistenceSettings` policy to save or reload to customize:

| Limit | Default |
| --- | --- |
| Each JSON document | 16 MiB |
| Encoded source-window WAV | 256 MiB |
| Decoded window samples | 256 MiB |
| Complete feature bundle | 512 MiB |
| Estimated generator workspace | 256 MiB |

The A2.1 recipe's feature-payload limit also applies. These are separate operation
limits, not an aggregate process-memory cap. Encoded source snapshots, generated
arrays, and readback arrays can coexist. Generator library overhead is outside its
workspace estimate. Use shorter windows or smaller limits where necessary.

`FeaturePersistenceError.code` identifies path, source, budget, integrity,
publication, and I/O failures. Existing schema, consent, and generation errors
retain their own types. Malformed JSON/NPY may raise validation/format errors;
all stop the operation. No partial bundle is returned as a successful result.

## Verification and next step

Run `python -m pytest tests/test_feature_persistence.py -q`, then the standard
`scripts/verify_project.ps1`. Tests use generated recordings with actual Phase 1
output, round-trip features, and simulate corrupt output, changed sources,
consent expiry, failed writes/publication, memory limits, and Windows junctions.
No dataset downloads or manual recording are required.

This task provides storage for one existing prepared window. B2.2 expands
shape/range/determinism tests; A2.3 will connect preparation and extraction/storage
as a production service over the window inventory.

## In plain language

The frequency-over-time table is now saved with a record of exactly which audio
window and settings produced it. Reloading checks that both the table and its
source still match that record before returning the numbers.
