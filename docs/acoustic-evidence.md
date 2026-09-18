# A3.3 — Timestamped acoustic-evidence JSON

## What this task adds

`audio_sentinel.acoustic_evidence` turns the paired A3.2 inference snapshot and
B3.2 aggregation result into a durable JSON document. The v1 contract records:

- a UTC creation timestamp and deterministic evidence ID;
- the preparation manifest, raw-audio, prepared-window, model-artifact, class-map,
  runtime, and label-mapping identities;
- the complete input window and patch inventory, including window hashes;
- explicit aggregation thresholds and merge settings;
- sample-exact and second-based event/contribution bounds; and
- each contribution's window, patch index, mapped score, and winning YAMNet class.

The schema is checked in at `docs/schemas/v1/acoustic-evidence.schema.json`.
`build_acoustic_evidence` recomputes B3.2 aggregation from the supplied inference
snapshot before accepting the pair, so results from different runs cannot be
silently combined.

## Meaning and boundaries

The document type is `acoustic_event_candidates`. It is model evidence, not an
incident report. `peak_score` and contribution `score` are uncalibrated YAMNet
sigmoid outputs, not probabilities that an event occurred. The document contains
no `risk_level`, incident decision, or cross-modal conclusion. A3.4 still owns
threshold evaluation; current thresholds must be supplied explicitly.

Integer sample offsets remain authoritative. Second values are derived as
`sample / 16000` and validated on every parse. End offsets are exclusive. YAMNet
patch spans remain clipped to real prepared audio, so neither preparation padding
nor model padding can extend evidence beyond a recording.

## Identity and safe persistence

The evidence ID is SHA-256 over canonical semantic content: source/model
provenance, settings, windows, and events. The creation timestamp is intentionally
excluded, allowing the same computation to reuse one output rather than creating
duplicates on every run. Outputs live at:

```text
data/processed/acoustic-evidence/<evidence-id>/evidence.json
```

Writes use a private staging directory, file flush/fsync, schema readback, a final
source recheck, and atomic directory publication. Existing outputs are never
overwritten: identical content is reused while corrupt or conflicting content
fails closed. Paths must stay inside managed directories and cannot traverse links
or junctions. Output/source overlap is rejected.

Reloading validates the schema and evidence ID, exact bundle path/inventory,
resource limits, current processing consent, preparation-manifest hash, raw-audio
identity, complete window inventory, and every prepared-window hash. No raw audio,
model score matrix, embedding, or spectrogram is copied into the JSON.

## Usage

```python
from audio_sentinel.acoustic_evidence import save_acoustic_evidence

saved = save_acoustic_evidence(paths, inference_result, aggregation_result)
print(saved.evidence_path)
```

The normal unit suite uses a fake callable and needs no TensorFlow install. The
real structural smoke test uses the existing isolated YAMNet environment and a
generated 1.6-second tone:

```powershell
.\.venv\yamnet\Scripts\python.exe .\scripts\smoke_test_acoustic_evidence.py
```

Its zero threshold exists only to exercise all mapped labels and persistence; it
makes no accuracy or deployment-threshold claim.

In plain language: this task creates an auditable receipt for what the sound model
observed, when it observed it, and exactly which files, model, mapping, and rules
produced that evidence—without declaring that an incident happened.
