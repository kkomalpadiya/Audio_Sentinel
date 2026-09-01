# V1 Data Contract

This document records the foundation contract for Task A0.1. It applies to offline preparation manifests and annotations. Future scoring and alert contracts can add fields, but must not silently change these v1 label meanings.

## Event Labels

| Label | Meaning | Minimum risk |
| --- | --- | --- |
| `ambient` | Normal environmental sound without a target event | `none` |
| `no_speech` | No speech detected in the annotated interval | `none` |
| `speech_present` | Speech is present; no language judgement is implied | `none` |
| `non_threatening_speech` | Speech that has been reviewed as harmless | `none` |
| `siren` | Emergency-vehicle or warning siren | `low` |
| `smoke_alarm` | Smoke or fire alarm tone | `low` |
| `glass_break` | Audible glass breakage or impact-like sound | `medium` |
| `crowd_panic` | Crowd panic or sustained alarmed crowd noise | `medium` |
| `distress_speech` | Spoken distress, including a call for help | `medium` |
| `threatening_speech` | Credible spoken threat or weapon reference | `high` |
| `weapon_reference` | Speech referring to a weapon in a threatening context | `high` |
| `gunshot` | Gunshot-like impulse sound | `high` |
| `explosion` | Explosion- or blast-like sound | `high` |

`critical` is reserved for later fusion logic that has corroborating evidence. A single offline annotation must never lower the minimum risk associated with its label.

## Consent Boundaries

- A prepared clip requires granted consent, an authorized device, an explicit acoustic-processing scope, and a non-empty opaque consent reference.
- Language-level speech labels (`non_threatening_speech`, `distress_speech`, `threatening_speech`, and `weapon_reference`) require the `acoustic_and_speech` scope. `speech_present` and `no_speech` are voice-activity results and may be recorded with `acoustic_only` consent.
- Denied or withdrawn consent permits no processing, device authorization, or raw-audio retention.
- Raw audio retention defaults to `false`. The manifest contains no raw audio, transcript text, identity, location, or consent text.
- `audio_path` must be a relative path below the approved data root. Consumers must resolve the path against that configured root instead of accepting an arbitrary filesystem path.

## JSON Schemas

The public models are `ConsentRecord`, `EventAnnotation`, and `PreparedClipRecord` in `audio_sentinel.contracts`. Generate portable JSON Schema documents with:

```powershell
python -c "from pathlib import Path; from audio_sentinel.contracts import write_json_schemas; write_json_schemas(Path('docs/schemas/v1'))"
```

The generated documents use draft 2020-12-compatible Pydantic JSON Schema and have stable v1 schema identifiers.
