# Offline recorded-clip evaluator

A8.1 adds one orchestration boundary for evaluating a single authorized local
recording. `audio_sentinel.evaluator.OfflineClipEvaluator` connects the existing
preparation, acoustic, speech, language, risk, agreement, and final-verification
stages without copying their decision logic.

The evaluator is offline. It creates no network request and has no alert publisher.
A Phase 7 `alert` result remains a local candidate with `review_required=true`; it
does not authorize or send a notification.

## Public API

Construction requires:

- the shared `AudioSentinelSettings` and a source-dataset label;
- an already loaded and verified YAMNet model;
- an explicit `AcousticAggregationSettings` threshold policy;
- both verified Silero VAD and Faster-Whisper models when clips may authorize
  speech processing; and
- optional explicitly trusted language, risk, and agreement rule sets. The
  checked-in built-in rules remain the default.

```python
from datetime import UTC, datetime

from audio_sentinel.acoustic_aggregation import AcousticAggregationSettings
from audio_sentinel.acoustic_loader import load_yamnet
from audio_sentinel.config import load_settings
from audio_sentinel.evaluator import (
    OfflineClipEvaluator,
    OfflineEvaluatorModels,
    OfflineEvaluatorPolicies,
)
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.transcription import load_transcription_model
from audio_sentinel.vad import load_silero_vad

settings = load_settings()
models = OfflineEvaluatorModels(
    acoustic=load_yamnet(settings.paths),
    vad=load_silero_vad(settings.paths),
    transcription=load_transcription_model(settings.paths),
)
policies = OfflineEvaluatorPolicies(
    # This value must be selected and reviewed for the intended evaluation.
    # A8.1 does not claim that an acoustic threshold is empirically calibrated.
    acoustic_aggregation=AcousticAggregationSettings.uniform(0.5),
)
evaluator = OfflineClipEvaluator(
    settings=settings,
    source_dataset="authorized-local-recordings",
    models=models,
    policies=policies,
)
result = evaluator.evaluate(
    InputAudio(clip_id="clip-001", audio_path=path, consent=consent),
)
```

Production calls should omit `now`, causing every existing stage to check the live
clock and consent expiry. A fixed timezone-aware `now` is supported for repeatable
tests and controlled validation only.

## Evaluation flow

For every permitted clip, the evaluator:

1. checks that the effective preparation recipe produces mono 16 kHz audio;
2. prepares and persists the recording and deterministic window inventory;
3. runs verified YAMNet inference over prepared waveform windows;
4. applies the caller's explicit acoustic thresholds and overlap aggregation;
5. builds, saves, reloads, and source-checks the acoustic evidence bundle;
6. follows the consent-specific branch described below;
7. integrates privacy-minimized risk inputs and applies the built-in or explicitly
   trusted Phase 6 scoring rules;
8. evaluates Phase 7 evidence agreement; and
9. asks the final verification service for exactly one no-action, log, review, or
   local-alert-candidate outcome.

For `acoustic_and_speech` consent, step 6 runs verified VAD segmentation,
accepted-only offline transcription, and versioned language analysis. Both speech
models must be present before preparation starts. For `acoustic_only` consent, the
evaluator never calls VAD, transcription, or language analysis. The risk input
contract records speech and language as `not_permitted`, preventing an acoustic
signal alone from becoming a local alert candidate.

Denied, withdrawn, expired, unauthorized-device, and no-scope records still fail
at the existing preparation consent boundary. The evaluator does not weaken or
reinterpret any consent rule.

## Result boundary

`OfflineEvaluationResult` retains the typed result from every completed stage:

- prepared audio and manifest;
- raw acoustic inference and aggregation diagnostics;
- saved acoustic evidence;
- optional speech segmentation, transcription, and language results;
- integrated risk inputs and the Phase 6 assessment;
- the B7.1 agreement evaluation; and
- the A7.2 final decision.

`to_summary()` is a small privacy-minimized diagnostic view. It includes artifact
identities, counts, score, severity, outcome, review state, and alert-candidate
state. It excludes transcript text, matched phrases, raw audio, tensors, and local
paths. The summary explicitly records `notification_sent=false`.

The result is not the versioned final JSON report. B8.1 owns that contract and its
serialization rules.

## Failure behavior

Evaluator-level preflight failures use `OfflineEvaluatorError` with stable codes:

| Code | Meaning |
| --- | --- |
| `invalid_clip` | The caller did not provide one `InputAudio` record. |
| `incompatible_audio_settings` | The recipe is not mono at 16 kHz and cannot feed the pinned models. |
| `speech_models_required` | Speech is authorized, but the local VAD/transcription pair is absent. |
| `invalid_prepared_output` | A preparation result escaped the configured interim directory. |

Once a stage starts, its original typed exception and stable code propagate
unchanged. This preserves the exact failure boundary for later CLI, API, and audit
handling. Existing completed preparation or acoustic-evidence bundles remain
available for a safe retry; A8.1 does not claim a filesystem-wide transaction.

## Verification

Run the focused evaluator suite:

```powershell
python -m pytest tests/test_evaluator.py -q
```

The tests cover full speech-authorized evaluation to a clean local alert candidate,
acoustic-only consent, no-speech completion, missing-model preflight, incompatible
audio recipes, stage-error preservation, deterministic retry and output reuse,
privacy-minimized summaries, immutability, and safe invalid-input handling. All
models used by this focused suite are deterministic local test doubles that satisfy
the real model metadata contracts; no model download or network access is needed.

