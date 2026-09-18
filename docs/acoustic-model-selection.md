# A3.1 — Pretrained acoustic model and label mapping

## Decision

Use **YAMNet, TensorFlow Hub revision 1**, as the first offline acoustic baseline.
The model uses MobileNet v1 and predicts 521 AudioSet classes. Its existing patch
outputs suit our planned timestamped evidence pipeline. This is an engineering
choice; accuracy and latency on this project's data have not been measured.
[Official model description](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet).

We also considered **PANNs Cnn14_16k**, an AudioSet model with an official 16 kHz
checkpoint and a PyTorch implementation. It fits the already installed PyTorch
runtime, but its tagging output needs additional temporal orchestration; PANNs
also offers separate sound-event-detection variants. Keep it as a comparison
candidate if YAMNet fails our later evaluation. Neither model accepts arbitrary
Log-Mel recipes just because dimensions match.
[Author's implementation and checkpoints](https://github.com/qiuqiangkong/audioset_tagging_cnn).

The project interpreter is Python 3.13.5. B3.1 now provides an isolated
TensorFlow 2.21.0 environment, loads the versioned SavedModel locally, and verifies
actual model metadata. The source revision below pins the vocabulary/reference
code. The downloaded SavedModel is separately pinned by the complete payload
digest documented in [the loader guide](acoustic-model-loading.md). No inference
or performance claim is part of A3.1 or B3.1.

## Executable specification

`src/audio_sentinel/acoustic_model.py` provides the frozen `YAMNET` specification,
`LABEL_MAPPING_VERSION`, immutable mapping/deferral tables, and offline
`validate_label_mapping()`. The v1 model is loaded from Google's official Kaggle
mirror: `https://www.kaggle.com/models/google/yamnet/tensorFlow2/yamnet/1`.

The bundled official CSV preserves all 521 ordered indexes, machine identifiers,
and display names. Its upstream revision is
`d598fb8b23d9cd2fb26b5789b8242de3f494aca7`; SHA-256 is
`cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2`.
License and provenance accompany it. Validation rejects changed bytes and mapping
drift. The loader must compare its loaded model's vocabulary against this reference
before applying mapping version 1.0.

## How completed preparation and features fit

Phase 1 preparation and Phase 2 Log-Mel extraction/storage are complete. The
prepared 16 kHz mono WAV windows are the input to the chosen detector. Decode to
finite float32 samples in a one-dimensional array, range [-1, 1]; retain consent
checks and verified source linkage. Alternate-rate prepared windows require an
explicit compatible preparation recipe before inference.
[Official waveform usage](https://www.tensorflow.org/hub/tutorials/yamnet).

YAMNet performs its own frontend: magnitude STFT with 400-sample windows and
160-sample hops, FFT 512, 64 mel bands spanning 125–7500 Hz, then natural
`log(mel + 0.001)`. Each patch has 96 frames and hops 0.48 seconds. Our stored
features use squared magnitudes, Slaney filters spanning 0–8000 Hz, power dB and
peak-relative clipping. They cannot be passed into YAMNet, even after transposing.
Keep them for inspection and future compatible models; use the model-owned
frontend for this baseline.
[Reference frontend](https://github.com/tensorflow/models/blob/d598fb8b23d9cd2fb26b5789b8242de3f494aca7/research/audioset/yamnet/features.py),
[reference parameters](https://github.com/tensorflow/models/blob/d598fb8b23d9cd2fb26b5789b8242de3f494aca7/research/audioset/yamnet/params.py).

The 96-frame patch spans 15,600 waveform samples (0.975 seconds), including its
final STFT window, and advances 7,680 samples. The model pads short/tail inputs.
A3.2/B3.2 must retain the prepared-window offset and real unpadded duration when
forming timestamps, clip evidence to real audio, and merge overlap without
counting the same sound twice. Patch scores are coarse evidence, not exact onset
or offset measurements. No temporal aggregation is implemented here.

## Mapping version 1.0

These are candidate acoustic associations. Exact index/MID/name triples live in
the executable mapping and are checked against the
[official class vocabulary](https://github.com/tensorflow/models/blob/d598fb8b23d9cd2fb26b5789b8242de3f494aca7/research/audioset/yamnet/yamnet_class_map.csv).

| Project label | Model indexes | Interpretation |
| --- | --- | --- |
| speech_present | 0, 1, 2, 3, 5, 12 | Speech/conversation/narration, synthetic speech, whispering; does not establish a person or speech meaning. |
| siren | 317, 318, 319, 390, 391 | Emergency-vehicle, general and civil-defense siren candidates. |
| smoke_alarm | 393, 394 | Smoke detector and fire alarm, consistent with the v1 project definition. |
| glass_break | 437 | Shatter is a proxy; the score cannot prove the material was glass. |
| gunshot | 421, 422 | Gunfire and machine-gun candidates, including possible recordings/lookalikes. |
| explosion | 420 | Broad explosion candidate; may overlap gunfire or fireworks. |

All seven other labels have explicit reasons in `DEFERRED_LABELS`:
`ambient`, `no_speech`, `non_threatening_speech`, `crowd_panic`, `distress_speech`,
`threatening_speech`, and `weapon_reference`. Unmatched classes produce no mapped
label; they do not prove safety. Crowd, screaming or crying alone do not establish
panic or spoken distress. A speech score cannot authorize transcription; speech
processing still requires its own consent scope.

For later inference, preserve raw per-class sigmoid scores and their provenance.
If a single label score is needed for one patch, use the maximum of its listed
classes; do not sum parent/child scores or normalize across labels. These scores
are uncalibrated and must not be described as incident probabilities. No event
threshold is approved by A3.1: evaluate thresholds in A3.4. Mapping alone must not
emit an `EventAnnotation` or calculate risk. Cross-label correlations (especially
explosion/gunfire), temporal aggregation and final risk belong to later tasks.

Alarm clocks, generic alarms, car alarms, glass clinks, cap guns and fireworks
are not directly mapped to the target labels. This does not prevent a model from
also scoring a broad target class highly; retain those confounders for evaluation.

## Verification and next task

`tests/test_acoustic_model.py` checks complete taxonomy coverage, reference
integrity, mapping drift, excluded lookalikes, duplicate associations, immutability,
and use without network or ML runtime imports. B3.1 adds artifact/runtime/signature
checks in `tests/test_acoustic_loader.py`; A3.2 adds verified raw-score orchestration,
B3.2 adds overlap-safe candidate aggregation, A3.3 persists auditable evidence,
and B3.3 completes the loader/aggregation boundary-test matrix. These still do not
verify detection quality. Next: **A3.4 — Evaluate acoustic detection against labeled samples**.

In plain language: we selected the sound recognizer, wrote its translation
dictionary, can load an exact verified local copy, run it on prepared waveforms,
and combine repeated window predictions without score inflation. The next task
serializes timestamped evidence; later work tests accuracy.
