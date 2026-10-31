# Audio upload, risk analysis and explainability

## What this feature adds

Before this feature, the pipeline could be run only on a recording already placed
under `data/raw/`, through the CLI or the loopback `POST /api/v1/evaluations`
endpoint. The browser UI (`/demo`) showed Log-Mel features for generated signals and
was not connected to inference.

This feature adds a browser workflow and two loopback-only endpoints:

| Route | Purpose |
| --- | --- |
| `GET /analyze` | Upload page: file picker, preview player, size and duration, scope, consent, Analyze Audio button, results, and "Is this audio dangerous?" |
| `GET /api/v1/uploads/limits` | Upload limits used by the page |
| `POST /api/v1/uploads/analyze` | Multipart upload → existing pipeline → score + evidence + explanation |
| `GET /api/v1/uploads/{upload_id}/explanation` | Re-answers "Is this audio dangerous?" from the saved, integrity-verified final report |

No model was trained or added, and no external or paid API is called. The upload
endpoint calls the same `evaluation_service` used by the CLI and the A8.3 API, so it
uses the same YAMNet acoustic inference, Silero VAD, Whisper tiny.en transcription,
language rules, risk scoring, consensus decision, final report and local audit.

## Running it

```powershell
pip install -e .[dev,speech]          # python-multipart is now a core dependency
# models: scripts/setup_yamnet.ps1, scripts/install_silero_vad.py, scripts/install_whisper_tiny_en.py
python -m uvicorn audio_sentinel.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/analyze`. If the local models are missing, the page shows
the loader's own error (for example `503 model_not_found`). It never shows a
substitute result.

## Processing flow

1. **Validate (before touching disk).** The file must be non-empty and at most 10 MB.
   The extension must be `.wav` or `.flac`, and the file must start with a RIFF/WAVE
   or `fLaC` signature. The HTTP layer also stops reading as soon as the 10 MB limit
   is passed.
2. **Stage safely.** Bytes are written with `O_EXCL` to
   `data/raw/uploads/<random-uuid>.<ext>`. The client filename is used only as a
   sanitized display label, never as a path.
3. **Probe.** `soundfile` must decode the container. Its detected format must match
   the extension, and the duration must be between 0.1 and 30 s. The RMS level is
   compared with the pipeline's existing `silence_floor_dbfs` (−60 dBFS) so that
   silent input is flagged.
4. **Run the existing pipeline** through `run_evaluation_with_report`. That function
   was split out of `run_evaluation`, which keeps the same behaviour. Upload consent
   is created only after the user ticks the authorization checkbox. It records
   `raw_audio_retention_allowed=False`. The acoustic threshold is the documented
   experimental value 0.5 used in the CLI and MVP-demo examples.
5. **Explain** with `risk_explanation.explain_report` (see below).
6. **Clean up, always.** A `finally` block deletes the staged upload and the prepared
   (resampled and windowed) WAV copy. What remains is the audio-free evidence JSON,
   final report, audit record, and a small upload receipt that links the upload id to
   its report. No transcript text is returned or persisted.

Model runs share the existing single-evaluation lock, so a concurrent request gets
`409 evaluation_busy`.

## How the explanation is produced

`risk_explanation.py` is deterministic and uses only the verified final report:

* **Score, level and outcome** are copied from the report and never recomputed for
  display. The scale is the project's 0–100 range. Levels use the existing severity
  bands: none 0, low 1–24, medium 25–49, high 50–74, critical 75–100.
* **Per-branch points** come from re-running the *existing* `_acoustic_score`,
  `_speech_score` and `_language_score` functions with the pinned built-in rule set
  (`audio-sentinel-risk-v1`). The breakdown is shown only if it reproduces the
  recorded score exactly. Otherwise `breakdown.reproduced=false` and the points are
  withheld.
* **Evidence** lists each acoustic event (label, time span, model peak score, rule
  points, high-confidence flag) and each language finding (category and rule
  reasons, no text).
* **Verdict**:

  | Condition | Verdict |
  | --- | --- |
  | silent input or missing evidence branch | `inconclusive` |
  | outcome `alert` | `elevated_risk_alert_candidate` |
  | outcome `review` | `possible_risk_review_recommended` |
  | outcome `log` | `low_indicators` |
  | outcome `no_action` | `no_indicators_detected` |

  The verdict maps the consensus outcome that already exists; no new thresholds are
  introduced. No answer ever claims an emergency is confirmed. Even an alert verdict
  states that it is "not a confirmed emergency and no notification was sent".
* **No invented confidence.** `model_confidence` is always `null`, because the
  pipeline has no calibrated overall confidence. Per-event peak scores are shown as
  model outputs.
* **Limitations** are always listed. They are extended when speech was not permitted,
  when the input was silent, or when transcription was used.

## Example response (abridged)

Produced by the automated test harness, which uses the real pipeline with a fake
acoustic model reporting `glass_break` at 0.88:

```json
{
  "upload_id": "9bdd14dd823941528b442fbf8e64d19e",
  "input": {"display_name": "sample_tone.wav", "format": "WAV", "sample_rate_hz": 16000,
            "channels": 1, "duration_seconds": 3.0, "rms_dbfs": -16.99, "silent": false},
  "evaluation": {"processing_scope": "acoustic_only", "outcome": "review",
                 "risk_score": 35.0, "risk_severity": "medium", "review_required": true,
                 "alert_candidate": false, "notification_delivery": "not_sent"},
  "explanation": {
    "verdict": "possible_risk_review_recommended",
    "answer": "Possible risk. The recording scored 35/100 (medium) based on: Glass breaking. Human review is recommended before drawing any conclusion.",
    "score_range": [0, 100], "model_confidence": null,
    "acoustic_events": [{"label": "glass_break", "start_seconds": 0.0, "end_seconds": 3.0,
                         "peak_score": 0.88, "rule_points": 25.0, "high_confidence": true}],
    "contributions": [{"branch": "acoustic", "points": 35.0, "max_points": 60.0,
                       "summary": "Strongest weighted event: Glass breaking (25 points, peak score 0.88) plus the high-confidence bonus."}],
    "breakdown": {"reproduced": true, "total_before_cap": 35.0, "rule_set_id": "audio-sentinel-risk-v1"}
  },
  "upload_retained": false
}
```

## Errors

All errors use the existing `{"code", "error"}` body.

| Situation | Status | Code |
| --- | --- | --- |
| Empty file | 400 | `empty_upload` |
| Not `.wav`/`.flac` | 415 | `unsupported_format` |
| Corrupted, undecodable, or extension/content mismatch | 400 | `invalid_audio` |
| Over 10 MB | 413 | `upload_too_large` |
| Over 30 s / under 0.1 s | 413 / 400 | `audio_too_long` / `audio_too_short` |
| Consent not confirmed | 403 | `consent_required` |
| Non-loopback client | 403 | `local_access_required` |
| Another run in progress | 409 | `evaluation_busy` |
| Model missing / runtime mismatch | 503 | existing loader codes |
| Model output invalid | 400 | existing stage codes (e.g. `invalid_output`) |
| Anything unexpected | 500 | `unexpected_error` (details logged, not returned) |
| Saved report fails verification | 409 | `report_unavailable` (no explanation given) |

## Bug fixed along the way

`risk_scoring.score_risk` capped the total with `min(rules.total_max_score, …)`. When
the cap applied, that returned the **integer** `100`. The assessment id was hashed
with `100`, but the validated document stores `100.0`. As a result, every recording
whose evidence summed past 100 failed final-report validation with "Evaluation
artifacts could not form a valid final report". Those are the highest-risk
recordings. The score is now always a `float`. Uncapped scores were already floats,
so no existing assessment id changes. The test
`test_speech_scope_capped_score_produces_alert_candidate_without_transcript` is the
regression test.

## Tests

```powershell
python -m pytest tests/test_upload_analysis.py -q
```

The 26 tests cover valid WAV, FLAC, stereo and 44.1 kHz input; the capped-score
regression; silent input; repeated uploads; empty, MP3, text, corrupted, mismatched,
oversized and over-long files; consent; scope validation; loopback rejection;
model-output failure with retry; redaction of unexpected errors; the busy lock;
unknown or tampered explanation ids; page and asset serving; and filename
sanitization. They run the real pipeline but use the existing fake YAMNet, VAD and
Whisper boundaries from `tests/test_evaluator.py`. They are **not** real-model
end-to-end tests. A real-model run requires the pinned local models.

Note: the checked-in JSON-schema snapshot tests pass with pydantic 2.9.x. Newer
pydantic releases serialize `Literal`/numeric bounds differently, and 18 snapshot
tests then fail independently of this feature.

## Limitations

* WAV and FLAC only (the formats the existing loader accepts). MP3 would need an
  extra decoder dependency.
* The limits (10 MB, 30 s) are upload-boundary limits chosen for an interactive demo.
  The underlying pipeline still allows up to 600 s.
* The explanation is rule-based text over model outputs, not a conversational model.
  It is only as good as the upstream detections, and the threshold and scoring rules
  are uncalibrated (see `docs/authorized-use-and-limitations.md`).
* Loopback-only prototype. It is not reviewed for network deployment (see the A10.4
  release blockers).
