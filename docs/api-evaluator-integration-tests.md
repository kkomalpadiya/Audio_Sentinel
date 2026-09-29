# API and evaluator integration tests

## What A8.4 verifies

A8.4 exercises the complete local HTTP evaluation workflow instead of replacing
the shared application service with a mock. Each successful request passes through
request validation, model loading, audio preparation, acoustic inference and
aggregation, the consent-specific speech branch, language analysis, risk scoring,
evidence agreement, final verification, final-report persistence, and local audit
creation.

The tests replace only YAMNet, Silero VAD, and Faster-Whisper execution with
deterministic in-process models that satisfy the production metadata and tensor
contracts. No model download or network access is required.

## Covered workflows

The six integration tests cover:

1. An acoustic-only loopback HTTP request. It runs the real evaluator while proving
   that VAD and transcription are never loaded, speech and language remain
   `not_permitted`, and no local alert is created.
2. A speech-authorized request with aligned acoustic and language evidence. It
   produces one verified final report, one audit, and one pending local-only alert
   whose delivery remains unsent and unauthorized.
3. A speech-authorized request where VAD accepts no speech. Transcription is not
   invoked, language records `no_accepted_text`, and the workflow completes with a
   zero-risk no-action result and no alert.
4. A high acoustic threshold passed through HTTP into the real aggregation policy.
   A model score below that threshold produces no acoustic event and no action.
5. A missing recording. The API returns the stable safe `file_not_found` response,
   writes no final report or audit, and releases its evaluation lock.
6. A real acoustic-stage contract failure followed by a retry. The first request
   returns `invalid_output` without final artifacts or private paths; the corrected
   model then completes successfully using the same clip identity.

## Integrity and privacy checks

Successful response identities are reloaded through the public final-report and
alert-audit verifiers. The tests confirm agreement among the HTTP response, final
report, audit, and optional local alert. They also search responses and persisted
Phase 8 documents for the synthetic private transcript and absolute project root.

Failure tests use actual preparation and inference boundaries rather than raising a
synthetic exception directly in the route. They verify that incomplete evaluation
attempts do not create a final-report or alert-audit bundle and that every failure
releases the single-evaluation lock for a later request.

These tests do not change model selection, thresholds, scoring weights, consensus
rules, alert eligibility, or notification policy.

## Run the tests

```powershell
python -m pytest tests/test_api_evaluator_integration.py -q
```

## Next task

B9.1 will build the repeatable evaluation manifest and metric calculations used for
measured false-positive and false-negative analysis.

In plain language: A8.4 proves that the local API reaches the real offline evaluator
and produces self-checking records for each permitted branch. It also proves that a
genuine pipeline failure is contained, reported safely, and recoverable.
