# Consent, authorized use, human review, and model limitations

## Purpose and status

Audio Sentinel is an offline research prototype for evaluating already recorded,
locally stored audio. It creates evidence, risk, consensus, report, and audit
records for an authorized human reviewer. It is not a live monitoring service, an
emergency-dispatch system, a medical or legal service, a surveillance product, or
an autonomous enforcement system.

No model result proves that an incident occurred or that a person intended harm.
Every `alert` is only a pending local candidate. The system has no recipient,
transport, publisher, or emergency-service integration and permanently records
`notification_delivery=not_sent` and `alert_delivery_authorized=false`.

This document describes the implemented technical boundary. It is not legal
advice and does not create permission to record, retain, analyze, disclose, or act
on audio. The project owner must obtain the authority required by the applicable
law, contract, institutional policy, and dataset license.

## Consent and processing boundary

The application stores an opaque reference to a consent record, not the consent
text or a person's identity. Authorization must be established outside this
application before a request is created.

| Consent state and scope | Permitted processing | Required behavior |
| --- | --- | --- |
| Active `granted` + `acoustic_only` + authorized device | Local audio preparation, YAMNet acoustic inference, risk, consensus, reporting, and audit | Do not run VAD, transcription, or language analysis. Speech and language are recorded as `not_permitted`. This scope cannot satisfy the two independent alert-support gates. |
| Active `granted` + `acoustic_and_speech` + authorized device | The acoustic path plus VAD, English transcription, and rule-based language analysis | Content analysis must have been specifically authorized. A clean critical result may create a pending local alert candidate, but human review remains required. |
| `denied`, `withdrawn`, expired, not-yet-active, `none`, or unauthorized device | Nothing | Reject the request before audio processing. Do not reinterpret a rejected request as acoustic-only permission. |

An active grant requires timezone-aware timestamps. `granted_at` must have taken
effect, and an optional `expires_at` must still be in the future at every checked
processing boundary. Long-running stages recheck consent so an expired grant does
not authorize a result merely because processing started earlier.

Raw-audio retention is a separate permission and defaults to false. It does not
expand the processing scope. The Phase 8 retention command does not delete raw
recordings or earlier derived artifacts; its exact scope is documented in
[`retention-and-deletion.md`](retention-and-deletion.md).

### Operator authorization checklist

Before every evaluation, the operator must confirm all of the following outside
the application:

1. The recording itself was lawfully and institutionally authorized, including
   the device, place, people, and intended purpose.
2. The referenced consent is authentic, current, not withdrawn, and broad enough
   for the requested `acoustic_only` or `acoustic_and_speech` processing.
3. The request uses the least intrusive scope needed. Speech content must not be
   analyzed under acoustic-only permission.
4. The audio remains below the approved local `data/raw` root, and opaque clip and
   consent identifiers contain no names or other direct identifiers.
5. The source dataset or recording license allows this use. Attribution,
   noncommercial, redistribution, and deletion obligations remain in force.
6. The selected threshold is an intentional evaluation setting, not a presumed
   probability cutoff or a production-approved value.
7. Local access controls, encryption, backups, retention settings, and the human
   review process are appropriate for the recording's sensitivity.

If any item is uncertain, do not submit the recording. Resolve the authority and
scope first; the API accepting a structurally valid request is not proof of legal
or organizational authorization.

## Authorized and prohibited uses

The supported use is a controlled, offline evaluation or demonstration using an
authorized recording and a trained human reviewer. It may be used to reproduce
tests, inspect model evidence, compare predeclared evaluation settings, and study
known limitations.

Do not use this prototype:

- for covert recording, indiscriminate surveillance, or continuous monitoring;
- as the sole basis for contacting emergency services, police, security, or a
  person alleged to be involved;
- to punish, accuse, rank, profile, or make employment, education, housing,
  insurance, healthcare, immigration, or access-control decisions about a person;
- to infer speaker identity, demographics, mental state, intent, guilt, or the
  truth of a statement;
- for medical diagnosis, legal conclusions, weapons detection certification, or
  a claim that a location is safe;
- as an internet-facing or hostile multi-user API, a live microphone pipeline, or
  a production alerting service;
- to bypass dataset licenses or the rights of recording participants; or
- to describe a local `alert` record as a notification, dispatch, delivery, or
  confirmed real-world incident.

The loopback API is a convenience boundary, not authentication or authorization
for an untrusted local machine. Do not bind it to `0.0.0.0`; use operating-system
access controls and a dedicated trusted environment.

## Meaning of outcomes

| Outcome | Correct interpretation | Human action |
| --- | --- | --- |
| `no_action` | No configured evidence crossed the current policy. It does not prove that the recording or situation is safe. | Retain or close under the approved evaluation protocol; investigate separately if other information suggests danger. |
| `log` | Low policy score without a review trigger. It is not an incident finding. | Review only when the study protocol or outside context requires it. |
| `review` | Score, uncertainty, missing evidence, ambiguity, or a branch conflict prevents automatic escalation. | A qualified reviewer must inspect the evidence and context before any real-world action. |
| `alert` | Every v1 software gate passed and a pending local candidate may be stored. It is not sent and does not confirm an incident. | Prompt human review is mandatory. Any external action is a separate human decision under the organization's procedure. |

False positives and false negatives are both expected. In particular, a
`no_action` result must never override direct observation, a person's request for
help, an alarm from another system, or an established emergency procedure.

## Human-review procedure

The current application records pending review but does not implement reviewer
authentication, assignment, disposition, dismissal, or notification delivery.
Those actions must remain in a separately controlled human workflow.

For a `review` or `alert` outcome, the reviewer should:

1. Verify the active consent, processing scope, authorized device, recording
   source, dataset license, and evaluation time. Stop if any authority is missing.
2. Reload the final report and audit through the provided inspection command so
   their identities, hashes, references, and canonical locations are checked.
   SHA-256 detects content changes; it is not a signature and does not prove who
   created or approved the record.
3. Inspect the recorded branch states and reasons. Distinguish `not_permitted`
   from `missing`, and do not treat either state as evidence of safety.
4. Examine acoustic candidates for timing, weak duration, overlapping classes,
   and known lookalikes. If listening to the source is necessary, do so only when
   the same consent and access policy permits it; review does not broaden consent.
5. For speech-authorized cases, check whether VAD found speech, whether the
   transcript was accepted, rejected, empty, or blocked for review, and whether
   language findings depend on quotation, negation, hypothetical wording, or
   insufficient context. Do not infer identity or intent from text.
6. Seek independent, authorized context. A corroborating human observation or
   trusted source is more important than repeatedly rerunning the same model.
7. Record an external disposition such as `confirmed`, `dismissed`, or
   `insufficient_evidence`, the reviewer, time, rationale, and supporting sources
   in the organization's approved system. These states are recommendations for
   the manual workflow; Audio Sentinel does not store or enforce them yet.
8. If a trained, authorized person decides that a real and immediate danger
   exists, follow the organization's local emergency procedure. Audio Sentinel
   cannot make that decision or contact anyone.
9. Apply the approved retention policy after review. Use a dry run first, protect
   pending records unless deletion is explicitly authorized, and handle raw data,
   backups, exports, and external copies through their separate controls.

## Model and policy limitations

| Component | What it supplies | Important limitations for a reviewer |
| --- | --- | --- |
| YAMNet acoustic model | Scores over 521 AudioSet classes, mapped to six v1 project labels | Scores are uncalibrated and are not incident probabilities. Patches are coarse in time. `glass_break` uses the broader `Shatter` class. Alarms, clinks, cap guns, fireworks, animals, wind, recordings, and other lookalikes can confuse the model. Seven project labels are deliberately deferred, and an unmatched class does not prove safety. |
| Silero VAD v6 | Local 32 ms speech-likeness scores | A score is not a calibrated probability that speech is present or correctly segmented. Noise, distance, overlapping sound, accents, and recording quality can cause misses or false activations. Speech presence says nothing about meaning or risk. |
| Faster-Whisper `tiny.en` | Local English transcript candidates | It is English-only. Its `derived_score` is not a calibrated word-correctness probability, and `language_confidence=1.0` records a fixed capability rather than detected language certainty. Names, rare words, noise, accents, code-switching, and urgent speech may be transcribed incorrectly. Only candidates at or above 0.80 flow automatically to language analysis; 0.50 to below 0.80 require review, below 0.50 are rejected, and empty results are not benign evidence. |
| English language rules | Deterministic matches with negation, quotation, hypothetical, and ambiguity safeguards | The finite rules and 74 synthetic fixtures cannot cover all wording, languages, dialects, sarcasm, context, or intent. A match is an indicator, not proof. Context suppression can also be wrong. |
| Risk scoring | A reproducible 0–100 policy score and review reasons | Weights, thresholds, caps, and severity bands are engineering policy values, not empirically calibrated incident likelihoods. A score is not a probability and must not be presented as one. |
| Consensus | Agreement and conflict checks across available branches | Acoustic support uses an explicit 0.85 policy boundary that is not an incident probability. Speech never supports risk by itself. Missing, conflicting, or ambiguous evidence goes to review; agreement can still be jointly wrong. Acoustic-only cases cannot become alerts. |

The pinned model artifacts and deterministic runtime controls improve
reproducibility and supply-chain integrity. They do not make the models accurate,
fair, complete, or suitable for a new population, language, microphone, room, or
deployment environment.

## Current empirical evidence

The A9.1 held-out run is the only checked-in end-to-end measurement. It used 32
balanced, acoustic-only clips from ESC-50 and UrbanSound8K at a predeclared `0.5`
threshold: 13 true positives, 2 false positives, 3 false negatives, and 14 true
negatives. Accuracy and balanced accuracy were 84.375%, precision 86.667%, recall
81.25%, specificity 87.5%, and F1 83.871%.

The two false positives were cow and wind clips scored as siren candidates. The
three false negatives were two glass-breaking clips and one siren clip. The run
was small, balanced, acoustic-only, previously used as an A3.4 holdout, and not
representative of deployment prevalence or environments. It did not test speech,
language, real emergencies, outcome correctness, fairness, or notification. It
does not approve a production threshold or accuracy claim. See
[`evaluation-findings.md`](evaluation-findings.md) for the exact denominators,
case analysis, provenance, and reproduction command.

## Privacy, security, and retention limitations

- Final reports and audit records omit raw audio, transcript text, matched phrases,
  direct identity, recipients, and absolute paths, but identifiers, timestamps,
  hashes, scores, and event evidence can still be sensitive.
- Intermediate speech evidence can contain transcript text. Access to project
  directories must therefore be restricted even when final reports are minimized.
- The local hashes provide integrity checks, not authorship, non-repudiation, or a
  signed chain of custody.
- The evaluator is local and single-process. Concurrent external writers, malware,
  administrators, and other users on the machine are outside its trust boundary.
- Retention cleanup is manual, disabled by default, and limited to verified Phase 8
  report and audit bundles. It is not secure erasure and does not cover raw audio,
  earlier derived artifacts, backups, exported copies, or external systems.
- Model setup may require a controlled download. Normal evaluation is offline, but
  offline execution alone does not satisfy consent, security, or retention duties.

## Release and demonstration gate

Before any demonstration or expanded deployment, document the authorized audience,
recording source, consent scope, model and dataset licenses, storage controls,
retention owner, human reviewers, escalation procedure, and known failure examples.
Run the complete verification suite and the exact A9.1 reproduction where the
ignored datasets and model are available. Re-evaluate with representative,
independently reviewed data before changing thresholds or making accuracy claims.

Do not move beyond a controlled offline demonstration until reviewer disposition,
access control, representative evaluation, incident response, and complete data
lifecycle controls have been designed and independently approved.

## Traceability

The main implementation contracts behind this guidance are:

- [`v1-data-contract.md`](v1-data-contract.md) for consent and data minimization;
- [`offline-evaluator.md`](offline-evaluator.md) and [`local-api.md`](local-api.md)
  for the local execution boundary;
- [`acoustic-model-selection.md`](acoustic-model-selection.md),
  [`vad-wrapper.md`](vad-wrapper.md), [`transcription-wrapper.md`](transcription-wrapper.md),
  and [`language-analysis.md`](language-analysis.md) for model behavior;
- [`speech-transcription.md`](speech-transcription.md) for confidence handling;
- [`risk-assessment.md`](risk-assessment.md), [`consensus-agreement.md`](consensus-agreement.md),
  and [`consensus-policy.md`](consensus-policy.md) for policy interpretation;
- [`final-report.md`](final-report.md) and [`local-alert-audit.md`](local-alert-audit.md)
  for report, alert, integrity, and delivery boundaries;
- [`evaluation-findings.md`](evaluation-findings.md) for measured evidence; and
- [`retention-and-deletion.md`](retention-and-deletion.md) for cleanup scope.

