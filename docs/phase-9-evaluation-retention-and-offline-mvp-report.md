# Phase 9 completion report: Evaluation, Retention, Responsible Use, and Offline MVP

## 1. Executive summary

Phase 9 turned the completed offline processing pipeline into something that can be
measured, reviewed, cleaned up safely, and demonstrated responsibly.

Before Phase 9, Audio Sentinel could already process an authorized local recording,
run acoustic and optional speech analysis, calculate a policy risk score, apply
consensus rules, and save verified local reports. Phase 9 answered the operational
questions that remained:

- How can a fixed collection of recordings be evaluated repeatedly without
  accidentally changing the cases or hiding failed runs?
- What did the first end-to-end held-out measurement actually show?
- How can old reports be deleted without breaking referenced audit records or
  silently deleting the wrong directory?
- What consent, human-review, and model-limit rules must an operator follow?
- How can the whole MVP be demonstrated through one repeatable, non-delivering,
  non-deleting workflow?

In common terms, Phase 9 added a measuring procedure, a cleanup procedure, a rulebook,
and a demonstration procedure.

It did not turn the prototype into a live monitoring or emergency-response system.
It did not add a notification recipient, a delivery transport, autonomous dispatch,
speaker identification, or authority to act on a model result. A local `alert`
remains a pending record for human review, and every Phase 9 output preserves
`notification_delivery=not_sent` and `alert_delivery_authorized=false`.

Phase 9 completed five formal tasks:

| Task | Result in common terms |
| --- | --- |
| B9.1 | Defined a repeatable evaluation manifest, collection runner, and metric report. |
| A9.1 | Ran a pinned 32-clip held-out acoustic evaluation and documented every error. |
| B9.2 | Added dry-run-first retention and guarded deletion with an immutable receipt. |
| A9.2 | Added the consent, authorized-use, human-review, and model-limit rulebook. |
| A9.3 | Added a single offline MVP demonstration command with independent report verification. |

Two post-A9.3 fixes were also completed after real authorized speech demonstrations
exposed practical Faster-Whisper boundary behavior:

1. transcription timestamp validation was generalized using a bounded one-second
   zero tail for every verified speech segment; and
2. transcript policy v1.1 lowered automatic language-analysis acceptance from
   `0.80` to `0.60`, while retaining the `0.50` review boundary and leaving alert
   consensus unchanged.

At the end of the work, the complete project suite passed **1,865 tests**. The only
reported warnings were the same two existing Pydantic deprecation warnings from the
API OpenAPI test.

## 2. Phase 9 flow in common terms

The major Phase 9 relationships are:

```text
Pinned labeled collection
        |
        v
Evaluation manifest -- verifies cases, truth, scope, order, and identity
        |
        v
Existing Phase 8 evaluator -- runs the real pipeline once per case
        |
        v
Evaluation run -- records completed cases, failures, metrics, and limitations

Verified report/audit inventory
        |
        v
Retention planner -- verifies age, references, pending alerts, and delete limit
        |
        +--> dry run: report the plan and change nothing
        |
        `--> explicit apply: stage, write receipt, verify receipt, then remove

Authorized demonstration recording
        |
        v
Preflight --> public CLI evaluation --> independent report reload --> retention dry run
        |
        v
Privacy-minimized result with no delivery or external-action authority
```

This design deliberately reuses the same application boundary used by the normal
CLI and loopback API. Phase 9 did not create a separate “evaluation model” or a
special demonstration-only scoring path.

## 3. B9.1 — Repeatable evaluation manifests and metrics

### 3.1 Problem being solved

Running a folder of clips is easy; producing a measurement that can be reviewed and
repeated is harder. A trustworthy evaluation must say exactly which clips were
included, what counted as a positive result, which truth label was assigned, which
cases failed to execute, and which formula produced each number.

Without that contract, two runs can appear comparable even if the case list,
threshold, consent scope, positive-outcome mapping, or failure handling changed.

### 3.2 What was implemented

`src/audio_sentinel/evaluation_manifest.py` adds two strict, immutable contracts:

- an `evaluation_manifest`, which describes the intended collection before a model
  is run; and
- an `evaluation_run`, which records what happened when that manifest was executed.

Each manifest case contains:

- a unique opaque case ID;
- the existing Phase 8 `EvaluationRequest`;
- a required binary truth label;
- an optional exact expected consensus outcome; and
- non-sensitive category tags for later error analysis.

The manifest also declares which consensus outcomes count as a positive prediction.
That mapping is stored as data instead of being hidden inside metric code.

Loading a manifest enforces:

- a regular local file rather than a symlink;
- a 2 MiB size limit;
- optional caller-supplied SHA-256 verification;
- strict JSON fields and finite values;
- timezone-aware timestamps;
- unique case and clip identities;
- safe relative audio paths;
- valid authorization data;
- canonical positive-outcome ordering; and
- a recomputed semantic manifest identity.

### 3.3 Content-addressed identity algorithm

The manifest identity is calculated from canonical JSON:

1. remove fields that should not change the meaning, such as `created_at` and the
   provisional ID;
2. serialize with sorted keys, fixed separators, UTF-8, and no NaN values;
3. calculate SHA-256; and
4. use the first 24 hexadecimal characters in the public manifest ID.

This makes the case list and rules identity-bearing, while allowing the same
semantic manifest to be rebuilt at a different clock time.

The evaluation run uses the same general method. Its semantic identity includes
the manifest receipt, expected and observed labels, risk results, safe failure
codes, metrics, and permanent delivery state. It excludes the run time and the
per-execution report, audit, and alert receipt IDs.

Why exclude those local receipt IDs? They prove where one execution stored its
artifacts, but they do not change the measured prediction. Including them would
make an exact rerun look like a different experimental result.

### 3.4 Collection execution algorithm

The runner processes cases in manifest order:

```text
for each manifest case:
    call the existing run_evaluation service
    if it succeeds:
        record opaque IDs, expected/observed labels, risk result, and receipts
    if it fails:
        record a stable safe error code
        continue to the next case

calculate metrics over completed cases only
mark the run incomplete if any case failed
calculate a semantic run ID
```

One failed clip does not erase the other results. It also does not become a false
negative. The run exposes total, completed, and failed counts plus completion rate,
so operational reliability cannot disappear from the denominator.

Unknown exception messages are reduced to `unexpected_error`. Paths, consent IDs,
transcripts, and private exception details are not copied into the collection report.

### 3.5 Metric algorithm

For the manifest's declared positive outcomes:

- **true positive (TP):** expected positive and observed positive;
- **false positive (FP):** expected negative but observed positive;
- **false negative (FN):** expected positive but observed negative; and
- **true negative (TN):** expected negative and observed negative.

The report calculates:

```text
accuracy            = (TP + TN) / (TP + FP + FN + TN)
precision           = TP / (TP + FP)
recall               = TP / (TP + FN)
specificity          = TN / (TN + FP)
F1                   = 2TP / (2TP + FP + FN)
balanced accuracy    = (recall + specificity) / 2
false-positive rate  = FP / (FP + TN)
false-negative rate  = FN / (FN + TP)
```

When a denominator is zero, the value is JSON `null`. It is not reported as zero,
because zero would falsely imply that the rate had been measured.

Exact consensus-outcome accuracy is separate and uses only completed cases that
actually carry an independently reviewed exact outcome label.

### 3.6 Why this approach was chosen

| Alternative | Why the Phase 9 approach was preferred |
| --- | --- |
| A loose CSV of paths and labels | CSV is simple, but it does not naturally carry nested authorization, thresholds, exact outcomes, strict types, or versioned schema rules. |
| A second evaluation-only pipeline | It could drift from the CLI/API behavior. Reusing `run_evaluation()` measures the actual product boundary. |
| Stop at the first failed clip | This hides later coverage. Continuing with safe failure records preserves visibility while still marking the whole run incomplete. |
| Count failed clips as negative predictions | That mixes infrastructure failure with model behavior and can manufacture false negatives or true negatives. |
| Use timestamps or random UUIDs as run IDs | They identify executions, not meaning. Canonical content addressing makes semantic reruns detectable. |
| Store full audio paths, transcripts, and exception messages | That would make the report more sensitive without improving its metric meaning. Opaque identities and safe codes are enough. |
| Use a database | A database would add migrations, service state, and backup concerns. Immutable local JSON bundles fit this offline, single-project prototype and remain easy to inspect. |

## 4. A9.1 — Held-out end-to-end evaluation

### 4.1 Selection goal

A9.1 supplied a concrete labeled collection to the B9.1 machinery. The goal was a
reproducible regression measurement, not a claim of deployment accuracy.

The builder pins the earlier A3.4 source report by both evaluation ID and SHA-256.
If that report changes, the A9.1 manifest refuses to rebuild silently.

### 4.2 Deterministic sampling algorithm

The selection algorithm was:

1. load only cases from the A3.4 holdout split;
2. include all 16 held-out target-positive clips;
3. split negative clips by dataset;
4. rank negative category names using SHA-256 over a fixed seed, dataset ID, and
   category name;
5. choose eight distinct negative categories from ESC-50 and eight from
   UrbanSound8K;
6. within each chosen category, rank clip paths by the same fixed-seed SHA-256
   method and select one clip; and
7. sort the final 32 cases canonically by dataset, category, and path.

The fixed seed was `audio-sentinel-a9.1-heldout-v1`. Selection did not inspect model
scores, which prevents choosing negatives that merely make the model look better.

The result contained:

- 16 target clips and 16 background clips;
- equal dataset representation;
- all four held-out examples for each target source category; and
- 16 distinct negative source categories.

### 4.3 Why deterministic hash ranking was used

| Alternative | Limitation |
| --- | --- |
| Manually select “representative” negatives | Easy to cherry-pick, hard to reproduce, and dependent on undocumented judgment. |
| Use the model's score to select hard examples | Leaks model behavior into test construction and biases the resulting metric. |
| Call a random-number generator without a fixed seed | Produces a different manifest on each machine or run. |
| Use a fixed seeded pseudo-random shuffle | Reproducible in one implementation, but library/version changes can affect ordering. SHA-256 ranking has a small, explicit, language-independent rule. |
| Include every available negative | More complete, but unnecessarily expensive for the initial Phase 9 regression and still not representative of deployment prevalence. |

### 4.4 Decision rule

The evaluation used a predeclared acoustic candidate threshold of `0.5`. It was not
tuned after observing these 32 cases.

For this acoustic-only measurement, `log`, `review`, and `alert` counted as a
positive signal; `no_action` counted as negative. Exact expected consensus outcomes
were intentionally omitted because ESC-50 and UrbanSound8K provide acoustic class
labels, not independently reviewed action labels.

### 4.5 Results

All 32 cases completed:

| Measure | Result | Denominator |
| --- | ---: | ---: |
| True positives | 13 | 16 target clips |
| False negatives | 3 | 16 target clips |
| True negatives | 14 | 16 background clips |
| False positives | 2 | 16 background clips |
| Accuracy | 84.375% | 27 / 32 |
| Balanced accuracy | 84.375% | mean of recall and specificity |
| Precision | 86.667% | 13 / 15 positive predictions |
| Recall | 81.25% | 13 / 16 target clips |
| Specificity | 87.5% | 14 / 16 background clips |
| F1 | 83.871% | harmonic mean of precision and recall |
| False-positive rate | 12.5% | 2 / 16 background clips |
| False-negative rate | 18.75% | 3 / 16 target clips |
| Completion rate | 100% | 32 / 32 cases |

Observed outcomes were 17 `no_action`, 9 `log`, 6 `review`, and 0 `alert`.
Acoustic-only evidence cannot satisfy the independent alert-support requirements,
so the absence of alerts is consistent with the safety policy.

### 4.6 Error analysis

The two apparent false positives were ESC-50 cow and wind clips that produced
low-risk siren candidates. A single-label public dataset can omit co-occurring
sounds, so “false positive” here means disagreement with the available label, not
proof that no siren-like sound was audible.

The three false negatives were two glass-breaking clips and one siren clip. The
two missed glass clips were the clearest weakness because glass recall was only
2 of 4 in this small set.

The correct next step is larger, independently reviewed, class-specific analysis of
timing, masking, and score distributions. It would be unsafe to retune a global
threshold from two false positives and three false negatives.

### 4.7 Repeatability defect found and fixed

The first exact rerun revealed that the evaluation-run identity included new local
report, audit, and alert receipt IDs. The predictions and metrics were identical,
but the run appeared different because the storage receipts changed.

The fix separated two ideas:

- **traceability:** retain execution-specific receipt IDs in the saved report; and
- **semantic identity:** exclude those receipt IDs when deciding whether two
  measurements mean the same thing.

After the correction, the pinned rerun returned the same
`evaluation-run-33c6a7a6fd06d11f4722ef03` identity with `reused=true`.

### 4.8 What the result does not prove

The A9.1 result is not a production accuracy claim because:

- the set is small and artificially balanced;
- it reuses an earlier project holdout rather than untouched external validation;
- deployment prevalence will not be 50% target / 50% background;
- recording devices, rooms, distances, and noise differ from deployment;
- dataset labels can be incomplete;
- it measures only the acoustic path;
- it does not measure transcript, language, fairness, intent, or real incident
  outcomes; and
- it does not validate a notification or emergency procedure.

## 5. B9.2 — Retention and deletion controls

### 5.1 Problem being solved

The Phase 8 workflow creates durable local report and audit bundles. Leaving them
forever is not automatically safe, but deleting files by age alone can break audit
references, remove a pending alert, target the wrong directory, or erase the only
record of what was deleted.

B9.2 therefore treats deletion as a verified transaction, not a housekeeping shell
command.

### 5.2 Policy settings

The strict `RetentionSettings` contract contains:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `enabled` | `false` | Applying deletion is forbidden until explicitly enabled. |
| `final_report_days` | `30` | Minimum report age; `null` means retain indefinitely. |
| `alert_audit_days` | `365` | Minimum audit age; `null` means retain indefinitely. |
| `allow_pending_alert_deletion` | `false` | Pending local alerts remain protected. |
| `max_delete_count` | `1000` | Hard upper bound on one plan. |

Age is based on the verified document's timezone-aware `created_at`, not mutable
filesystem modification time.

### 5.3 Planning algorithm

The planner accepts no user-supplied artifact path. It discovers only the two fixed
managed locations:

- `final-reports/<report_id>/`; and
- `alert-audit/<audit_id>/`.

It then:

1. verifies the processed-data root is inside the project and does not overlap raw,
   interim, or model directories;
2. rejects links, junctions, noncanonical bundle names, or unexpected directory
   shapes;
3. reloads every report and audit through its existing integrity verifier;
4. selects expired audit bundles first;
5. protects any pending alert unless the policy explicitly allows its deletion;
6. builds the set of report IDs still referenced by retained audits;
7. selects only expired reports not protected by that reference set;
8. sorts audit targets before report targets; and
9. rejects the entire plan if the target count exceeds `max_delete_count`.

This is a small dependency-graph algorithm. An audit points to a report, so a
retained audit keeps its report alive even when the report's own age threshold has
passed.

### 5.4 Dry run and dual authorization

Planning is the default and writes nothing. Applied deletion requires both:

1. `enabled=true` in a reviewed project-local policy; and
2. an explicit `--apply` command-line action.

Either one alone is insufficient. This protects against accidentally applying the
checked-in example policy or accidentally passing `--apply` with a disabled policy.

### 5.5 Transactional deletion algorithm

Applied deletion uses a same-filesystem staging transaction:

```text
verify the complete plan
create a private quarantine directory
move every target bundle into quarantine
build the immutable deletion receipt
write the receipt into a private staging directory
read and validate the staged receipt
publish or verify the canonical receipt bundle
reload the published receipt
remove the quarantined bundles
```

If receipt creation or verification fails before publication, bundles are moved
back in reverse order. If restoration fails, the operation returns a distinct
`rollback_failed` code rather than pretending cleanup succeeded.

The receipt records the policy, time, scanned/protected counts, and opaque deleted
identities. It permanently says raw audio was not deleted and external deletion was
not authorized. It contains no path, audio, consent ID, transcript, recipient, or
transport setting.

Retention receipts are outside the target set, so the service cannot erase the
record of its own deletion.

### 5.6 Why this algorithm was chosen

| Alternative | Why it was not used |
| --- | --- |
| Direct recursive deletion | A partial failure could leave an unrecorded half-deleted inventory with no rollback opportunity. |
| Select by filesystem modification time | Modification time can change during copy, restore, or maintenance. The verified document creation time is the meaningful policy field. |
| Let callers pass arbitrary paths | That enlarges the destructive boundary and creates traversal and wrong-target risk. Fixed managed roots are safer. |
| Delete reports independently of audits | Retained audits could point to missing reports, breaking review and traceability. |
| Automatically delete pending alerts | A pending record still requires review. Deletion requires an explicit policy opt-in. |
| Run scheduled cleanup automatically | Scheduling creates a new authority and concurrency boundary. Phase 9 keeps deletion manual and reviewable. |
| Securely overwrite files | Reliable secure erasure depends on filesystem, SSD, backup, snapshot, and encryption behavior. The application cannot honestly guarantee it. |
| Use a database transaction | The project stores immutable filesystem bundles. A database would add an unrelated persistence system without solving external backups or raw-data lifecycle. |

### 5.7 Retention scope limits

B9.2 covers final reports and alert/audit bundles only. It does not delete:

- raw recordings;
- prepared windows;
- acoustic, speech, or language evidence;
- model files;
- datasets;
- backups, exports, or external copies; or
- retention receipts.

Those assets need separate dependency and authority rules before automated deletion
could be safe.

## 6. A9.2 — Consent, authorized use, review, and limitations

### 6.1 Why a rulebook was required

Technical validation can prove that a request contains a valid timestamp and an
allowed scope value. It cannot prove that recording people was lawful, that an
institution approved the purpose, or that a reviewer is authorized to act.

A9.2 makes that separation explicit.

### 6.2 Consent and scope algorithm

The implemented decision is intentionally simple:

| State | Result |
| --- | --- |
| Active `granted`, authorized device, `acoustic_only` | Run preparation and acoustic analysis only; mark speech/language `not_permitted`. |
| Active `granted`, authorized device, `acoustic_and_speech` | Permit acoustic, VAD, transcription, and language analysis. |
| Denied, withdrawn, expired, future, none, or unauthorized device | Reject before processing. |

The system stores an opaque consent reference, not a person's identity or the full
consent text. Authorization is established outside the application and rechecked at
processing boundaries. Raw-audio retention is a separate permission and does not
expand processing scope.

### 6.3 Outcome interpretation

| Outcome | Plain-language meaning |
| --- | --- |
| `no_action` | No configured policy signal crossed the current boundary. This is not proof of safety. |
| `log` | A low policy score was recorded. This is not an incident. |
| `review` | Score, uncertainty, missing evidence, ambiguity, or conflict requires a person. |
| `alert` | All software gates passed for a pending local candidate. Nothing was sent. |

### 6.4 Human review

The documented review procedure requires a person to verify authority and artifact
integrity, inspect every evidence branch, respect consent when listening, evaluate
transcript context, seek independent authorized information, and record the final
disposition in a separate approved workflow.

The prototype does not store reviewer identity, assignment, confirmation,
dismissal, or external escalation. This was deliberate: adding those fields without
authentication, organizational policy, and access control would create the
appearance of a complete incident-management system when none exists.

### 6.5 Model limitations made explicit

- **YAMNet:** AudioSet scores are not incident probabilities; timing is coarse and
  visually or acoustically similar events can be confused.
- **Silero VAD:** speech-likeness does not prove speech was correctly segmented and
  says nothing about meaning.
- **Faster-Whisper tiny.en:** English-only transcript candidates can be wrong;
  `derived_score` is not word-correctness probability.
- **Language rules:** deterministic phrases and context safeguards are inspectable,
  but finite rules cannot understand every dialect, quotation, joke, or intent.
- **Risk scoring:** the 0–100 result is an engineering policy score, not probability.
- **Consensus:** independent branches can still agree and be wrong; speech alone
  never satisfies alert support.

### 6.6 Why deterministic documented policy was preferred

Phase 9 did not add a learned intent classifier or large language model. The
existing rules are finite, versioned, inspectable, offline, and covered by fixtures
for negation, quotation, hypothetical language, and ambiguity. That makes their
limitations visible and their behavior reproducible.

A learned intent model might cover more wording, but it would add training-data,
calibration, bias, explainability, resource, and supply-chain questions that Phase 9
did not have representative data to answer.

## 7. A9.3 — Offline MVP demonstration workflow

### 7.1 Goal

The demonstration script provides one operator path that proves the real local
application can process an authorized recording, save a result, independently
verify that result, and preview cleanup without sending or deleting anything.

### 7.2 Preflight algorithm

Before loading a model, the script checks:

- explicit confirmation that the external consent record was reviewed;
- explicit confirmation that the device is authorized;
- bounded opaque identifiers;
- active offset-aware consent timestamps;
- the requested processing scope;
- a finite acoustic threshold from 0 through 1;
- a regular relative audio path inside `data/raw`;
- project-local optional configuration paths;
- required Python runtime modules; and
- required pinned local model directories.

Preflight never downloads or repairs a model. Missing prerequisites stop before
audio processing.

### 7.3 Three-stage demonstration algorithm

#### Stage 1: evaluate

The script constructs the normal `audio-sentinel evaluate` command and runs it as a
child process with the project `src` directory on `PYTHONPATH`. It requires
structured JSON and immediately checks that notification remains `not_sent` and
delivery authority remains false.

#### Stage 2: verify

The script calls the public `inspect-report` command on the returned report path.
The loader checks canonical location, hashes, identities, cross-document references,
and strict contracts. The script then compares:

- report ID;
- outcome;
- risk score;
- risk severity;
- review requirement; and
- alert-candidate state.

Any mismatch fails the demonstration.

#### Stage 3: plan retention

The script invokes the normal retention command without `--apply`. It rejects the
result if it claims deletion, raw-audio removal, or notification delivery.

The final summary includes only privacy-minimized identities, statuses, counts, and
permanent no-delivery fields.

### 7.4 Why the demo uses the public CLI

| Alternative | Limitation |
| --- | --- |
| Call private Python functions directly | The demo could succeed while the installed operator interface is broken or behaves differently. |
| Duplicate the pipeline inside the script | Two orchestration paths would drift and require separate safety review. |
| Trust the first JSON response | A response could disagree with the persisted artifact. Reloading verifies the durable result independently. |
| Demonstrate through a live microphone | This creates new device, continuous-consent, latency, buffering, and monitoring risks outside the offline MVP. |
| Add a web dashboard for the final demo | More UI code would not improve evidence integrity and would enlarge the surface being presented as complete. |
| Automatically apply retention afterward | A demonstration is not deletion authority. Dry-run-only behavior is easier to explain and safer to repeat. |

### 7.5 Outcome boundary

The demo never promises a particular result. A `review` or `alert` requires an
external human workflow. Even an `alert` remains local and pending; there is no
recipient, network transport, dispatch, or external action.

## 8. Post-A9.3 transcription fixes

### 8.1 How the timestamp issue appeared

Real authorized recordings produced valid speech fragments whose Faster-Whisper
timestamps extended slightly beyond the exact VAD fragment. Examples included:

| Verified source duration | Model end timestamp |
| ---: | ---: |
| 0.980 s | 1.000 s |
| 0.448 s | 0.840 s |
| 1.340 s | 1.520 s |
| 1.628 s | 2.000 s |

The decoded speech began inside real audio, but strict `end <= source_duration`
validation rejected the whole segment as `invalid_output`.

The first repair handled only sub-second inputs by padding them to one second. A
later recording proved the same behavior occurs for longer fragments, so the final
solution was generalized.

### 8.2 Final bounded-tail algorithm

At the model boundary:

1. validate the verified source waveform;
2. allocate an owned model input with `source_samples + 16,000` samples;
3. copy the exact source into the prefix;
4. fill the final 16,000 samples—one second at 16 kHz—with zeros;
5. count that tail against the configured model-input sample limit;
6. require the model-reported duration to match the actual padded input;
7. require every chunk to start before the verified source ends;
8. reject any end later than the padded input plus the tiny numeric epsilon;
9. publish `min(model_end, verified_source_duration)` as the public end; and
10. record source length, model-input length, and tail length separately.

This makes the accommodation explicit. Synthetic padding can help validate the
model's feature-window behavior, but it can never enlarge the public evidence span.

### 8.3 Why one bounded second was chosen

The observed outputs extended into the next feature region but remained within one
second of the verified source. A fixed one-second tail is:

- simple to reason about;
- independent of whether the source is shorter or longer than one second;
- large enough for all four observed boundary cases;
- bounded by the existing model-input limit; and
- auditable through explicit provenance fields.

### 8.4 Alternatives considered

| Alternative | Why it was rejected |
| --- | --- |
| Increase the timestamp epsilon globally | The required overshoot was hundreds of milliseconds, not floating-point noise. A large epsilon would hide materially invalid output. |
| Clip every timestamp without validating it | This could turn an arbitrary or malicious model timestamp into apparently valid evidence. Validation must happen before clipping. |
| Accept any end after the source | Unbounded acceptance destroys the timestamp trust boundary. |
| Ignore or remove timestamps | Timing is needed to connect text to the verified speech interval and later evidence. |
| Re-run the internal Whisper VAD | Silero and the Phase 4 segment contract already own VAD. A second hidden VAD would change segment meaning and reproducibility. |
| Merge neighboring speech fragments | That can change transcript context and evidence boundaries and does not directly solve model feature padding. |
| Pad only clips shorter than one second | The second real recording proved longer fragments can exhibit the same overshoot. |

### 8.5 Regression coverage

Tests prove that:

- the source prefix is byte-for-byte/sample-for-sample preserved;
- the tail contains zeros only;
- all four observed duration pairs are accepted;
- public timestamps stop at the verified source;
- timestamps entirely in padding are rejected;
- timestamps beyond the bounded model input are rejected;
- the input limit includes the synthetic tail; and
- orchestration verifies the new provenance fields.

The real pinned model was also exercised with synthetic arrays at all four observed
source sizes.

## 9. Transcript reliability policy fix

### 9.1 Why the original 0.80 boundary changed

The pinned Faster-Whisper `tiny.en` model produced clear, useful authorized test
phrases with derived scores roughly between `0.60` and `0.70`. Under the original
`0.80` policy, none could proceed to the language rules, so clearly recognized
weapon and threat wording contributed no language evidence.

The derived score is:

```text
exp(token-count-weighted mean of chunk average log probabilities)
```

It is bounded to `[0, 1]`, but it is not a calibrated probability that the
transcript is correct.

### 9.2 Policy v1.1

The new default bands are:

| Derived score | Handling |
| ---: | --- |
| Below 0.50 | `rejected_low_confidence`; do not pass text forward. |
| 0.50 to below 0.60 | `review_required`; retain visibly but block automatic handoff. |
| At least 0.60 | `accepted`; permit deterministic language analysis. |

Previously recorded v1.0 policies remain loadable with their explicitly stored
thresholds. New evidence records policy v1.1, so the decision is reproducible and
old evidence is not silently reinterpreted.

### 9.3 Why 0.60 was selected

`0.60` was the lowest boundary that admitted most of the observed clear phrases
while preserving a non-empty human-review band from `0.50` to below `0.60`.

It was preferred over:

- **keeping 0.80:** too insensitive for the pinned tiny model on the authorized
  real recording;
- **accepting at 0.50:** would eliminate the review band and pass every non-rejected
  transcript automatically; and
- **choosing a highly precise value from five phrases:** the sample is far too small
  to support statistical calibration.

The correct interpretation is an engineering sensitivity adjustment, not proof
that 0.60 is optimal.

### 9.4 False-positive safeguards retained

Lowering the transcript boundary means more transcription errors can reach language
matching. The risk is controlled, not eliminated, by keeping:

- explicit negation suppression;
- quotation/reported-speech context;
- hypothetical and rehearsal context;
- ambiguity handling;
- versioned deterministic rule fixtures;
- separate risk scoring;
- independent consensus requirements; and
- mandatory human review for the resulting real demo outcome.

Boundary tests show that active weapon/threat wording at exactly `0.60` produces
active findings, while negated and rehearsal/hypothetical examples remain context
suppressed.

The alert policy was not lowered. Speech still cannot independently authorize an
alert.

### 9.5 Real post-fix demonstration

The completed `acoustic_and_speech` demonstration on the authorized third test
recording produced:

| Field | Verified result |
| --- | --- |
| Speech segments | 5 |
| Accepted transcripts | 3 |
| Review-required transcripts | 2 |
| Language findings | 3 |
| Weapon-reference findings | 1 |
| Threat findings | 2 |
| Risk score | 62 |
| Risk severity | `high` |
| Consensus outcome | `review` |
| Alert candidate | `false` |
| Notification | `not_sent` |

This is the intended safety behavior: language evidence influences risk, but missing
independent alert support keeps the result in human review.

## 10. Technology stack and why it was chosen

### 10.1 Core language and contracts

| Technology | Phase 9 use | Why it fits |
| --- | --- | --- |
| Python 3.11+ | All services, scripts, contracts, metrics, and tests | Matches the existing audio/ML pipeline, has mature type hints and libraries, and avoids a cross-language serialization boundary. |
| Pydantic 2 | Strict immutable manifest, run, retention, and evidence contracts | Provides bounded field validation, forbidden unknown fields, finite-number checks, cross-field validators, JSON parsing, and JSON Schema generation from the same source. |
| JSON | Manifests, reports, policies, receipts, and CLI responses | Human-readable, portable, diffable, easy to inspect offline, and supported by Pydantic and other languages. |
| JSON Schema | Public v1 contract files | Allows non-Python tools to validate the same document shapes and prevents the Python model from becoming the only specification. |

### 10.2 Identity and filesystem stack

| Technology | Use | Reason |
| --- | --- | --- |
| `hashlib` SHA-256 | Manifest, run, artifact, and receipt identity/integrity | Deterministic, widely supported, and adequate for detecting content changes. It is explicitly not treated as a digital signature. |
| Canonical JSON | Stable semantic hashing | Sorted keys and fixed serialization prevent formatting differences from changing meaning. |
| `pathlib` and `os.path` | Containment and canonical path checks | Cross-platform path handling with explicit project-root boundaries. |
| `tempfile`, same-filesystem rename, and `fsync` | Staged receipt and deletion operations | Enables readback verification and rollback before irreversible removal. |
| Immutable directory bundles | Reports and audits | Easy to inspect, copy, hash, and verify without running a database server. |

### 10.3 Operator interfaces

| Technology | Use | Reason |
| --- | --- | --- |
| Typer | Installed `audio-sentinel` CLI | Typed options, clear help, structured tests, and reuse of Pydantic-backed services. |
| `argparse` | Standalone evaluation and demo scripts | Small standard-library entry points that can run from a checkout without another framework layer. |
| `subprocess` | Demo calls to the public CLI | Verifies the real operator boundary and isolates structured command success/failure. |
| FastAPI/Uvicorn | Existing loopback API reused by Phase 8 integration | Useful for local programmatic access, but Phase 9 does not expose it to the network or use it as authorization. |

### 10.4 Audio and model stack reused by Phase 9

| Technology | Role | Why retained |
| --- | --- | --- |
| NumPy | Typed waveform arrays and deterministic numeric handling | Standard in Python audio/ML, efficient, and already used throughout the pipeline. |
| SoundFile | Local WAV/FLAC reading | Reliable local decoding with explicit sample formats. |
| SciPy and librosa | Resampling, transforms, and features | Established signal-processing implementations already validated in earlier phases. |
| TensorFlow 2.21 + YAMNet | Acoustic event scoring | Pretrained AudioSet coverage, local execution, pinned runtime and model artifact. |
| ONNX Runtime 1.23.2 + Silero VAD v6 | Speech activity detection | Lightweight CPU inference with a pinned portable ONNX model. |
| Faster-Whisper 1.2.1 + CTranslate2 4.8.2 | English transcription | Efficient local CPU inference, deterministic decoding controls, and downloadable model files that can be pinned and verified. |

The model stack is split because each runtime is strong at a specific boundary:
TensorFlow serves YAMNet, ONNX Runtime serves the Silero graph, and CTranslate2
serves the converted Whisper model efficiently. Reimplementing or converting all
three into one runtime during Phase 9 would add model-equivalence and validation
risk without improving the evaluation, retention, or demonstration objectives.

### 10.5 Testing

`pytest` was chosen because the project is Python-based and already uses fixtures,
parameterization, temporary directories, monkeypatching, and contract comparisons.
Model boundaries are faked in ordinary tests so the suite remains fast and offline;
separate smoke checks and the final real demonstration exercise installed models.

## 11. Safety and privacy design principles

Phase 9 consistently follows these rules:

1. **Fail closed.** Invalid paths, timestamps, contracts, inventories, hashes, or
   model output stop the operation.
2. **Make authority explicit.** Consent scope, device authorization, retention
   enablement, and apply action are separate facts.
3. **Keep failures visible.** Failed evaluation cases remain failures rather than
   being counted as model predictions.
4. **Minimize outputs.** Collection reports and demo summaries omit audio,
   transcripts, direct identities, recipients, and absolute paths.
5. **Verify persisted state.** The demo reloads the report; retention reloads the
   receipt; evaluation verifies manifests and semantic identity.
6. **Separate evidence from action.** Model findings can affect risk and review,
   but they do not create delivery authority.
7. **Prefer reversible steps before destructive ones.** Retention plans first,
   stages targets, publishes a receipt, and restores on pre-publication failure.
8. **Record limitations beside results.** Metrics, model scores, and thresholds are
   not described as probabilities or production approval.

## 12. Testing and verification summary

Phase 9 added focused coverage for:

- content-addressed manifest and run identities;
- duplicate, malformed, unsafe, oversized, and tampered manifests;
- every confusion-matrix cell and undefined metric denominator;
- exact-outcome measurement separate from binary measurement;
- failed-case visibility and exception redaction;
- deterministic A9.1 manifest reproduction and pinned results;
- documented false-positive and false-negative parity;
- retention age, dependencies, pending alerts, limits, dry run, apply, rollback,
  receipt validation, and path safety;
- consent states, processing scopes, outcome interpretation, human review, and
  model-limit documentation parity;
- demo preflight, scope preservation, child commands, report comparison, no-delivery
  enforcement, and retention dry-run enforcement;
- all four observed Whisper timestamp overshoot cases;
- model-input ownership, zero-only tail, clipping, and invalid padding output;
- the 0.50/0.60 transcript policy boundaries and v1.0 compatibility; and
- active, negated, quoted, hypothetical, rehearsal, and ambiguous language cases.

The final verification state was:

- Python compilation: passed;
- focused speech/language tests: passed;
- connected evaluator/risk/consensus tests: passed;
- real pinned Faster-Whisper synthetic-length checks: passed;
- complete real offline MVP demonstration: passed; and
- full project suite: **1,865 passed**, with two existing Pydantic deprecation
  warnings.

## 13. Main Phase 9 artifacts

### Evaluation

- `src/audio_sentinel/evaluation_manifest.py`
- `scripts/run_evaluation_manifest.py`
- `scripts/build_a9_1_evaluation_manifest.py`
- `configs/a9-1-evaluation-manifest.json`
- `docs/evaluation-manifest.md`
- `docs/evaluation-findings.md`
- `docs/schemas/v1/evaluation-manifest.schema.json`
- `docs/schemas/v1/evaluation-run.schema.json`
- `tests/test_evaluation_manifest.py`
- `tests/test_evaluation_findings.py`

### Retention

- `src/audio_sentinel/retention.py`
- `configs/retention.example.json`
- `docs/retention-and-deletion.md`
- `docs/schemas/v1/retention-audit.schema.json`
- `tests/test_retention.py`

### Responsible use

- `docs/authorized-use-and-limitations.md`
- `tests/test_authorized_use_documentation.py`

### Offline demonstration

- `scripts/run_offline_mvp_demo.py`
- `docs/offline-mvp-demo.md`
- `tests/test_offline_mvp_demo.py`

### Post-phase fixes

- `src/audio_sentinel/transcription.py`
- `src/audio_sentinel/speech_transcription.py`
- `src/audio_sentinel/speech_contracts.py`
- `docs/transcription-wrapper.md`
- `docs/speech-evidence.md`
- `docs/speech-transcription.md`
- `tests/test_transcription.py`
- `tests/test_speech_transcription.py`
- `tests/test_speech_integration.py`
- `tests/test_language_analysis.py`

## 14. Manual operating steps that remain

The implementation cannot perform these organizational decisions automatically:

1. create and verify the authoritative external consent record;
2. decide whether `acoustic_only` or `acoustic_and_speech` is authorized;
3. install pinned model assets through the reviewed setup scripts;
4. choose experimental thresholds before evaluating a collection;
5. inspect `review` and `alert` outcomes through an approved human workflow;
6. choose retention periods from applicable policy or law;
7. review a retention dry run before enabling and applying deletion;
8. handle raw audio, intermediate evidence, backups, exports, and external copies
   through separate lifecycle controls; and
9. perform representative external validation before any deployment claim.

## 15. Reproduction commands

Rebuild the pinned A9.1 manifest:

```powershell
python scripts/build_a9_1_evaluation_manifest.py
```

Run the pinned collection where the ignored datasets and YAMNet model are installed:

```powershell
$env:TF_CPP_MIN_LOG_LEVEL='2'
& '.venv\yamnet\Scripts\python.exe' scripts\run_evaluation_manifest.py `
  configs\a9-1-evaluation-manifest.json `
  --expected-sha256 24da0b0fe6a352f00c1b79e9830db8ce79e8cd3f942776a6ea91be8c5dffefe9 `
  --output outputs\a9_1_evaluation\evaluation-run.json
```

Preview retention without deleting:

```powershell
audio-sentinel retention `
  --project-root . `
  --retention-config configs/retention.example.json
```

Run the demo preflight and demonstration only with a real, current, offset-aware
authorization record. The exact commands and safety checklist are in
[`offline-mvp-demo.md`](offline-mvp-demo.md).

Run the complete automated verification:

```powershell
python -m compileall -q src tests
python -m pytest -q
```

## 16. Final conclusion

Phase 9 completed the offline MVP boundary without overstating what the prototype
can do.

The project can now define a measurement before running it, preserve execution
failures, calculate standard metrics with explicit denominators, explain observed
errors, protect and delete selected durable records through a reviewed transaction,
state the consent and human-review rules in one place, and demonstrate the complete
workflow through its public local interface.

The post-phase speech fixes also show why real demonstrations matter. Synthetic
contract tests alone did not reveal every model timestamp behavior or the practical
sensitivity of the original transcript threshold. The final solution addressed
those observations without removing validation, expanding timestamps, weakening
alert consensus, or claiming calibration from a handful of phrases.

The correct overall interpretation is:

> Audio Sentinel is now a reproducible, inspectable, offline research MVP with
> explicit authorization, evidence, review, and retention boundaries. It is not an
> autonomous safety decision-maker or notification system.
