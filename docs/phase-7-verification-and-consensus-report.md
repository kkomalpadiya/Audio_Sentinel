# Phase 7 completion report: Verification and Consensus

## 1. Executive summary

Phase 7 built the safety layer that sits between the Phase 6 risk assessment and
the later alert-handling work in Phase 8.

In common terms, Phase 6 answers:

> "How concerning does the available evidence look?"

Phase 7 answers a different question:

> "Do the independent evidence branches tell a compatible enough story to choose
> no action, create a log entry, request review, or create a local alert
> candidate?"

That distinction is important. A large risk score is not treated as proof that an
incident occurred. Phase 7 checks whether the score is supported by the acoustic,
speech, and language branches, whether any branch contradicts another, whether
evidence is missing, whether consent limited the available evidence, and whether
uncertainty remains.

The completed phase provides:

- a strict contract for four mutually exclusive outcomes;
- a versioned and SHA-256-pinned evidence-agreement rule set;
- a deterministic cross-branch agreement and conflict algorithm;
- a final verification service that rechecks provenance before deciding;
- a false-alert safety matrix around thresholds and contradictions;
- a stable interface for a possible future trained verification model;
- a hash-pinned end-to-end acceptance suite and reproducible validation report;
- privacy-minimized, immutable, portable JSON-compatible outputs; and
- explicit safeguards that prevent a Phase 7 alert result from sending a
  notification.

At Phase 7 completion:

- all 182 focused Phase 7 tests pass;
- all 1,726 project tests pass;
- Python compilation passes;
- the generated audio-preparation smoke test passes;
- all 13 end-to-end consensus acceptance scenarios pass; and
- no Phase 7 production path requires a network connection, model download, or
  machine-learning runtime.

The most important safety result is simple: only a clean critical case supported
by both the acoustic and language branches can become a local alert candidate.
Conflicts, missing evidence, or blocking uncertainty send the case to review.

## 2. What Phase 7 does in common terms

Imagine three specialists reviewing the same recording:

- the **acoustic branch** looks for sound events such as an explosion, gunshot,
  siren, glass breaking, or speech-like audio;
- the **speech branch** says whether speech was detected and whether its
  transcript was reliable enough to accept; and
- the **language branch** checks accepted text for distress, threats, weapon
  references, ambiguity, or safe/suppressed context.

Phase 6 already turns the available evidence into a risk score and explains why.
Phase 7 acts like a cautious supervisor. It asks:

1. Do all records describe the same Phase 6 assessment?
2. Did the exact approved rules produce the agreement result?
3. Which branches support the risk assessment?
4. Which branches are neutral?
5. Do any branches contradict one another?
6. Is evidence missing or outside the user's consent scope?
7. Is uncertainty still present?
8. Given all of that, what is the safest next local action?

The answer is exactly one of these:

| Outcome | Common meaning |
| --- | --- |
| `no_action` | Complete evidence has a zero score and no reason to retain or review the result. |
| `log` | A small, clean signal is retained locally for later reporting. |
| `review` | A person or later review process should inspect the case because the score, uncertainty, missing evidence, or contradiction prevents safe automatic escalation. |
| `alert` | Every v1 safety gate passed, so Phase 8 may later create a local alert record. This is not notification delivery. |

## 3. Complete Phase 7 flow

```text
Phase 6 risk assessment
        |
        |  score, severity, reasons, evidence status, exact identity
        v
revalidate the complete assessment
        |
        v
apply the exact pinned agreement rules
        |
        +--> acoustic branch: supports / neutral / conflicts / unavailable
        +--> speech branch: neutral / conflicts / unavailable
        +--> language branch: supports / neutral / conflicts / unavailable
        |
        v
create a deterministic agreement evaluation
        |
        v
final verification service
        |
        +--> revalidate assessment and agreement
        +--> recompute the assessment SHA-256
        +--> rerun the trusted agreement algorithm
        +--> require an exact match with the supplied agreement
        +--> apply the fixed consensus decision policy
        |
        v
exactly one result: no_action / log / review / alert
        |
        v
privacy-minimized decision receipt
```

The future-model interface sits beside this deterministic path. It can eventually
suggest a candidate outcome, but it cannot replace the final verification service
or authorize notification.

## 4. Work completed task by task

| Task | Purpose | Main result |
| --- | --- | --- |
| A7.1 | Define the consensus policy and output contract | Strict four-outcome contract with conservative alert gates |
| B7.1 | Implement evidence agreement and conflict detection | Deterministic branch classifier with four cross-branch conflict rules |
| A7.2 | Implement the final verification service | Fail-closed trusted replay and outcome selection |
| B7.2 | Test disagreements, confidence boundaries, and false alerts | 27-test safety matrix with 15 primary scenarios |
| A7.3 | Define a future trainable verification-model interface | Stable feature, training, descriptor, prediction, trainer, and predictor contracts |
| A7.4 | Validate the complete deterministic pipeline | 13 pinned scenarios and a reproducible 13/13 passing report |

## 5. A7.1: Consensus policy and outcome contract

### 5.1 Why the contract was defined first

The output contract was created before the decision implementation. This keeps
the meanings of `no_action`, `log`, `review`, and `alert` stable instead of letting
implementation details gradually redefine them.

The contract is implemented in `src/audio_sentinel/consensus_contracts.py` and is
also exported as JSON Schema for non-Python consumers.

Contract-first design was chosen over loose dictionaries because loose records can
silently omit fields, add misspelled fields, mix incompatible versions, or store a
combination of values that does not make sense. The strict contract rejects those
states immediately.

### 5.2 The four outcomes

The v1 score ranges are:

| Score and state | Normal outcome |
| --- | --- |
| Exactly 0 with complete, non-conflicting, unsupported evidence | `no_action` |
| 1 through 24 with no review trigger | `log` |
| 25 or more, or any uncertainty/conflict/missing-data review trigger | `review` unless every alert gate passes |
| 75 or more with critical severity and every alert gate passing | `alert` |

These are not four independent flags. Exactly one primary outcome is allowed.

### 5.3 Alert safety gates

An alert candidate requires all of the following:

1. The risk score is at least 75.
2. The severity is `critical`.
3. The acoustic branch supports risk.
4. The language branch supports risk.
5. No branch conflicts with the risk story.
6. No expected branch is missing.
7. No blocking uncertainty is present.

Blocking uncertainty includes a transcript that still requires review, ambiguous
language, or the Phase 6 missing-data review reason.

Speech presence is deliberately excluded from alert support. Speech tells the
system that someone spoke; it does not, by itself, say that the speech was
dangerous.

### 5.4 Why alert still means review

An `alert` outcome sets both:

- `alert_candidate = true`; and
- `review_required = true`.

This is intentional. Phase 7 authorizes only a local candidate for later handling.
It does not declare a confirmed real-world incident, choose a recipient, contact
emergency services, or send email, SMS, push, or any other notification.

### 5.5 Canonical branch inventory

Every decision always contains acoustic, speech, and language states in that
order. A branch that is missing, not permitted, or not applicable cannot be
silently relabeled as neutral.

| Phase 6 input status | Required Phase 7 state |
| --- | --- |
| `present` | `supports_risk`, `neutral`, or `conflicts_risk` |
| `missing` | `unavailable` |
| `not_permitted` | `not_permitted` |
| `not_applicable` or `no_accepted_text` | `not_applicable` |

This preserves the difference between "we checked and found no support" and "we
could not or were not allowed to check."

## 6. B7.1: Evidence-agreement and conflict algorithm

### 6.1 Algorithm overview

The agreement engine is a deterministic rule engine. It does not average model
scores and it does not ask the branches to vote equally. Instead, it gives each
branch a role that matches what that branch can actually establish.

The algorithm performs these steps:

1. Revalidate the complete Phase 6 assessment.
2. Load and verify the exact local agreement-rule artifact.
3. Classify each branch independently as support, neutral, unavailable,
   not-permitted, or not-applicable.
4. Apply cross-branch contradiction rules.
5. Let a detected contradiction override support or neutrality for the involved
   branches.
6. Put every branch and every reason in canonical order.
7. Calculate supporting and conflicting branch lists.
8. Create a deterministic evaluation identity from canonical content.

### 6.2 Acoustic classification

The acoustic branch supports risk when at least one configured risk-bearing label
has a peak score greater than or equal to `0.85`.

In simplified pseudocode:

```text
if branch is not present:
    preserve missing / not-permitted / not-applicable state
else if any risk-bearing acoustic signal has peak_score >= 0.85:
    supports_risk
else if a risk-bearing acoustic signal exists below 0.85:
    neutral with low-confidence reason
else:
    neutral with no-risk-signal reason
```

The exact boundary is inclusive: `0.85` supports, while `0.849999` does not.

### 6.3 Speech classification

Speech is always neutral by itself. It can participate in consistency checks, but
it never counts as risk support.

This avoids the unsafe assumption that detected speech is suspicious. Most speech
is ordinary, and the language branch owns the meaning of accepted text.

### 6.4 Language classification

The language branch supports risk when at least one finding has one of these
categories:

- `distress`;
- `threat`; or
- `weapon_reference`.

These categories remain neutral:

- `no_concerning_match`;
- `ambiguous`; and
- `context_suppressed`.

Ambiguity is neutral for agreement but still remains a Phase 6 uncertainty reason
that can force review. This prevents uncertainty from being treated as positive
support while still preventing unsafe automatic escalation.

### 6.5 The four contradiction rules

The engine checks four concrete disagreements:

1. **Acoustic no-speech versus detected speech**
   - High-confidence acoustic `no_speech` conflicts with VAD speech segments.
   - Language also conflicts when findings claim analyzed speech in that state.

2. **Acoustic speech versus zero VAD speech**
   - A high-confidence speech-like acoustic label conflicts with zero speech
     segments.

3. **Non-threatening speech versus concerning language**
   - High-confidence `non_threatening_speech` conflicts with active distress,
     threat, or weapon-reference language.

4. **Language findings without accepted text**
   - Language findings conflict with a speech summary that has no accepted
     transcript.

### 6.6 Why conflicts override support

Suppose the acoustic branch appears to support risk, but the speech and language
provenance contradicts the story. Averaging the values could still produce a large
number and hide the contradiction. Phase 7 instead keeps the contradiction visible
and routes the case to review.

This is a fail-safe design: incompatible evidence is not allowed to cancel itself
out numerically.

### 6.7 Versioned local rules

The agreement policy is stored in
`src/audio_sentinel/resources/consensus-agreement-rules-v1.json` with:

- a rule-set ID;
- semantic version;
- format version;
- complete acoustic and language taxonomies;
- all mandatory conflict switches; and
- an exact SHA-256 digest.

The loader rejects changed bytes, missing categories, reordered required lists,
duplicates, unknown fields, oversized input, and attempts to disable required
conflict checks.

## 7. A7.2: Final verification and decision algorithm

### 7.1 Why the service does not trust the supplied agreement

The final service treats the agreement document as a claim that must be verified.
Before choosing an outcome, it:

1. revalidates the complete Phase 6 assessment;
2. revalidates the complete B7.1 agreement evaluation;
3. recomputes the canonical assessment SHA-256;
4. confirms the agreement references the same assessment ID and hash;
5. reruns B7.1 using the trusted rule artifact and original evaluation time; and
6. requires the recomputed agreement to equal the supplied agreement exactly.

If any check fails, the service stops with a stable safe error. It does not make a
best-effort decision from partially trusted data.

This is called **trusted replay** in this report: the final authority reproduces
the important intermediate result rather than trusting it blindly.

### 7.2 Decision tree

The simplified decision algorithm is:

```text
alert_eligible =
    score >= 75
    AND severity == critical
    AND acoustic supports risk
    AND language supports risk
    AND there are no conflicts
    AND there are no missing branches
    AND there is no blocking uncertainty

if alert_eligible:
    outcome = alert
else if score >= 25
        OR Phase 6 requires review
        OR any conflict exists
        OR any expected evidence is missing:
    outcome = review
else if 1 <= score <= 24:
    outcome = log
else if score == 0 AND no branch supports risk:
    outcome = no_action
else:
    fail closed because the state is inconsistent
```

### 7.3 Why this precedence order is used

The order is `alert`, then `review`, then `log`, then `no_action`.

Alert is evaluated first because a clean critical case also has a Phase 6 review
flag by design. If review were checked first, no case could ever reach the alert
candidate state.

Review comes before log and no action because uncertainty, missing evidence, and
conflicts must never be hidden by a low or zero score.

### 7.4 Explicit reason receipts

The service records why it chose the result. Examples include:

- `no_risk_evidence`;
- `low_risk_logged`;
- `score_review_required`;
- `risk_uncertainty_review_required`;
- `missing_evidence_review_required`;
- `evidence_conflict_review_required`;
- `insufficient_alert_agreement`;
- `consent_limited_evidence`;
- `critical_risk`;
- `multi_branch_alert_agreement`; and
- `alert_candidate`.

The reasons are data, not free-form prose. This makes them portable, testable, and
safe to use in later reports.

## 8. Deterministic identities and canonical hashing

### 8.1 What canonical JSON means

The same logical object can be written as JSON in different key orders or with
different whitespace. Before hashing, Phase 7 converts relevant content to a fixed
representation:

- keys are sorted;
- separators are fixed;
- enumeration values are serialized consistently;
- non-finite numbers are forbidden; and
- timestamps that should not change semantic identity are excluded.

SHA-256 is then calculated over those canonical bytes.

### 8.2 What is hash-pinned

Phase 7 uses SHA-256 to pin:

- the Phase 6 risk assessment consumed by B7.1 and A7.2;
- the agreement-rule artifact;
- the agreement evaluation's semantic content;
- the final consensus decision's semantic content;
- future-model feature vectors;
- future training source documents and feature vectors;
- future model and training-dataset artifacts; and
- the A7.4 scenario corpus and validation report identity.

### 8.3 Why SHA-256 was chosen

SHA-256 is widely implemented, available in Python's standard library, suitable
for integrity checks, and easy to reproduce in other languages.

It was chosen over timestamps or random UUIDs because those identify a run, not
its meaning. Two identical decisions should have the same semantic identity even
when created at different times.

The hashes are used for integrity and provenance. They are not encryption and do
not make private content safe to publish.

## 9. B7.2: Disagreement and false-alert safety matrix

B7.2 added 27 tests, including a primary table of 15 representative evidence
combinations. The tests run the real Phase 6 scorer, B7.1 agreement engine, and
A7.2 final service together.

The main cases cover:

| Case | Required behavior |
| --- | --- |
| Clean zero evidence | No action |
| Clean low positive score | Log |
| Medium score | Review |
| Critical score just below acoustic support threshold | Review for insufficient agreement |
| Critical score exactly at acoustic support threshold | Local alert candidate |
| Low-confidence no-speech disagreement | No conflict because the acoustic evidence is below the trusted boundary |
| High-confidence no-speech disagreement | Explicit conflict and review |
| Acoustic speech with zero VAD segments | Explicit acoustic/speech conflict |
| Non-threatening speech with threat language | Explicit acoustic/language conflict |
| Language without accepted transcript | Explicit speech/language conflict |
| Transcript needing review | Otherwise eligible alert stays in review |
| Ambiguous language | Neutral support state but explicit uncertainty review |
| Missing acoustic evidence | Missing branch forces review |
| Acoustic-only consent | Consent limitation stays explicit and cannot become alert |
| Safe language with high acoustic evidence | Language stays neutral and cannot add support |

Boundary tests explicitly cover acoustic peak scores `0.0`, `0.849999`, `0.85`,
and `1.0`. Taxonomy tests cover all six language categories.

### Why scenario matrices were used

Single-function unit tests are necessary but not sufficient. A false alert can
result from several individually correct components interacting incorrectly.

The matrix was chosen because it lets one row state the complete expected story:
score, support, conflicts, outcome, reasons, review state, and alert-candidate
state. It also makes threshold cases and dangerous near misses easy to review.

## 10. A7.3: Future trainable verification-model interface

### 10.1 What was built

A7.3 defines a safe integration boundary for a possible future trained model. It
does not implement, train, load, or run such a model.

The interface provides:

- a canonical feature-vector contract;
- reviewed/adjudicated training-target contracts;
- hash-pinned training examples;
- reproducible model descriptors;
- probability-bearing prediction contracts;
- a runtime-checkable trainer protocol; and
- a runtime-checkable predictor protocol.

### 10.2 Trusted feature extraction

Feature extraction first passes the assessment and agreement through the A7.2
trusted-source boundary. This prevents a training or prediction pipeline from
learning from an agreement document that the production verifier would reject.

The fixed feature vector contains:

- score, severity, and Phase 6 review state;
- one presence flag for every Phase 6 risk reason;
- the status and agreement state of every branch;
- one presence flag for every B7.1 reason on each branch;
- counts of supporting, eligible-supporting, conflicting, missing, and
  not-permitted branches; and
- the overall conflict flag.

### 10.3 Privacy and target-leakage controls

Features exclude:

- clip, consent, assessment, agreement, evidence, and decision IDs;
- timestamps;
- raw audio;
- transcript or matched text;
- speaker identity;
- filesystem paths;
- model tensors; and
- the current A7.2 outcome.

The current outcome is excluded because including it would give a supervised model
the answer it is supposed to learn. That mistake is called target leakage.

### 10.4 Training targets

Only `human_reviewed` or `adjudicated` outcome labels are accepted. A label
generated by the existing deterministic system is not automatically treated as
independent training truth.

This prevents a future model from merely copying the current policy while being
described as independently trained.

### 10.5 Prediction contract

A prediction must provide exactly one probability for each outcome in canonical
order. Probabilities must:

- be finite;
- fall between zero and one;
- sum to one within a small numeric tolerance; and
- agree with the reported maximum-probability recommendation and confidence.

Every prediction permanently records:

- `requires_deterministic_verification = true`; and
- `alert_is_local_candidate_only = true`.

Even a future model's `alert` recommendation cannot bypass A7.2.

### 10.6 Why an interface was built before a model

The project does not yet have representative, independently reviewed, end-to-end
training data for a trustworthy verification model. Training now would encourage
overfitting to synthetic rules and would create confidence without evidence.

Defining the interface now still provides value:

- data collection has a stable target format;
- privacy exclusions are enforceable before datasets grow;
- model artifacts will need explicit provenance;
- different algorithms can be evaluated behind the same boundary; and
- the deterministic safety layer remains authoritative.

## 11. A7.4: Executable end-to-end acceptance validation

### 11.1 What was validated

A7.4 added a separate, SHA-256-pinned corpus of 13 reviewed acceptance scenarios.
Each scenario expands into the public Phase 6 input contract and passes through:

1. `score_risk()`;
2. `evaluate_evidence_agreement()`; and
3. `decide_consensus()`.

The validator compares:

- risk score;
- severity;
- Phase 6 review state;
- every branch's input and agreement state;
- supporting branches;
- conflicting branches;
- final outcome;
- decision reason codes;
- review state; and
- local alert-candidate state.

### 11.2 The 13 acceptance scenarios

| Scenario | Expected result |
| --- | --- |
| Quiet present evidence | No action |
| Low siren evidence | Log |
| Medium glass-break evidence | Review |
| Critical evidence below acoustic support boundary | Review; one supporting branch is insufficient |
| Clean critical evidence at boundary | Local alert candidate |
| High-confidence no-speech contradiction | Review with all involved branches conflicting |
| Speech-like acoustic evidence without VAD speech | Review with acoustic/speech conflict |
| Non-threatening acoustic speech with threat language | Review with acoustic/language conflict |
| Language without accepted transcript | Review with speech/language conflict |
| Transcript uncertainty in an otherwise eligible alert | Review |
| All branches missing | Review, not no action |
| Acoustic-only consent with high risk | Review with consent receipt |
| Safe language beside high acoustic evidence | Review with language neutral |

All 13 pass. Only the clean critical multi-branch case has
`alert_candidate = true`.

### 11.3 What the validation report means

The report status is:

`validated_for_deterministic_v1_scenarios`

This means the implemented v1 mechanics match the reviewed synthetic cases. It
does not mean the system's real-world incident accuracy is known.

The report explicitly records this limitation:

`synthetic_acceptance_validation_not_empirical_incident_accuracy`

### 11.4 Reproducible report identity

The validation ID includes the scenario-suite hash, risk-rule descriptor,
agreement-rule descriptor, consensus policy, and expected-versus-observed case
results. Run time is excluded, so a later reproduction with unchanged artifacts
has the same validation ID.

## 12. Why these algorithms were chosen

### 12.1 Deterministic rules instead of a trained end-to-end classifier

**Chosen:** explicit thresholds, typed states, contradiction rules, and a fixed
decision tree.

**Alternatives considered:** neural network, gradient-boosted classifier,
logistic regression, or another learned ensemble.

**Why the deterministic approach was chosen now:**

- the project does not yet have representative reviewed training labels;
- the decision can be explained exactly;
- important safety gates cannot be learned away accidentally;
- boundary behavior is directly testable;
- no ML runtime or network is needed;
- rule and policy changes are easy to audit; and
- the result does not pretend to be empirically calibrated.

A learned verifier may later improve ranking or recommendation quality, but only
after suitable data and evaluation exist.

### 12.2 Hard safety gates instead of one weighted average

**Chosen:** alert requires every independent gate to pass.

**Alternative:** combine support and conflict values into one weighted total.

A weighted total can hide a dangerous contradiction. Strong acoustic evidence
could numerically outweigh the fact that language has no accepted transcript, or
two uncertain signals could add up to a confident-looking number. Hard gates keep
each safety requirement visible and non-negotiable.

### 12.3 Asymmetric branches instead of majority vote

**Chosen:** acoustic and language may support; speech is neutral or conflicting.

**Alternative:** let all three branches vote equally.

The branches do not answer the same question. Speech presence is not evidence of
danger. A two-of-three vote could incorrectly treat ordinary speech as a risk vote
or let two correlated text-derived signals dominate. The asymmetric design follows
the meaning and independence of each branch.

### 12.4 Conflict override instead of conflict cancellation

**Chosen:** contradictions replace support/neutral states for involved branches
and force review.

**Alternative:** subtract a conflict penalty from support.

A penalty still allows a contradiction to disappear inside a sufficiently large
score. Override semantics preserve the fact that the evidence story is internally
inconsistent.

### 12.5 Trusted replay instead of trusting upstream metadata

**Chosen:** recompute the assessment hash and agreement result at the final gate.

**Alternative:** accept a caller's claimed IDs, checksums, and agreement result.

Metadata can be stale, mismatched, or tampered with. Replaying the deterministic
agreement calculation is inexpensive and proves that the exact trusted rules
produce the supplied result.

### 12.6 Fail closed instead of best-effort recovery

**Chosen:** invalid, mismatched, or unexplained states stop with safe error codes.

**Alternative:** ignore bad fields, drop unavailable branches, or guess an
outcome.

Best-effort recovery is useful in some user-interface workflows, but unsafe at a
verification boundary. A partial decision could appear valid while hiding why the
input failed.

### 12.7 Fixed feature inventory instead of sparse ad hoc features

**Chosen:** every future-model feature vector contains every reason and branch in
canonical order, using explicit presence flags.

**Alternative:** store only features that happen to be present.

Fixed shape makes missing features distinguishable from absent signals, prevents
schema drift, makes dataset columns stable, and is easier to validate across
training and prediction.

### 12.8 Synthetic acceptance scenarios instead of claiming real-world accuracy

**Chosen:** reviewed synthetic cases prove policy mechanics and boundaries.

**Alternative:** report accuracy, sensitivity, or false-positive rates from a
small or unrepresentative sample.

Synthetic tests can reliably prove what the code does. They cannot prove how often
real incidents occur or how models perform across environments, microphones,
languages, speakers, and noise conditions. The report states that limitation
instead of manufacturing a misleading metric.

## 13. Technology stack and why it was chosen

### 13.1 Python 3.11+

Phase 7 uses Python because the rest of the Audio Sentinel pipeline already uses
Python, and Python provides mature support for typed data models, JSON, hashing,
testing, audio/ML integration, and command-line tools.

Python 3.11+ was appropriate because it provides:

- modern type annotations;
- enums, dataclasses, and protocols in the standard language;
- stable standard-library JSON, hashing, paths, and package resources;
- good test tooling; and
- compatibility with the project's existing audio and optional ML dependencies.

Using another language only for consensus would add a deployment boundary and
duplicate contracts without providing a clear Phase 7 benefit.

### 13.2 Pydantic 2

Pydantic supplies the strict runtime contracts. Phase 7 models generally use:

- `extra="forbid"` so unknown fields are rejected;
- frozen records so validated outputs cannot be mutated;
- finite-number checks;
- field bounds and regular-expression constraints;
- cross-field model validators; and
- generated JSON Schema.

It was chosen over plain dictionaries because dictionaries do not enforce
structure. It was chosen over standard-library dataclasses alone because
dataclasses provide structure but do not automatically provide the same runtime
validation and schema generation.

### 13.3 Enums and literal types

Outcomes, branch states, reason codes, and coverage markers use enums. Fixed
versions and safety flags use literal types where appropriate.

This was chosen over free-form strings because a typo such as `reveiw` should be a
validation error, not a new accidental state.

### 13.4 JSON and JSON Schema

JSON is used for local rule artifacts, validation scenarios, examples, and saved
reports. JSON Schema describes the portable public contracts.

JSON was chosen because it is:

- human-readable;
- language-neutral;
- easy to review in Git;
- supported without a service or database; and
- appropriate for small versioned policy artifacts and reports.

A binary format would be more compact but harder to inspect. A database would add
operational complexity without helping these small immutable artifacts.

### 13.5 SHA-256 through `hashlib`

Python's standard `hashlib` supplies SHA-256 for artifact integrity and semantic
identities. This avoids a third-party cryptography dependency for a standard hash
operation.

### 13.6 `importlib.resources`

Pinned JSON rules and scenario files are loaded as package resources. This keeps
runtime lookup independent of the user's current working directory and allows the
files to travel with the installed Python package.

### 13.7 `dataclasses` and `typing.Protocol`

Frozen dataclasses bind validated rule objects to exact byte-level provenance.
Runtime-checkable protocols define the future trainer and predictor interfaces.

Protocols were chosen over a required ML base class because they preserve loose
coupling. A future implementation can use scikit-learn, PyTorch, TensorFlow,
ONNX Runtime, or another system as long as it honors the same public behavior.

### 13.8 pytest 8

pytest provides parameterized scenario matrices, fixtures, temporary directories,
exception assertions, and straightforward focused/full-suite execution.

It was chosen over manual testing because threshold behavior, ordering, tamper
rejection, and false-alert invariants need repeatable regression protection.

### 13.9 Standard-library command-line tools

The A7.4 validation command uses `argparse`, `pathlib`, and `json`. No separate CLI
framework is needed for one bounded command.

The command writes a new validated report and refuses to overwrite an existing
file, preserving prior evidence.

### 13.10 Git and checked-in policy artifacts

Rules, schemas, scenarios, examples, documentation, and generated acceptance
evidence are stored in Git. Git provides reviewable change history around policy
and thresholds.

The SHA-256 pins detect byte-level changes at runtime; Git explains when and why a
reviewed change entered the project. These controls serve different purposes and
work together.

### 13.11 Technologies deliberately not required in Phase 7

Phase 7 does not require:

- TensorFlow;
- PyTorch;
- ONNX Runtime;
- a language model;
- cloud inference;
- a database;
- a message queue;
- an email or SMS provider;
- a web API; or
- network access.

The deterministic consensus problem uses small typed records and policy rules, so
adding those technologies now would increase the attack surface and deployment
cost without improving the validated decision mechanics.

## 14. Worked examples

### 14.1 No action

Input story:

- risk score is 0;
- acoustic evidence is present but has no risk-bearing signal;
- no speech is detected;
- language is not applicable because there is no accepted text;
- nothing conflicts; and
- nothing is missing.

Result: `no_action` with `no_risk_evidence`.

This does not mean the recording is universally proven safe. It means this complete
v1 evidence package contains no configured risk signal or review trigger.

### 14.2 Log

Input story:

- a siren produces a score of 10;
- the acoustic peak remains below the 0.85 support threshold;
- no conflict or uncertainty exists.

Result: `log` with `low_risk_logged`.

The small signal is retained without escalating it to review.

### 14.3 Review because a score is critical but support is incomplete

Input story:

- an explosion signal is just below acoustic support at `0.849999`;
- accepted language contains a threat finding;
- the total risk score is 82, which is critical;
- language supports risk, but acoustic remains neutral.

Result: `review` with `score_review_required` and
`insufficient_alert_agreement`.

The high score does not override the two-branch corroboration rule.

### 14.4 Clean local alert candidate

Input story:

- explosion peak is exactly `0.85`;
- accepted language contains a threat finding;
- the score is 92 and severity is critical;
- acoustic and language both support;
- speech is neutral;
- there are no conflicts, missing branches, or blocking uncertainties.

Result: `alert` with `critical_risk`, `multi_branch_alert_agreement`, and
`alert_candidate`.

The record still says `review_required = true` and sends nothing externally.

### 14.5 Review because evidence conflicts

Input story:

- high-confidence acoustic evidence says `no_speech`;
- VAD reports speech segments;
- language reports a threat finding.

Result: all involved branches are marked conflicting and the final outcome is
`review` with `evidence_conflict_review_required`.

The system does not average these claims into a confident result.

### 14.6 Review because consent limits evidence

Input story:

- the authorized scope is acoustic only;
- acoustic evidence is high-risk;
- speech and language are `not_permitted`, not missing.

Result: `review` with `consent_limited_evidence`.

The system preserves what it was not allowed to process and cannot fabricate the
second supporting branch needed for alert.

## 15. Privacy, safety, and integrity controls

### 15.1 Data minimization

Consensus artifacts contain IDs, hashes, numeric scores, enumerated states, rule
descriptors, and reason codes. They exclude raw audio, transcript text, matched
phrases, speaker identity, recipient data, and delivery details.

### 15.2 Consent preservation

Not-permitted branches remain explicitly not permitted. They are not relabeled as
safe, missing, or neutral evidence.

### 15.3 Missing data is not safety

Missing expected evidence forces review. A zero score combined with missing data
cannot become `no_action`.

### 15.4 No confidence rounding across the boundary

`0.849999` remains below the 0.85 acoustic agreement threshold. Display rounding
cannot promote it to support.

### 15.5 No one-branch alert

Only acoustic and language are alert-eligible, and both are required. Strong
evidence from one branch cannot impersonate independent corroboration.

### 15.6 No notification side effect

The decision service creates a document only. It does not know recipient contact
details and has no email, SMS, push, or emergency-service integration.

### 15.7 Bounded artifacts

Rule and scenario loaders enforce byte limits, exact formats, complete taxonomies,
and checksum expectations. This reduces accidental or hostile resource abuse.

### 15.8 Stable safe errors

Failures use codes such as `invalid_assessment`, `assessment_mismatch`,
`untrusted_agreement`, or `risk_rule_set_mismatch`. Error messages do not echo
private transcript or audio content.

## 16. Testing and verification summary

| Task | Focused tests added | Main coverage |
| --- | ---: | --- |
| A7.1 | 30 | Contract validity, all outcomes, safety gates, canonical ordering, schema and example |
| B7.1 | 42 | Rule integrity, support taxonomy, all conflicts, status preservation, determinism and privacy |
| A7.2 | 19 | Trusted replay, all outcomes, tamper rejection, custom-rule trust and deterministic identity |
| B7.2 | 27 | Thresholds, disagreements, taxonomy, missing/consent states and false-alert invariants |
| A7.3 | 40 | Feature integrity, privacy, reviewed targets, predictions, protocols and offline import |
| A7.4 | 24 | Pinned suite, end-to-end results, failure reports, CLI, privacy and offline execution |
| **Total Phase 7** | **182** | Complete verification and consensus behavior |

Final project verification at A7.4 completion:

- **1,726 tests passed**;
- Python compilation passed;
- the generated preparation smoke test passed;
- the Phase 7 suite passed all **182 tests**; and
- the A7.4 acceptance report passed **13 of 13 scenarios**.

## 17. Main files produced during Phase 7

### 17.1 Production modules

- `src/audio_sentinel/consensus_contracts.py`
- `src/audio_sentinel/consensus_rules.py`
- `src/audio_sentinel/consensus_service.py`
- `src/audio_sentinel/verification_model.py`
- `src/audio_sentinel/consensus_validation.py`

### 17.2 Versioned resources

- `src/audio_sentinel/resources/consensus-agreement-rules-v1.json`
- `src/audio_sentinel/resources/consensus-validation-scenarios-v1.json`

### 17.3 Schemas and examples

- `docs/schemas/v1/consensus-decision.schema.json`
- `docs/schemas/v1/consensus-agreement.schema.json`
- `docs/schemas/v1/consensus-agreement-rule-set.schema.json`
- `docs/examples/consensus-decision.json`
- `docs/examples/consensus-agreement.json`

The verification-model module can generate feature, training-example, model, and
prediction schemas when a non-Python consumer needs them. Those derived schema
files are not checked in.

### 17.4 Tests

- `tests/test_consensus_contracts.py`
- `tests/test_consensus_rules.py`
- `tests/test_consensus_service.py`
- `tests/test_consensus_safety.py`
- `tests/test_verification_model.py`
- `tests/test_consensus_validation.py`

### 17.5 Commands, reports, and documentation

- `scripts/validate_consensus_pipeline.py`
- `outputs/a7_4_validation/consensus_pipeline_validation.json`
- `docs/consensus-policy.md`
- `docs/consensus-agreement.md`
- `docs/consensus-service.md`
- `docs/consensus-safety-tests.md`
- `docs/trainable-verification-model.md`
- `docs/consensus-validation.md`
- this Phase 7 completion report

## 18. Reproduction commands and manual steps

No model download, network connection, account, secret, or external service is
required for Phase 7.

Run all Phase 7 tests:

```powershell
python -m pytest tests/test_consensus_contracts.py tests/test_consensus_rules.py tests/test_consensus_service.py tests/test_consensus_safety.py tests/test_verification_model.py tests/test_consensus_validation.py -q
```

Run the complete project verification:

```powershell
.\scripts\verify_project.ps1
```

Reproduce the A7.4 report at a new path:

```powershell
python scripts/validate_consensus_pipeline.py --output outputs/a7_4_validation/reproduced.json
```

The command deliberately refuses to overwrite an existing report. A reproduced
report should have the same `validation_id`; only `created_at` may differ.

## 19. Important limitations

Phase 7 does not prove:

- real-world incident accuracy;
- sensitivity or specificity;
- false-positive or false-negative rates;
- calibration of the 0.85 or 75 boundaries against field incidents;
- equal performance across microphones, rooms, noise types, accents, languages,
  or speakers;
- that the underlying acoustic, speech, or language evidence is always correct;
- notification delivery reliability; or
- that an alert candidate represents a confirmed emergency.

The rules are an explicit, reviewable v1 safety policy. Thresholds can only be
empirically justified after representative labeled evaluation data exists.

The future verification-model interface is inactive. It has no trained model,
model artifact, selected algorithm, or production prediction path.

## 20. What Phase 7 hands to Phase 8

Phase 8 receives a strict `ConsensusDecisionDocument` containing:

- the exact risk-assessment reference;
- the consensus policy version;
- acoustic, speech, and language branch states;
- one primary outcome;
- explicit reason codes;
- the review requirement; and
- the local alert-candidate flag.

Phase 8 can now build offline evaluator orchestration and local audit records
without reinterpreting raw model outputs. It must continue to respect the rule
that a Phase 7 alert is only a candidate and not delivery authorization.

## 21. Short glossary

| Term | Meaning |
| --- | --- |
| Agreement | Whether one evidence branch supports, is neutral to, or conflicts with the risk story |
| Branch | One evidence source: acoustic, speech, or language |
| Canonical JSON | A fixed JSON representation used to make hashing reproducible |
| Conflict | Two evidence claims that cannot safely be treated as one compatible story |
| Consensus | The final cross-branch decision process, not simple majority voting |
| Deterministic | The same validated input and rules produce the same semantic result |
| Fail closed | Stop safely instead of guessing when required integrity checks fail |
| Hash pin | An expected SHA-256 digest used to detect changed artifact bytes |
| Local alert candidate | A record that later code may handle; it is not a sent notification |
| Provenance | Evidence showing which exact input and rule artifacts produced a result |
| Target leakage | Giving a training feature the answer the model is supposed to learn |
| Trusted replay | Recomputing an intermediate result with trusted rules before relying on it |

## 22. Conclusion

Phase 7 changed Audio Sentinel from a system that can calculate a risk score into
one that can make a cautious, explainable, and reproducible local consensus
decision.

The design deliberately favors explicit safety gates over opaque confidence,
contradiction review over numeric cancellation, exact provenance over trust in
metadata, and honest synthetic validation over unsupported accuracy claims.

The result is not a finished alerting system. It is the verified decision boundary
that a later alerting system can safely build upon.
