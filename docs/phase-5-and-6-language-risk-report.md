# Phase 5 and Phase 6: Language Understanding and Risk Assessment

## Executive summary

Phase 5 turned accepted speech transcripts into structured, explainable language
evidence. Phase 6 combined that language evidence with acoustic and speech evidence
to produce a bounded 0-100 risk assessment.

The central design choice in both phases was to use deterministic, versioned rules
instead of pretending that the project already had enough representative labeled
data for a reliable learned model. Every important result can therefore answer:

- what input was used;
- which exact rule artifact was used;
- what matched and where;
- why a match was active, ambiguous, or suppressed;
- how each branch contributed to the score;
- why human review was or was not required; and
- whether any required evidence was missing or outside the consent scope.

This is intentionally an evidence and triage pipeline. It does not confirm that an
incident occurred, identify a speaker, infer intent, or send an alert. Phase 7 will
make a separate consensus decision using the Phase 6 assessment as one input.

At the end of these phases:

- the focused Phase 5 suite passes 325 tests;
- the focused Phase 6 suite passes 185 tests;
- the complete repository passes 1,544 tests plus the generated audio-preparation
  smoke test;
- the built-in Phase 6 acceptance suite passes all 13 scenarios; and
- no Phase 5 or Phase 6 production path requires network access or an ML runtime.

## The pipeline in common terms

The complete flow covered by these phases is:

```text
accepted speech from Phase 4
        |
        v
verify transcript and consent provenance
        |
        v
normalize words while remembering original positions
        |
        v
match versioned keywords, phrases, and negations
        |
        v
apply bounded context safeguards
        |
        v
produce privacy-minimized language evidence
        |
        +-----------------------------+
                                      |
acoustic evidence + speech evidence --+--> verify all evidence belongs together
                                               |
                                               v
                                      create compact risk inputs
                                               |
                                               v
                                      apply bounded scoring rules
                                               |
                                               v
                                  score + severity + reasons + review flag
```

In plain language, Phase 5 creates a careful receipt for what the accepted words
suggest. Phase 6 takes that receipt and the earlier acoustic and speech receipts,
checks that they describe the same authorized audio, and calculates a transparent
triage score.

## Phase 5 goals

Phase 5 had five tasks:

| Task | Purpose | Main result |
| --- | --- | --- |
| A5.1 | Define language categories and evidence | Strict language-evidence contract and JSON Schema |
| B5.1 | Define versioned language rules | 59 pinned English keyword, phrase, and negation rules |
| A5.2 | Implement transcript analysis | Deterministic local analyzer with exact spans and provenance |
| B5.2 | Create labeled fixtures | 74 synthetic transcripts with 75 expected findings |
| A5.3 | Reduce false positives | Bounded hypothetical, reported-speech, and ambiguity safeguards |

## A5.1: Language-evidence contract

The contract was defined before implementing matching. This prevents the analysis
algorithm from inventing convenient output fields after the fact.

The contract has six categories:

| Category | Common meaning |
| --- | --- |
| `no_concerning_match` | None of the configured rules matched. This is not proof of safety. |
| `distress` | The accepted words contain a configured distress indicator. |
| `threat` | The accepted words contain a configured threatening indicator. |
| `weapon_reference` | The accepted words contain a configured weapon reference. |
| `ambiguous` | Something concerning matched, but the available wording is too unclear. |
| `context_suppressed` | Something matched, but nearby wording shows negation, hypothetical framing, or quotation/reporting. |

Each finding also carries typed reason codes such as `keyword_match`,
`phrase_match`, `explicit_negation`, `quoted_or_reported_speech`, and
`insufficient_context`. These reason codes stop a downstream component from seeing
only the word "threat" without seeing why it received that label.

Only transcripts marked `accepted` by Phase 4 can enter automatic language
analysis. Review-required, rejected, failed, or missing transcripts cannot be
represented as valid automatic Phase 5 inputs.

The evidence stores transcript hashes, character counts, byte counts, timing,
match spans, rule IDs, and hashes of matched text. It does not copy full transcript
text into the Phase 5 artifact. This reduces repeated sensitive data while keeping
the result verifiable against the authorized source.

### Why contract-first design was used

The alternatives were to return loose dictionaries or let the matching engine
define output informally. Those approaches are quicker initially, but they make it
easy for fields, meanings, and privacy boundaries to drift.

Pydantic contracts and JSON Schema were chosen because they provide:

- runtime validation in Python;
- portable validation for non-Python consumers;
- rejection of unknown fields;
- exact numeric and string bounds;
- immutable records after validation; and
- generated, reviewable schemas rather than prose-only promises.

## B5.1: Versioned language-rule data

The first English rule artifact contains 59 rules:

- 17 single-word keyword rules;
- 23 multi-word phrase rules; and
- 19 explicit-negation rules.

Active rules cover distress, threats, and weapon references. Negation rules never
assign an active category; they can only help suppress an otherwise concerning
match.

The rule file is local JSON with a fixed ID, semantic version, format version, and
SHA-256 digest. The loader rejects changed bytes, duplicate IDs, duplicate patterns,
bad ordering, malformed tokens, missing category coverage, incompatible reason
codes, unknown fields, oversized files, or a changed rule-set identity.

### Why rules are token arrays instead of executable regular expressions

Each pattern is stored as normalized tokens such as:

```json
{"kind": "phrase", "tokens": ["has", "a", "gun"]}
```

Configuration-provided regular expressions were not used because they would be
harder to audit, could introduce pathological execution behavior, and would blur
the difference between data and executable matching logic. Token arrays are small,
portable, and understandable during review.

## A5.2: Transcript-analysis algorithm

The analyzer performs the following steps for every accepted transcript.

### 1. Revalidate the Phase 4 handoff

The complete speech artifact is validated again. Its semantic identifier is
recomputed, and the downstream transcript inventory is reconstructed from segments
whose policy result is `accepted` and whose text is allowed downstream.

The supplied downstream tuple must match this reconstructed inventory exactly,
including order, text, timing, confidence, and confidence type. Injecting,
removing, changing, or reordering text causes the entire operation to fail before
matching.

This was chosen over trusting a caller-provided text list because the caller could
otherwise bypass consent or reliability decisions made in Phase 4.

### 2. Normalize text without losing original positions

The algorithm applies:

1. Unicode NFKC normalization;
2. Unicode case folding;
3. conversion of curly apostrophes to the ASCII apostrophe; and
4. English token extraction using `[a-z]+(?:'[a-z]+)?`.

While normalized characters are produced, the analyzer retains a mapping back to
their original character ranges. A rule can therefore match normalized text while
the evidence still points to the exact original substring.

NFKC plus case folding was preferred over simple lowercase conversion because it
handles more equivalent Unicode forms consistently. ASCII stripping was rejected
because it can silently change text and destroy the location of the original
evidence.

### 3. Index rules by their first token

Rules are grouped by their first token. At each transcript token, the analyzer
checks only rules that could start there. Rules sharing a first token are ordered
with longer patterns first.

For this 59-rule baseline, this is simpler and more inspectable than building a
trie, Aho-Corasick automaton, or semantic search index. Its approximate work is:

```text
number of transcript tokens
  x candidate rules sharing the current first token
  x candidate pattern length
```

A global rule-check budget prevents unexpectedly large work. A more sophisticated
multi-pattern automaton would become attractive if the rule inventory grew by
orders of magnitude, but it would add complexity without meaningful benefit here.

### 4. Match exact token sequences

A candidate matches only when its complete token sequence equals the corresponding
transcript tokens. This prevents partial-word mistakes such as matching `bomb`
inside `bombastic` or `rifle` inside `rifleman`.

For every match, the analyzer records:

- the rule ID and kind;
- original start and end character offsets; and
- SHA-256 of the exact original matched text.

### 5. Prefer the most specific same-category match

If a longer phrase fully contains a shorter match from the same category, the
longer phrase wins. For example, a configured phrase like `please help me` prevents
the contained `help` keyword from creating duplicate evidence.

All-match output was rejected because it would inflate finding counts and make a
single phrase look like multiple independent indicators. The algorithm still keeps
separate occurrences and overlapping matches from different categories when they
carry genuinely different information.

### 6. Apply bounded explicit negation

A negation can suppress a concerning match only when it:

- ends before the concerning match begins;
- is no more than three intervening tokens away; and
- does not cross `.`, `!`, `?`, `;`, `:`, or a newline.

The nearest eligible negation wins. If two are equally near, the longer negation
phrase wins, followed by stable rule-ID ordering.

The system does not use global negation. A `not` elsewhere in a paragraph cannot
erase an unrelated finding. It also does not let a negation token inside a
configured phrase suppress that phrase; `I cannot breathe` remains an active
distress phrase.

A full dependency parser was not used because the project has no validated parser
model or linguistic benchmark for this domain. The three-token bounded rule is
conservative, auditable, and easy to test, though less linguistically flexible.

### 7. Apply bounded context safeguards

When explicit negation does not apply, the analyzer checks the same hard-bounded
clause for narrowly defined cues:

- modal or conditional speech framing, such as a character who "might say" the
  concerning words;
- quoted or reported speech framing, such as a report or statement that "said"
  the words; and
- two limited ambiguity patterns: a keyword-only modal question with no object, or
  a keyword mentioned only as a discussion topic.

Hypothetical and reported wording becomes `context_suppressed`. Insufficient
context becomes `ambiguous`. Direct phrases and keywords with meaningful trailing
objects are preserved. Explicit negation has higher precedence than these cues.

These safeguards were chosen over broad intent inference. A large language model
could recognize more paraphrases, sarcasm, and discourse context, but it would also
introduce model downloads, nondeterminism, harder provenance, more private text
exposure, and an evaluation requirement the project has not yet satisfied.

### 8. Produce deterministic evidence

If no active rule remains, the transcript receives one `no_concerning_match`
finding. Otherwise, findings are sorted deterministically by location, category,
and content-derived ID.

Finding IDs and the final evidence ID are SHA-256 hashes of canonical JSON content.
The audit timestamp is deliberately excluded from identity, so rerunning the same
inputs and rules produces the same evidence ID at a different time.

### 9. Enforce resource limits and safe failures

Default limits are:

| Resource | Limit |
| --- | ---: |
| Accepted transcripts | 100,000 |
| Total transcript UTF-8 bytes | 1 MiB |
| Total normalized tokens | 250,000 |
| Indexed rule checks | 10,000,000 |
| Findings | 100,000 |

Exceeded limits produce stable error codes without echoing transcript content and
without returning partial evidence.

## B5.2 and A5.3: Fixtures and false-positive safeguards

B5.2 created a SHA-256-pinned answer key containing 74 synthetic transcripts and
75 expected findings:

| Purpose | Count |
| --- | ---: |
| Active indicator | 43 |
| Context suppressed | 23 |
| No concerning match | 6 |
| Ambiguous | 2 |

Every active keyword and phrase rule appears in an active fixture. Every negation
rule appears in a harmless suppression fixture. The fixture loader validates full
coverage so accidentally deleting a case cannot quietly reduce the benchmark.

A5.3 then ran all 74 fixtures through the real analyzer and compared categories,
reason codes, supporting rule IDs, and finding order. It added the bounded
hypothetical, quotation/reporting, and ambiguity logic described above. Additional
tests verify clause boundaries, direct questions with objects, and explicit
negation precedence.

Synthetic fixtures were used instead of real transcripts because they are safe to
check into source control, have unambiguous expected outcomes, cover every rule,
and contain no personal data. They are excellent regression tests, but they do not
measure real-world language accuracy or demographic bias.

## Phase 5 algorithm choice summary

| Considered approach | Decision | Reason |
| --- | --- | --- |
| Versioned token rules | Selected | Fully local, deterministic, explainable, and appropriate without a labeled training corpus |
| Raw regular expressions in configuration | Rejected | Harder to audit and bound; configuration would behave like executable logic |
| Stemming or fuzzy matching | Deferred | Improves recall but can create unpredictable false positives and span ambiguity |
| Embedding similarity | Deferred | Requires a model and calibrated similarity thresholds; explanations are weaker |
| Transformer text classifier | Deferred | Requires representative labeled data, model/runtime governance, and bias evaluation |
| External LLM API | Rejected for this phase | Conflicts with offline/privacy goals and makes exact reproducibility difficult |
| Full syntactic/dependency parser | Deferred | More linguistic power, but no validated parser baseline or domain benchmark yet |

## Phase 5 limitations

The language analyzer is English-only and intentionally narrow. It may miss:

- synonyms and paraphrases absent from the rule file;
- spelling errors, slang, code words, and multilingual speech;
- long-range negation or context outside the bounded clause;
- sarcasm and indirect intent; and
- meaning that depends on speaker, conversation history, or the real-world scene.

Conversely, broad words such as `help`, `shoot`, `bomb`, or `knife` can still be
false positives outside the covered safeguards. This is why findings remain
evidence rather than incident decisions.

## Phase 6 goals

Phase 6 had six tasks:

| Task | Purpose | Main result |
| --- | --- | --- |
| A6.1 | Define risk inputs and outputs | Typed 0-100 assessment contract and missing-data policy |
| B6.1 | Implement scoring | Versioned, configurable, bounded deterministic scorer |
| A6.2 | Integrate prior evidence | Cross-branch provenance checks and compact risk inputs |
| B6.2 | Test behavior and edges | 54-scenario matrix and numeric overflow safeguard |
| A6.3 | Prepare for future training | Privacy-minimized feature and trainer/predictor interfaces |
| A6.4 | Validate the built-in policy | Pinned 13-case acceptance suite and report |

## A6.1: Risk-assessment contract

The risk contract accepts summaries from three branches:

- acoustic event candidates from Phase 3;
- speech presence and transcript-reliability counts from Phase 4; and
- language findings from Phase 5.

Every branch has an explicit status:

| Status | Meaning |
| --- | --- |
| `present` | Valid hash-pinned evidence is available. |
| `missing` | Evidence was expected but is unavailable. |
| `not_permitted` | Consent does not allow this branch. |
| `not_applicable` | The branch does not apply. |
| `no_accepted_text` | Speech ran, but no transcript was accepted for language analysis. |

This distinction is crucial. Missing evidence is uncertainty and can require human
review. Evidence forbidden by consent is not mislabeled as a technical failure.

The score range is fixed at 0-100 with these severity bands:

| Score | Severity |
| ---: | --- |
| 0 | None |
| greater than 0 through 24 | Low |
| 25 through 49 | Medium |
| 50 through 74 | High |
| 75 through 100 | Critical |

The contract validates counts, sample spans, timestamps, evidence kinds, canonical
ordering, score/severity agreement, and missing-data summaries. Unknown fields and
non-finite numbers are rejected.

## A6.2: Evidence-integration algorithm

`integrate_risk_inputs()` is a trust boundary, not a scorer. Its algorithm is:

1. validate the trusted risk source;
2. revalidate every supplied acoustic, speech, and language artifact;
3. verify clip, consent, sample-rate, sample-count, prepared-manifest, and raw-audio
   provenance across branches;
4. verify that language evidence references the exact speech evidence and accepted
   transcripts;
5. reduce each branch to the fields needed for scoring; and
6. assign explicit missing, consent-limited, or no-accepted-text states.

The adapter hashes each complete normalized evidence document and places that hash
in the compact risk input. It does not copy raw audio, transcript text, matched
text, model tensors, speaker data, or absolute paths.

This adapter was preferred over having the scorer read all upstream documents
directly. Separating integration from scoring keeps provenance verification,
privacy reduction, and arithmetic independently testable.

## B6.1: Deterministic risk-scoring algorithm

The score is an additive, independently capped weighted policy:

```text
final score = min(
    100,
    acoustic branch score
    + speech branch score
    + language branch score
)
```

### Acoustic branch

```text
acoustic score = min(
    60,
    maximum configured weight among positive acoustic events
    + 10 if any positive event confidence is at least 0.85
)
```

Built-in positive weights are:

| Event | Points |
| --- | ---: |
| Siren | 10 |
| Smoke alarm | 15 |
| Glass break | 25 |
| Crowd panic | 30 |
| Distress speech | 30 |
| Threatening speech | 40 |
| Weapon reference | 40 |
| Explosion | 40 |
| Gunshot | 45 |

Ambient, no-speech, speech-present, and non-threatening-speech labels are fixed at
zero.

The maximum event weight was selected instead of summing acoustic events because
overlapping windows and related detections can describe the same sound. Summing
them would let duplicate detections inflate risk. The confidence bonus preserves a
small distinction for a strong candidate without turning model confidence directly
into an unbounded score.

### Speech branch

```text
speech score = min(
    12,
    2 if any speech segment exists
    + min(10, 5 x review-required transcript count)
)
```

Speech presence contributes only a small amount. Uncertain transcripts contribute
more because they need attention, but the branch cap prevents repeated uncertainty
from dominating the assessment.

The implementation checks whether a transcript count reaches the cap using exact
integer ratios before floating-point multiplication. This prevents overflow even
for a contract-valid count such as `10**400` or a subnormal custom weight.

### Language branch

```text
language score = min(50, sum of configured finding weights)
```

Built-in weights are:

| Category | Points |
| --- | ---: |
| Distress | 25 |
| Threat | 40 |
| Weapon reference | 35 |
| Ambiguous | 5 |
| No concerning match | 0 |
| Context suppressed | 0 |

Language findings are summed because distinct accepted transcripts may contain
separate evidence. The 50-point cap prevents repeated findings from growing without
bound. Protected no-match and context-suppressed categories are contractually fixed
at zero, so a configuration edit cannot silently turn them into risk evidence.

### Human-review algorithm

Human review is required when any of these is true:

- total score is at least 25;
- at least one transcript is review-required;
- language evidence is ambiguous; or
- any expected evidence branch is missing.

Missing branches add no points. They create branch-specific reason codes and a
missing-data review reason. This avoids the dangerous shortcut of interpreting
"no data" as "safe."

### Deterministic output

Scores are rounded once to six decimal places. Reason codes use a fixed canonical
order. The assessment ID hashes the scoring policy, compact inputs, score,
severity, reasons, missing-data state, and review decision. The audit timestamp is
not part of identity.

## Why a capped weighted score was chosen

The main alternatives were a learned classifier, logistic regression, Bayesian
inference, fuzzy logic, or an opaque expert-system implementation.

The capped weighted policy was selected because:

- the project does not yet have a representative labeled end-to-end incident
  dataset;
- every point can be explained to a reviewer;
- weights and thresholds are versioned data rather than hidden code;
- branch caps prevent one noisy source from dominating;
- missing data and consent can be modeled explicitly; and
- the scorer remains local, fast, deterministic, and easy to reproduce.

The weights are engineering policy values, not statistically calibrated incident
probabilities. A score of 75 does not mean a 75 percent chance of an incident.

## B6.2: Scenario and edge testing

B6.2 added 54 public-API scenarios covering:

- all 13 acoustic labels;
- all six language categories;
- all five severity levels;
- confidence immediately below, exactly at, and above 0.85;
- severity and review boundaries;
- acoustic maximum aggregation;
- speech, language, and total caps;
- missing and consent-related states;
- deterministic reason ordering;
- fractional custom rules; and
- extremely large counters.

These tests found and fixed the large-count floating-point overflow risk described
above. Testing the public scorer instead of private helpers proves that validation,
rule loading, scoring, reasons, and output contracts work together.

## A6.3: Future trainable-model interface

Phase 6 did not train a custom model. It created a stable interface so a future
model can be added without redesigning the data boundary.

The feature extractor produces an identity-free vector containing:

- branch availability statuses;
- acoustic counts and maximum confidence for every canonical event label;
- speech, accepted-transcript, and review-required counts plus maximum VAD score;
  and
- language counts for every canonical category.

It excludes clip and consent IDs, evidence IDs, transcript text, matched text,
speaker identity, raw audio, paths, and tensors.

Training examples require a target labeled `human_reviewed` or `adjudicated`.
Descriptors record model version, runtime version, artifact hash, training-dataset
hash, and training/validation counts. Runtime-checkable Python protocols define
`train()` and `predict()` without selecting TensorFlow, PyTorch, ONNX Runtime, or a
specific algorithm.

This interface-first approach was chosen to preserve future flexibility while
avoiding an unvalidated model today. Logistic regression, gradient boosting, or a
small neural network could later implement the protocol after suitable data,
splits, calibration, and bias evaluation exist.

## A6.4: Executable policy validation

A6.4 added a pinned 13-case acceptance suite. Each compact scenario is expanded
through the real A6.1 input contract and passed to the public scorer. The report
compares expected and observed score, severity, review decision, reasons, and
missing branches.

The suite covers all severity levels, the inclusive 0.85 confidence boundary, the
score-25 review boundary, branch and total caps, transcript uncertainty, ambiguity,
context suppression, missing data, acoustic-only consent, and no accepted text.

All 13 scenarios pass. The generated report records:

- suite and rule SHA-256 hashes;
- exact threshold and cap values;
- expected and observed outcomes;
- pass/fail counts; and
- a deterministic validation ID.

The report explicitly says
`synthetic_acceptance_validation_not_empirical_incident_calibration`. This result
validates software and policy behavior, not real-world accuracy.

## Phase 6 algorithm choice summary

| Considered approach | Decision | Reason |
| --- | --- | --- |
| Capped weighted deterministic rules | Selected | Transparent, bounded, configurable, offline, and testable without claiming statistical calibration |
| Sum all acoustic detections | Rejected | Overlapping or related detections could multiply one event's effect |
| Maximum acoustic weight | Selected | Limits duplicate inflation while retaining the strongest candidate |
| Sum language findings with a cap | Selected | Separate transcripts can add evidence, while the cap controls repetition |
| Missing data equals zero risk | Rejected | Absence of evidence is uncertainty, not proof of safety |
| Learned risk classifier now | Deferred | No suitable labeled end-to-end training and calibration dataset exists yet |
| Model-specific training API | Rejected | Would lock the project to a framework before an algorithm is selected |
| Framework-neutral protocols | Selected | Allows later implementations while preserving data and provenance requirements |

## Technology stack

### Python 3.11+

Python is the main implementation language. It was selected because the rest of the
audio pipeline already uses Python, its standard library provides reliable Unicode,
hashing, JSON, packaging, typing, and date/time tools, and it has a mature path to
future ML work.

Using one language across audio, speech, language, risk, tests, scripts, and API
layers reduces integration overhead. A lower-level language could be faster, but
Phase 5 and Phase 6 workloads are small compared with audio-model inference and do
not justify the added implementation complexity.

### Pydantic 2

Pydantic provides immutable typed records, strict bounds, enum validation,
cross-field validators, unknown-field rejection, JSON serialization, and JSON
Schema generation.

Dataclasses alone would not provide equivalent runtime validation or portable
schemas. Ad hoc dictionary checks would be more error-prone and harder to maintain.

### JSON and JSON Schema

Rules, fixtures, examples, scoring policy, validation scenarios, and reports use
JSON. JSON is human-reviewable, language-neutral, easy to hash, and supported by
the existing project tooling. JSON Schema documents the public exchange format and
lets other implementations validate the same records.

A database was unnecessary because these artifacts are small, versioned, static,
and intended to live beside the code. YAML was not used for the security-sensitive
rule artifacts because JSON has a smaller, more predictable data model and direct
Pydantic support.

### SHA-256 and canonical JSON

`hashlib.sha256` pins exact artifact bytes and creates deterministic semantic IDs
from canonical JSON. SHA-256 was chosen because it is widely available, has strong
collision resistance for this use, and is suitable for integrity and provenance.

These hashes prove content identity, not authorship. They are not digital
signatures and do not replace access control or release signing.

### Python standard library

The phases rely heavily on:

- `unicodedata` for NFKC normalization;
- `re` for bounded token scanning and identifier checks;
- `hashlib` for SHA-256;
- `json` for canonical serialization;
- `importlib.resources` for local package artifacts;
- `datetime` with UTC-aware timestamps;
- `dataclasses` for small immutable loaded-artifact wrappers; and
- `typing.Protocol` for framework-neutral future model interfaces.

Using standard-library components keeps the runtime small and avoids unnecessary
dependencies.

### pytest 8

Pytest supplies parameterized boundary matrices, fixtures, temporary directories,
subprocess isolation, and readable failure reporting. It was selected over only
script-based checks because the phases contain many contracts and edge conditions
that need repeatable regression coverage.

Current focused coverage is 325 Phase 5 tests and 185 Phase 6 tests.

### Git and pinned package resources

Rule and fixture JSON files are package resources committed with the code. Git
provides review history, while fixed SHA-256 constants make unexpected byte changes
fail at runtime and in tests. `.gitattributes` protects hash-sensitive fixture
bytes from line-ending conversion where needed.

### Optional project technologies not used here

The wider project includes FastAPI/Uvicorn for a local demonstration API and
optional TensorFlow, PyTorch, ONNX Runtime, Faster-Whisper, and related packages for
earlier or future model work. The Phase 5 and Phase 6 core deliberately does not
import those ML runtimes.

That separation provides three benefits:

- language and risk tests run quickly on an ordinary development environment;
- loading these modules cannot trigger a model download or network request; and
- policy behavior remains reproducible independently of accelerator, model, or
  runtime differences.

## Privacy, safety, and integrity controls

Across both phases, the implementation uses several layers of protection:

1. Consent and reliability are checked before text analysis.
2. Upstream artifacts are revalidated rather than trusted by type alone.
3. Cross-branch IDs, hashes, timing, and source metadata must agree.
4. Full transcript and matched text are not copied into downstream artifacts.
5. Unknown fields and non-finite values are rejected.
6. Rule, fixture, scoring, and validation artifacts are versioned and hash-pinned.
7. Outputs and inventory order are deterministic.
8. Resource limits prevent unbounded input and rule processing.
9. Stable errors avoid echoing private transcript content.
10. Missing data requires review instead of silently lowering concern.
11. Context-suppressed language is protected at zero risk weight.
12. Risk output remains separate from consensus and alert decisions.

## Worked example

Suppose an authorized clip produces:

- one high-confidence explosion candidate at confidence 0.93;
- speech with one accepted transcript and one review-required transcript; and
- one accepted distress-language finding.

Phase 5 records the distress finding, its supporting rule and span, and the hash of
the accepted transcript without copying the transcript text into the evidence.

Phase 6 calculates:

```text
acoustic: explosion 40 + high-confidence bonus 10 = 50
speech:   speech present 2 + one review-required transcript 5 = 7
language: one distress finding = 25
total:    50 + 7 + 25 = 82
```

The result is `critical` because 82 is in the 75-100 band. Human review is required
because the score is at least 25 and a transcript is review-required. The result is
still a risk assessment, not a confirmed emergency or automatic alert.

## Testing and verification summary

| Area | Current focused result |
| --- | ---: |
| Phase 5 language contracts, rules, analyzer, and fixtures | 325 passed |
| Phase 6 contracts, scoring, integration, scenarios, model interface, validation | 185 passed |
| A6.4 acceptance scenarios | 13 passed, 0 failed |
| Complete repository at Phase 6 completion | 1,544 passed |
| Generated preparation smoke test | Passed |

Important test themes include malformed artifacts, checksum drift, ordering,
unknown fields, exact threshold boundaries, privacy exclusions, deterministic IDs,
resource limits, missing branches, consent states, huge counters, offline imports,
and operation without ML runtimes.

## Main files

### Phase 5

- `src/audio_sentinel/language_contracts.py`: language evidence contracts.
- `src/audio_sentinel/language_rules.py`: rule contracts and pinned loader.
- `src/audio_sentinel/language_analysis.py`: matching and context algorithm.
- `src/audio_sentinel/language_fixtures.py`: labeled fixture contracts and loader.
- `src/audio_sentinel/resources/language-rules-en-v1.json`: 59 rules.
- `src/audio_sentinel/resources/language-fixtures-en-v1.json`: 74 fixtures.
- `docs/schemas/v1/language-evidence.schema.json`: public evidence schema.
- `docs/schemas/v1/language-rule-set.schema.json`: rule schema.
- `docs/schemas/v1/language-fixture-set.schema.json`: fixture schema.

### Phase 6

- `src/audio_sentinel/risk_contracts.py`: risk input/output contracts.
- `src/audio_sentinel/risk_integration.py`: evidence provenance and reduction.
- `src/audio_sentinel/risk_scoring.py`: deterministic scoring algorithm.
- `src/audio_sentinel/risk_model.py`: future trainable-model interface.
- `src/audio_sentinel/risk_validation.py`: acceptance validation runner.
- `src/audio_sentinel/resources/risk-scoring-rules-v1.json`: built-in scoring policy.
- `src/audio_sentinel/resources/risk-validation-scenarios-v1.json`: acceptance suite.
- `outputs/a6_4_validation/risk_score_validation.json`: passing validation report.
- `scripts/validate_risk_policy.py`: report reproduction CLI.

## What remains for later phases

Phase 5 and Phase 6 establish trustworthy mechanics, but they do not finish the
product. Later work still needs to:

- define and implement consensus across evidence and risk outputs;
- decide when to do nothing, log, request review, or create an alert;
- evaluate complete end-to-end behavior on representative labeled data;
- measure false-positive and false-negative rates;
- study subgroup, environment, language, and recording-condition bias;
- calibrate thresholds against the real operational cost of errors;
- document retention, deletion, authorization, and human-review procedures; and
- validate any future learned model before it can replace or supplement the
  deterministic scorer.

## Glossary

**Artifact:** A saved, versioned data file such as a rule set, fixture set, evidence
document, or validation report.

**Canonical JSON:** JSON serialized in one stable order and format so identical
content always produces identical bytes for semantic hashing.

**Contract:** A strict definition of allowed fields, values, relationships, and
validation rules between parts of the system.

**Evidence:** Structured observations that may support later decisions but do not
themselves confirm an incident.

**Finding:** One Phase 5 language result linked to a rule match and explanation.

**Hash:** A fixed-length content fingerprint. Changing the content changes the
fingerprint with overwhelming probability.

**Human review:** A requirement that a person inspect uncertain, incomplete, or
sufficiently concerning evidence before later action.

**NFKC:** A Unicode normalization form that converts compatibility-equivalent text
to a more consistent representation.

**Provenance:** Information proving where data came from and which exact source or
rule artifact produced it.

**Risk score:** A bounded policy value used for triage. It is not a probability and
not an incident verdict.

**Synthetic fixture:** A deliberately written test example with a known expected
result, containing no real personal or incident data.

## Conclusion

Phase 5 and Phase 6 produced a local, explainable, privacy-conscious bridge from
accepted speech to risk triage. The system favors evidence receipts, strict
provenance, bounded rules, and human review over opaque certainty.

The chosen algorithms are intentionally modest. They are strong at reproducibility,
auditability, and safe failure, and weak at broad semantic understanding and
real-world statistical calibration. That tradeoff is appropriate for the current
data and maturity of the project. The architecture leaves room for learned models
later, but only behind the same contracts, provenance, privacy, and validation
requirements established here.
