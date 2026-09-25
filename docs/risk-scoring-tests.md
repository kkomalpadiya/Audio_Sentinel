# Risk-scoring scenario tests

## What B6.2 adds

B6.2 adds `tests/test_risk_scenarios.py`, a 54-case behavior matrix for the A6.1,
B6.1, and A6.2 scoring boundary. The tests call the public `score_risk()` function
with validated `RiskInputSet` documents. They do not bypass input contracts or
assert against private scoring helpers.

The matrix complements the B6.1 rule-loader and contract tests. It focuses on what
an assessment means when normal and edge-case evidence reaches the scorer.

## Normal scenarios

Five representative scenarios produce every severity level:

| Scenario | Evidence | Score | Severity | Review |
| --- | --- | ---: | --- | --- |
| No risk evidence | Empty present branches | 0 | None | No |
| Low | Siren below the confidence bonus | 10 | Low | No |
| Medium | Threat-language finding | 40 | Medium | Yes |
| High | High-confidence gunshot | 55 | High | Yes |
| Critical | High-confidence explosion, speech, and distress | 77 | Critical | Yes |

The tests separately cover all 13 built-in acoustic labels and all six language
categories. They verify exact weights, reason codes, ambiguity review, and the zero
weight of `no_concerning_match` and `context_suppressed`.

## Boundary and cap coverage

The scenario matrix verifies:

- acoustic confidence immediately below, exactly at, and above 0.85;
- score and review behavior at 1, 24, 25, 49, 50, 74, 75, and 100;
- maximum-weight acoustic aggregation rather than event summation;
- acoustic, speech, language, and final score caps;
- speech presence plus zero, one, two, and three review-required transcripts;
- each independently missing branch and nonmissing language absence states;
- complete, unique, canonically ordered reason codes; and
- one-time rounding of configurable fractional scores to six decimal places.

## Large-count safeguard

The A6.1 contract intentionally permits nonnegative integer counters without an
arbitrary upper bound. B6.2 identified that multiplying an extremely large valid
review-required count by a floating-point weight could overflow before the speech
cap was applied. The scorer now determines whether the count has reached the cap
before multiplying. Exact integer ratios also avoid overflow when a custom rule
uses the smallest positive finite floating-point weight. A count as large as
`10**400` therefore produces the expected 12 speech points and review requirement
instead of an exception in both configurations.

## Verification

Run only the B6.2 matrix:

```powershell
python -m pytest tests/test_risk_scenarios.py -q
```

Run the complete focused Phase 6 suite:

```powershell
python -m pytest tests/test_risk_contracts.py tests/test_risk_scoring.py tests/test_risk_integration.py tests/test_risk_scenarios.py -q
```

Run all project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Beginner-friendly explanation

These tests use the scorer like a table of worked examples. They check ordinary
quiet, low, medium, high, and critical cases, then press on exact boundaries and
unusually large values. That makes the documented numbers executable and catches
failures that a few happy-path examples would miss.
