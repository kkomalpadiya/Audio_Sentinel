"""Run the A6.4 deterministic risk-policy acceptance validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.risk_validation import (  # noqa: E402
    save_risk_validation_report,
    validate_builtin_risk_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "a6_4_validation"
            / "risk_score_validation.json"
        ),
    )
    arguments = parser.parse_args()
    report = validate_builtin_risk_policy()
    save_risk_validation_report(report, arguments.output)
    print(
        json.dumps(
            {
                "validation_id": report.validation_id,
                "decision_status": report.decision_status,
                "scenario_count": report.scenario_count,
                "passed_count": report.passed_count,
                "failed_count": report.failed_count,
                "output": str(arguments.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
