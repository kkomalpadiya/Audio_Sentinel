"""Run the A7.4 deterministic consensus-pipeline acceptance validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.consensus_validation import (  # noqa: E402
    save_consensus_validation_report,
    validate_builtin_consensus_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "a7_4_validation"
            / "consensus_pipeline_validation.json"
        ),
    )
    arguments = parser.parse_args()
    report = validate_builtin_consensus_pipeline()
    save_consensus_validation_report(report, arguments.output)
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
