"""Run one B9.1 manifest through the shared Phase 8 evaluation service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.config import load_settings  # noqa: E402
from audio_sentinel.evaluation_manifest import (  # noqa: E402
    load_evaluation_manifest,
    run_evaluation_manifest,
    save_evaluation_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--expected-sha256",
        help="Optional pinned SHA-256 for the exact manifest bytes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to outputs/b9_1_evaluations/<run_id>.json.",
    )
    arguments = parser.parse_args()

    loaded = load_evaluation_manifest(
        arguments.manifest,
        expected_sha256=arguments.expected_sha256,
    )
    report = run_evaluation_manifest(load_settings(ROOT), loaded)
    output = arguments.output or (
        ROOT / "outputs" / "b9_1_evaluations" / f"{report.run_id}.json"
    )
    reused = save_evaluation_run(report, output)
    summary = {
        "run_id": report.run_id,
        "manifest_id": report.manifest_id,
        "decision_status": report.decision_status,
        "total_case_count": report.metrics.total_case_count,
        "completed_case_count": report.metrics.completed_case_count,
        "failed_case_count": report.metrics.failed_case_count,
        "true_positive": report.metrics.binary.true_positive,
        "false_positive": report.metrics.binary.false_positive,
        "false_negative": report.metrics.binary.false_negative,
        "true_negative": report.metrics.binary.true_negative,
        "output": str(output),
        "reused": reused,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
