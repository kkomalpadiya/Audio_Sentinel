"""Run the A10.3 in-process live-path performance measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.live_performance import (  # noqa: E402
    run_in_process_live_benchmark,
    save_live_performance_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "a10_3_measurement"
            / "live-performance-report.json"
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing report at --output.",
    )
    arguments = parser.parse_args()
    report = run_in_process_live_benchmark()
    save_live_performance_report(
        report,
        arguments.output,
        replace=arguments.replace,
    )
    p95 = {item.kind.value: item.p95_ms for item in report.latency}
    print(
        json.dumps(
            {
                "report_id": report.report_id,
                "measurement_status": report.measurement_status,
                "chunks_accepted": report.reliability.counters.chunks_accepted,
                "windows_processed": report.reliability.counters.windows_processed,
                "local_alert_probe_completions": (
                    report.reliability.counters.local_alert_probe_completions
                ),
                "p95_ms": p95,
                "notification_delivery": report.notification_delivery,
                "output": str(arguments.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
