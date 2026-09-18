"""Run the A3.4 labeled-sample benchmark with the isolated YAMNet runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.acoustic_evaluation import evaluate_labeled_samples, save_evaluation_report  # noqa: E402
from audio_sentinel.acoustic_loader import load_yamnet  # noqa: E402
from audio_sentinel.config import load_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acknowledge-noncommercial-dataset-licenses",
        action="store_true",
        help="Confirm this local evaluation complies with the ESC-50 and UrbanSound8K licenses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "a3_4_evaluation" / "acoustic_detection_evaluation.json",
    )
    arguments = parser.parse_args()
    if not arguments.acknowledge_noncommercial_dataset_licenses:
        parser.error("the dataset-license acknowledgement flag is required")
    settings = load_settings(ROOT)
    report = evaluate_labeled_samples(settings.paths.raw_data, load_yamnet(settings.paths))
    save_evaluation_report(report, arguments.output)
    summary = {
        "evaluation_id": report["evaluation_id"],
        "output": str(arguments.output),
        "sample_count": report["sample_count"],
        "holdout": {
            label: details["holdout"] for label, details in report["thresholds"].items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
