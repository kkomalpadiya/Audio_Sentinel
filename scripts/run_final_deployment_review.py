"""Generate the A10.4 privacy, security, and deployment review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.deployment_review import (  # noqa: E402
    build_final_deployment_review,
    save_final_deployment_review,
    verify_review_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "a10_4_review" / "final-deployment-review.json",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing review at --output.",
    )
    arguments = parser.parse_args()
    review = build_final_deployment_review(ROOT)
    changed = verify_review_evidence(review, ROOT)
    if changed:
        raise RuntimeError("Review evidence changed during generation: " + ", ".join(changed))
    save_final_deployment_review(review, arguments.output, replace=arguments.replace)
    print(
        json.dumps(
            {
                "review_id": review.review_id,
                "release_decision": review.release_decision,
                "verified_controls": review.counts.verified_controls,
                "conditional_controls": review.counts.conditional_controls,
                "release_blockers": review.counts.release_blockers,
                "evidence_artifacts": len(review.evidence),
                "production_approved": review.production_approved,
                "output": str(arguments.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
