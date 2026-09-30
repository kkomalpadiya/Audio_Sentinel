"""Build the pinned A9.1 held-out end-to-end evaluation manifest."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.consensus_contracts import ConsensusOutcome  # noqa: E402
from audio_sentinel.evaluation_manifest import (  # noqa: E402
    EvaluationManifestCase,
    build_evaluation_manifest,
)
from audio_sentinel.evaluation_service import (  # noqa: E402
    EvaluationRequest,
    EvaluationScope,
)


SOURCE_REPORT = (
    ROOT / "outputs" / "a3_4_evaluation" / "acoustic_detection_evaluation.json"
)
SOURCE_REPORT_SHA256 = (
    "f1d805d0977c6d1931d907770d5fbd797dfb706e4e80aa16b4764e5cb8a5c5c9"
)
SOURCE_EVALUATION_ID = "a3-4-92838b4c8910027e4ffc00ad"
SELECTION_SEED = "audio-sentinel-a9.1-heldout-v1"
NEGATIVE_CATEGORIES_PER_DATASET = 8
ACOUSTIC_THRESHOLD = 0.5
CREATED_AT = datetime(2026, 9, 30, tzinfo=UTC)
TARGET_TRUTH_FIELDS = ("siren", "glass_break", "gunshot")


def _rank(*parts: object) -> str:
    value = "|".join((SELECTION_SEED, *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_source_report(path: Path) -> dict[str, object]:
    document = path.read_bytes()
    if hashlib.sha256(document).hexdigest() != SOURCE_REPORT_SHA256:
        raise ValueError("A9.1 source evaluation report checksum mismatch")
    report = json.loads(document)
    if report.get("evaluation_id") != SOURCE_EVALUATION_ID:
        raise ValueError("A9.1 source evaluation identity mismatch")
    if report.get("decision_status") != "research_baseline_not_production_calibration":
        raise ValueError("A9.1 source evaluation status mismatch")
    return report


def _is_positive(sample: dict[str, object]) -> bool:
    truth = sample.get("truth")
    if not isinstance(truth, dict):
        raise ValueError("source sample is missing truth labels")
    return any(truth.get(field) is True for field in TARGET_TRUTH_FIELDS)


def _select_samples(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    source_samples = report.get("samples")
    if not isinstance(source_samples, list):
        raise ValueError("source evaluation is missing samples")
    holdout = [
        sample
        for sample in source_samples
        if isinstance(sample, dict) and sample.get("split") == "holdout"
    ]
    positives = [sample for sample in holdout if _is_positive(sample)]
    negatives = [sample for sample in holdout if not _is_positive(sample)]
    if len(positives) != 16:
        raise ValueError("A9.1 expects all 16 held-out target-positive clips")

    selected_negatives: list[dict[str, object]] = []
    datasets = sorted({str(sample["dataset_id"]) for sample in negatives})
    if datasets != ["esc50", "urbansound8k"]:
        raise ValueError("A9.1 expects ESC-50 and UrbanSound8K holdout negatives")
    for dataset_id in datasets:
        by_category: dict[str, list[dict[str, object]]] = {}
        for sample in negatives:
            if sample["dataset_id"] == dataset_id:
                by_category.setdefault(str(sample["category"]), []).append(sample)
        category_names = sorted(
            by_category,
            key=lambda category: (_rank("category", dataset_id, category), category),
        )[:NEGATIVE_CATEGORIES_PER_DATASET]
        if len(category_names) != NEGATIVE_CATEGORIES_PER_DATASET:
            raise ValueError("A9.1 source lacks enough negative categories")
        for category in category_names:
            selected_negatives.append(
                min(
                    by_category[category],
                    key=lambda sample: (
                        _rank("clip", sample["relative_audio_path"]),
                        str(sample["relative_audio_path"]),
                    ),
                )
            )
    selected = positives + selected_negatives
    if len(selected) != 32 or len({str(item["relative_audio_path"]) for item in selected}) != 32:
        raise ValueError("A9.1 selection must contain 32 unique clips")
    return tuple(
        sorted(
            selected,
            key=lambda sample: (
                str(sample["dataset_id"]),
                str(sample["category"]),
                str(sample["relative_audio_path"]),
            ),
        )
    )


def _case(sample: dict[str, object]) -> EvaluationManifestCase:
    dataset_id = str(sample["dataset_id"])
    category = str(sample["category"])
    relative_path = str(sample["relative_audio_path"])
    positive = _is_positive(sample)
    suffix = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    case_id = f"a9-1-{dataset_id}-{category}-{suffix}"
    return EvaluationManifestCase(
        case_id=case_id,
        request=EvaluationRequest(
            audio_path=relative_path,
            clip_id=f"{case_id}-clip",
            consent_id="a9-1-authorized-research-dataset",
            processing_scope=EvaluationScope.ACOUSTIC_ONLY,
            device_authorized=True,
            granted_at=CREATED_AT,
            source_dataset="a3-4-heldout-esc50-urbansound8k-v1",
            acoustic_threshold=ACOUSTIC_THRESHOLD,
        ),
        expected_positive=positive,
        categories=(
            dataset_id,
            category,
            "split_holdout",
            "target_positive" if positive else "background_negative",
        ),
    )


def build_manifest(source_report: Path = SOURCE_REPORT):
    report = _read_source_report(source_report)
    cases = tuple(_case(sample) for sample in _select_samples(report))
    return build_evaluation_manifest(
        "A9.1 held-out end-to-end evaluation",
        cases,
        positive_outcomes=(
            ConsensusOutcome.LOG,
            ConsensusOutcome.REVIEW,
            ConsensusOutcome.ALERT,
        ),
        now=CREATED_AT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs" / "a9-1-evaluation-manifest.json",
    )
    arguments = parser.parse_args()
    manifest = build_manifest()
    document = (
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.exists():
        if arguments.output.read_bytes() != document:
            raise FileExistsError("A different evaluation manifest already exists")
        reused = True
    else:
        arguments.output.write_bytes(document)
        reused = False
    print(
        json.dumps(
            {
                "manifest_id": manifest.manifest_id,
                "case_count": len(manifest.cases),
                "positive_count": sum(case.expected_positive for case in manifest.cases),
                "negative_count": sum(not case.expected_positive for case in manifest.cases),
                "output": str(arguments.output),
                "sha256": hashlib.sha256(document).hexdigest(),
                "reused": reused,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
