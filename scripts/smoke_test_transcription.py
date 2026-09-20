"""Real-model smoke test for the B4.2 offline transcription wrapper."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.config import Paths  # noqa: E402
from audio_sentinel.transcription import (  # noqa: E402
    load_transcription_model,
    transcribe_segment,
)


def main() -> None:
    loaded = load_transcription_model(Paths.from_root(ROOT))
    silence = np.zeros(16_000, dtype=np.float32)
    first = transcribe_segment(loaded, silence)
    second = transcribe_segment(loaded, silence)
    if first != second or first.candidate is not None or first.chunks:
        raise RuntimeError("Pinned tiny.en did not produce the deterministic empty-silence result")
    print(json.dumps({
        "artifact_sha256": loaded.metadata.artifact_sha256,
        "compute_type": loaded.metadata.compute_type,
        "input_num_samples": first.input_num_samples,
        "language": loaded.metadata.language,
        "model_id": loaded.metadata.model_id,
        "transcript": None,
    }, indent=2))


if __name__ == "__main__":
    main()
