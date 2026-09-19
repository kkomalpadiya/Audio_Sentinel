"""Real pinned-model structural smoke test for the B4.1 VAD wrapper."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import faster_whisper
from faster_whisper.vad import SileroVADModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.config import Paths  # noqa: E402
from audio_sentinel.vad import infer_vad_probabilities, load_silero_vad  # noqa: E402


def main() -> None:
    loaded = load_silero_vad(Paths.from_root(ROOT))
    samples = np.zeros(17_000, dtype=np.float32)
    samples[4_000:12_000] = (
        0.15 * np.sin(2 * np.pi * 220 * np.arange(8_000) / 16_000)
    ).astype(np.float32)
    first = infer_vad_probabilities(loaded, samples)
    second = infer_vad_probabilities(loaded, samples.copy())
    probabilities = [frame.speech_probability for frame in first.frames]
    if first != second or first.frame_count != 34 or first.frames[-1].padding_samples != 408:
        raise RuntimeError("Silero VAD wrapper produced an unexpected or nondeterministic frame grid")
    padded = np.pad(samples, (0, first.padded_num_samples - len(samples)))
    packaged_model = Path(faster_whisper.__file__).resolve().parent / "assets" / "silero_vad_v6.onnx"
    reference = SileroVADModel(str(packaged_model))(padded)
    if not np.array_equal(np.asarray(probabilities, dtype=np.float32), reference):
        raise RuntimeError("VAD wrapper probabilities differ from the pinned reference implementation")
    print(json.dumps({
        "artifact_sha256": loaded.metadata.artifact_sha256,
        "frame_count": first.frame_count,
        "input_num_samples": first.input_num_samples,
        "max_probability": max(probabilities),
        "min_probability": min(probabilities),
        "model_id": loaded.metadata.model_id,
        "model_version": loaded.metadata.model_version,
        "runtime_version": loaded.metadata.runtime_version,
        "reference_match": True,
        "status": "passed",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
