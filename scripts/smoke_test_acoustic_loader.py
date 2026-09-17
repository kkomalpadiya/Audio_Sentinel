"""Load and inspect the pinned local YAMNet model without running inference."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.acoustic_loader import load_yamnet  # noqa: E402
from audio_sentinel.config import load_settings  # noqa: E402


def main() -> None:
    loaded = load_yamnet(load_settings(ROOT).paths)
    metadata = loaded.metadata.as_dict()
    assert metadata["num_classes"] == 521
    assert metadata["runtime_version"] == "2.21.0"
    assert metadata["input"] == {"name": "waveform", "shape": (None,), "dtype": "float32"}
    assert [item["shape"] for item in metadata["outputs"]] == [
        (None, 521), (None, 1024), (None, 64)
    ]
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
