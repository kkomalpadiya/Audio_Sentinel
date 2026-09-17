"""Download the pinned public YAMNet v1 SavedModel into the ignored models directory."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import kagglehub  # noqa: E402

from audio_sentinel.acoustic_loader import MODEL_RELATIVE_PATH, model_artifact_sha256  # noqa: E402
from audio_sentinel.acoustic_model import YAMNET  # noqa: E402


def main() -> None:
    target = ROOT / "models" / Path(*MODEL_RELATIVE_PATH.parts)
    target.mkdir(parents=True, exist_ok=True)
    downloaded = Path(kagglehub.model_download(YAMNET.download_handle, output_dir=str(target)))
    if downloaded.resolve() != target.resolve():
        raise RuntimeError("Kaggle downloaded YAMNet to an unexpected directory")
    digest = model_artifact_sha256(target)
    if digest != YAMNET.artifact_sha256:
        raise RuntimeError("Downloaded YAMNet digest differs from the pinned artifact")
    print(json.dumps({
        "model_handle": YAMNET.model_handle,
        "model_path": MODEL_RELATIVE_PATH.as_posix(),
        "artifact_sha256": digest,
        "status": "verified",
    }, indent=2))


if __name__ == "__main__":
    main()
