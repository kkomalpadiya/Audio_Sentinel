"""Install the pinned Silero VAD v6 asset from the verified faster-whisper package."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import faster_whisper  # noqa: E402

from audio_sentinel.vad import (  # noqa: E402
    INSTALL_MARKER,
    MODEL_FILENAME,
    MODEL_RELATIVE_PATH,
    SILERO_VAD,
    model_artifact_sha256,
)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    if metadata.version(SILERO_VAD.source_distribution) != SILERO_VAD.source_distribution_version:
        raise RuntimeError("Installed faster-whisper version differs from the pinned VAD source")
    source = Path(faster_whisper.__file__).resolve().parent / "assets" / MODEL_FILENAME
    if not source.is_file() or source.stat().st_size != SILERO_VAD.artifact_size_bytes:
        raise RuntimeError("Pinned faster-whisper package does not contain the expected VAD asset")
    if hashlib.sha256(source.read_bytes()).hexdigest() != SILERO_VAD.artifact_sha256:
        raise RuntimeError("Packaged VAD asset digest differs from the pinned v6 artifact")

    target = ROOT / "models" / Path(*MODEL_RELATIVE_PATH.parts)
    if target.exists():
        if model_artifact_sha256(target) != SILERO_VAD.artifact_sha256:
            raise RuntimeError("Existing VAD directory is invalid; move it aside before retrying")
        status = "reused"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(mkdtemp(prefix=".silero-vad-stage-", dir=target.parent))
        try:
            shutil.copyfile(source, stage / MODEL_FILENAME)
            (stage / INSTALL_MARKER).write_text(
                json.dumps(SILERO_VAD.marker(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _fsync_file(stage / MODEL_FILENAME)
            _fsync_file(stage / INSTALL_MARKER)
            if model_artifact_sha256(stage) != SILERO_VAD.artifact_sha256:
                raise RuntimeError("Staged VAD artifact failed verification")
            os.replace(stage, target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        status = "installed"
    print(json.dumps({
        "artifact_sha256": SILERO_VAD.artifact_sha256,
        "model_path": MODEL_RELATIVE_PATH.as_posix(),
        "model_version": SILERO_VAD.model_version,
        "source_distribution": SILERO_VAD.source_distribution,
        "source_distribution_version": SILERO_VAD.source_distribution_version,
        "status": status,
    }, indent=2))


if __name__ == "__main__":
    main()
