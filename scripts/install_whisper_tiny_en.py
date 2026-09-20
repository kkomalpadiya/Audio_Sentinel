"""Install the pinned Faster-Whisper tiny.en model from Hugging Face."""

from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import hf_hub_download  # noqa: E402

from audio_sentinel.transcription import (  # noqa: E402
    INSTALL_MARKER,
    MODEL_RELATIVE_PATH,
    WHISPER_TINY_EN,
    model_artifact_sha256,
)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    versions = {
        "faster-whisper": WHISPER_TINY_EN.faster_whisper_version,
        "ctranslate2": WHISPER_TINY_EN.ctranslate2_version,
        "tokenizers": WHISPER_TINY_EN.tokenizers_version,
        "huggingface-hub": WHISPER_TINY_EN.huggingface_hub_version,
    }
    for distribution, expected in versions.items():
        if metadata.version(distribution) != expected:
            raise RuntimeError(f"Installed {distribution} version differs from the pinned runtime")

    target = ROOT / "models" / Path(*MODEL_RELATIVE_PATH.parts)
    if target.exists():
        if model_artifact_sha256(target) != WHISPER_TINY_EN.artifact_sha256:
            raise RuntimeError("Existing transcription directory is invalid; move it aside before retrying")
        status = "reused"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(mkdtemp(prefix=".whisper-stage-", dir=target.parent))
        try:
            for artifact in WHISPER_TINY_EN.artifacts:
                cached = hf_hub_download(
                    repo_id=WHISPER_TINY_EN.source_repository,
                    filename=artifact.filename,
                    revision=WHISPER_TINY_EN.source_revision,
                )
                shutil.copyfile(cached, stage / artifact.filename)
            (stage / INSTALL_MARKER).write_text(
                json.dumps(WHISPER_TINY_EN.marker(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for artifact in WHISPER_TINY_EN.artifacts:
                _fsync_file(stage / artifact.filename)
            _fsync_file(stage / INSTALL_MARKER)
            if model_artifact_sha256(stage) != WHISPER_TINY_EN.artifact_sha256:
                raise RuntimeError("Staged transcription model failed verification")
            os.replace(stage, target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        status = "installed"
    print(json.dumps({
        "artifact_sha256": WHISPER_TINY_EN.artifact_sha256,
        "model_path": MODEL_RELATIVE_PATH.as_posix(),
        "model_version": WHISPER_TINY_EN.model_version,
        "source_repository": WHISPER_TINY_EN.source_repository,
        "source_revision": WHISPER_TINY_EN.source_revision,
        "status": status,
    }, indent=2))


if __name__ == "__main__":
    main()
