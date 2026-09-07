"""A2.3: one-call preparation and saved features, using a generated recording."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audio_sentinel.config import load_settings
from audio_sentinel.contracts import ConsentRecord
from audio_sentinel.feature_pipeline import AudioFeatureService
from audio_sentinel.interfaces import InputAudio


def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def main() -> None:
    with TemporaryDirectory(prefix="audio-sentinel-features-") as directory:
        root = Path(directory)
        settings = load_settings(root, audio_config_path=PROJECT_ROOT / "configs/preprocessing.example.json",
                                 log_mel_config_path=PROJECT_ROOT / "configs/log-mel.example.json")
        settings.paths.raw_data.mkdir(parents=True)
        path = settings.paths.raw_data / "tone.wav"
        tone = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(76800) / 48000)
        sf.write(path, np.column_stack((tone, tone * 0.5)), 48000, subtype="PCM_16")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        consent = ConsentRecord(consent_id="feature-pipeline-smoke-001", status="granted",
                                processing_scope="acoustic_only", device_authorized=True,
                                granted_at=datetime.now(UTC))
        clip = InputAudio("tone-001", path, consent)
        service = AudioFeatureService(settings, "synthetic-tone")
        first = service.prepare(clip)
        require([f.metadata.shape for f in first.features] == [(64,97),(64,97),(64,97),(64,497),(64,997)],
                "Unexpected feature inventory")
        second = service.prepare(clip)
        require(second.reused and len(second.features) == 5, "Repeat did not reuse the complete inventory")
        require(before == hashlib.sha256(path.read_bytes()).hexdigest(), "Source changed")
        report = {"status": "passed", "feature_count": 5,
                  "shapes": [list(f.metadata.shape) for f in second.features],
                  "reused": second.reused, "source_unchanged": True}
    require(not root.exists(), "Temporary files remain")
    report["temporary_files_removed"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
