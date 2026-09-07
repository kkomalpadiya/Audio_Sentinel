"""A1.5: run the public preparation service on a generated sample clip."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

# Make this checkout's service available without requiring an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audio_sentinel.config import load_settings
from audio_sentinel.contracts import ConsentRecord
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.pipeline import AudioPreparationService
from audio_sentinel.preparation import PreparedAudioManifest


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    with TemporaryDirectory(prefix="audio-sentinel-preparation-") as directory:
        root = Path(directory)
        settings = load_settings(root, audio_config_path=PROJECT_ROOT / "configs/preprocessing.example.json")
        settings.paths.raw_data.mkdir(parents=True)
        path = settings.paths.raw_data / "sample.wav"
        tone = np.sin(2 * np.pi * 440 * np.arange(76_800) / 48_000)
        sf.write(path, np.column_stack((tone * 0.2, tone * 0.1)), 48_000, subtype="PCM_16")
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        consent = ConsentRecord(
            consent_id="generated-smoke-001", status="granted", processing_scope="acoustic_only",
            device_authorized=True, granted_at=datetime.now(UTC),
        )
        service = AudioPreparationService(settings, "generated-smoke-tone")
        clip = InputAudio("sample-001", path, consent)
        saved = service.prepare(clip)
        manifest = PreparedAudioManifest.model_validate_json(saved.manifest_path.read_bytes())
        require(manifest == saved.manifest, "Saved manifest changed during readback")
        require(manifest.source.sha256 == source_hash, "Source provenance hash differs")
        full, rate = sf.read(saved.audio.audio_path, dtype="int16", always_2d=True)
        require(rate == 16_000 and full.shape == (25_600, 1), "Unexpected prepared dimensions")
        require(sf.info(saved.audio.audio_path).subtype == "PCM_16", "Unexpected output encoding")
        expected_spans = [(0, 16_000, 0), (8_000, 24_000, 0), (16_000, 25_600, 6_400),
                          (0, 25_600, 54_400), (0, 25_600, 134_400)]
        require([(w.start_sample, w.end_sample, w.padding_samples) for w in manifest.windows] == expected_spans,
                "Unexpected sample window boundaries")
        for window in manifest.windows:
            actual, window_rate = sf.read(settings.paths.interim_data / window.audio_path, dtype="int16", always_2d=True)
            real = full[window.start_sample:window.end_sample]
            require(window_rate == rate and len(actual) == len(real) + window.padding_samples, "Window dimensions differ")
            require(np.array_equal(actual[:len(real)], real) and not actual[len(real):].any(), "Window samples differ")
        rms_dbfs = float(20 * np.log10(np.sqrt(np.mean((full.astype(np.float64) / 32768) ** 2))))
        require(abs(rms_dbfs + 20) < 0.03, "Prepared loudness missed the expected target")
        repeated = service.prepare(clip)
        require(repeated.reused and repeated.directory == saved.directory, "Repeat save did not reuse output")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == source_hash, "Source was modified")
        require(not list(saved.directory.parent.glob(".pending-*")), "Staging files were left behind")
        report = {"status": "passed", "source": "generated 440 Hz stereo tone, 48 kHz",
                  "sample_rate_hz": rate, "channels": full.shape[1], "duration_seconds": 1.6,
                  "window_count": len(manifest.windows), "rms_dbfs": round(rms_dbfs, 3),
                  "reused": repeated.reused, "source_unchanged": True}
    require(not root.exists(), "Temporary sample files were not removed")
    report["temporary_files_removed"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
