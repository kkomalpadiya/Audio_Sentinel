"""B2.1: generate features from saved preparation windows of a synthetic tone."""

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
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.log_mel import generate_log_mel
from audio_sentinel.pipeline import AudioPreparationService


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    with TemporaryDirectory(prefix="audio-sentinel-log-mel-") as directory:
        root = Path(directory)
        settings = load_settings(root)
        settings.paths.raw_data.mkdir(parents=True)
        source = settings.paths.raw_data / "tone.wav"
        tone = (0.2 * np.sin(2 * np.pi * 1000 * np.arange(76800) / 48000)).astype(np.float32)
        sf.write(source, np.column_stack((tone, tone)), 48000, subtype="PCM_16")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        consent = ConsentRecord(consent_id="log-mel-smoke-001", status="granted",
                                processing_scope="acoustic_only", device_authorized=True,
                                granted_at=datetime.now(UTC))
        bundle = AudioPreparationService(settings, "synthetic-tone").prepare(InputAudio("tone-001",source,consent))
        shapes=[]
        for window in bundle.manifest.windows:
            path = settings.paths.interim_data / window.audio_path
            before = path.read_bytes()
            samples, rate = sf.read(path,dtype="float32",always_2d=True)
            result = generate_log_mel(samples,rate,settings.log_mel)
            repeated = generate_log_mel(samples,rate,settings.log_mel)
            require(np.array_equal(result.values,repeated.values),"Repeated features differ")
            require(np.isfinite(result.values).all(),"Features contain non-finite values")
            require(result.values.dtype==np.float32 and result.values.flags.c_contiguous,"Wrong array format")
            require(path.read_bytes()==before,"Prepared source changed")
            shapes.append(list(result.values.shape))
        require(shapes==[[64,97],[64,97],[64,97],[64,497],[64,997]],"Unexpected feature shapes")
        require(hashlib.sha256(source.read_bytes()).hexdigest()==digest,"Raw source changed")
        report={"status":"passed","shapes":shapes,"dtype":"float32","extractor_version":result.extractor_version,
                "source_unchanged":True,"repeatable":True}
    require(not root.exists(),"Temporary files remain")
    report["temporary_files_removed"]=True
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
