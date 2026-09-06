"""Save and reload a temporary synthetic recording and its complete window bundle."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from audio_sentinel.audio_loader import load_audio
from audio_sentinel.audio_transforms import prepare_signal
from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import ConsentRecord
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.persistence import save_prepared_audio
from audio_sentinel.preparation import PreparedAudioManifest


def main() -> None:
    with TemporaryDirectory(prefix="audio-sentinel-persistence-") as directory:
        settings = AudioSentinelSettings.from_project_root(Path(directory))
        settings.paths.raw_data.mkdir(parents=True)
        path = settings.paths.raw_data / "generated.wav"
        tone = np.sin(2 * np.pi * 440 * np.arange(76_800) / 48_000)
        sf.write(path, np.column_stack((tone * 0.2, tone * 0.1)), 48_000, subtype="PCM_16")
        original = path.read_bytes()
        consent = ConsentRecord(
            consent_id="synthetic-persistence-001", status="granted", processing_scope="acoustic_only",
            device_authorized=True, granted_at=datetime.now(UTC),
        )
        loaded = load_audio(InputAudio("generated-001", path, consent), settings)
        prepared = prepare_signal(loaded, settings.audio)
        result = save_prepared_audio(prepared, settings.paths, source_dataset="synthetic-tone",
                                     policy=settings.persistence)
        manifest = PreparedAudioManifest.model_validate_json(result.manifest_path.read_bytes())
        assert manifest == result.manifest and len(manifest.windows) == 5
        audio, rate = sf.read(result.audio.audio_path, dtype="float32", always_2d=True)
        assert rate == 16_000 and audio.shape == (25_600, 1)
        np.testing.assert_allclose(audio, prepared.samples, atol=1 / 65536)
        for window in manifest.windows:
            info = sf.info(settings.paths.interim_data / window.audio_path)
            assert info.frames == window.end_sample - window.start_sample + window.padding_samples
        repeated = save_prepared_audio(prepared, settings.paths, source_dataset="synthetic-tone")
        assert repeated.reused and repeated.directory == result.directory
        assert path.read_bytes() == original
        assert not list(result.directory.parent.glob(".pending-*"))
        print("Verified full 16 kHz mono WAV, 5 window WAVs, and validated JSON manifest.")
        print("Repeated save reused the identical bundle; original source stayed unchanged.")
        print("All synthetic files are removed on exit. No microphone or download is used.")


if __name__ == "__main__":
    main()
