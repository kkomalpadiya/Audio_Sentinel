"""Exercise B1.1 with a generated tone; no recording or dataset permission needed."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import wave

import numpy as np

from audio_sentinel.audio_loader import load_audio
from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import ConsentRecord
from audio_sentinel.interfaces import InputAudio


def main() -> None:
    with TemporaryDirectory(prefix="audio-sentinel-loader-") as directory:
        settings = AudioSentinelSettings.from_project_root(Path(directory))
        settings.paths.raw_data.mkdir(parents=True)
        rate = 16_000
        samples = (np.sin(2 * np.pi * 440 * np.arange(1_600) / rate) * 8_000).astype("<i2")
        path = settings.paths.raw_data / "generated-tone.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(samples.tobytes())
        consent = ConsentRecord(
            consent_id="synthetic-test-001", status="granted", processing_scope="acoustic_only",
            device_authorized=True, granted_at=datetime.now(UTC),
        )
        loaded = load_audio(InputAudio("generated-tone-001", path, consent), settings)
        np.testing.assert_array_equal(loaded.samples[:, 0], samples.astype(np.float32) / 32_768)
        print(f"Loaded generated tone: {loaded.source.num_frames} frames, "
              f"{loaded.source.sample_rate_hz} Hz, {loaded.source.channels} channel, {loaded.samples.dtype}.")
        print("PCM scaling and source metadata verified. Temporary audio is removed on exit.")


if __name__ == "__main__":
    main()
