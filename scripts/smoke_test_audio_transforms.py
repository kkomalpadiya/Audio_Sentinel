"""Generate stereo audio and exercise the loader plus A1.2 without a dataset."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import wave

import numpy as np

from audio_sentinel.audio_loader import load_audio
from audio_sentinel.audio_transforms import prepare_signal
from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.contracts import ConsentRecord
from audio_sentinel.interfaces import InputAudio


def main() -> None:
    with TemporaryDirectory(prefix="audio-sentinel-transforms-") as directory:
        settings = AudioSentinelSettings.from_project_root(Path(directory))
        settings.paths.raw_data.mkdir(parents=True)
        rate, frequency = 44_100, 500
        waveform = np.sin(2 * np.pi * frequency * np.arange(rate // 2) / rate)
        pcm = (np.column_stack((waveform * 0.5, waveform * 0.25)) * 32_767).astype("<i2")
        path = settings.paths.raw_data / "generated-stereo.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(pcm.tobytes())
        consent = ConsentRecord(
            consent_id="synthetic-transform-001", status="granted", processing_scope="acoustic_only",
            device_authorized=True, granted_at=datetime.now(UTC),
        )
        loaded = load_audio(InputAudio("generated-stereo-001", path, consent), settings)
        result = prepare_signal(loaded, settings.audio)
        assert result.samples.shape == (8_000, 1)
        assert result.duration_seconds == 0.5
        measured_rms = 20 * np.log10(np.sqrt(np.mean(result.samples.astype(np.float64) ** 2)))
        assert abs(measured_rms - (-20)) < 1e-5
        frequencies = np.fft.rfftfreq(result.num_frames, 1 / result.sample_rate_hz)
        measured_pitch = frequencies[np.argmax(abs(np.fft.rfft(result.samples[:, 0])))]
        assert measured_pitch == frequency
        assert result.source == loaded.source
        print(f"Input:  {loaded.source.num_frames} frames, {rate} Hz, 2 channels.")
        print(f"Output: {result.num_frames} frames, {result.sample_rate_hz} Hz, 1 channel, {measured_rms:.2f} dBFS RMS.")
        print(f"Verified duration: {result.duration_seconds}s; pitch: {measured_pitch:.0f} Hz.")
        print("Temporary source audio is removed on exit. Prepared samples stay in memory only.")


if __name__ == "__main__":
    main()
