"""Run loading, preparation, and segmentation on a temporary synthetic recording."""

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
from audio_sentinel.segmentation import iter_windows


def main() -> None:
    with TemporaryDirectory(prefix="audio-sentinel-windows-") as directory:
        settings = AudioSentinelSettings.from_project_root(Path(directory))
        settings.paths.raw_data.mkdir(parents=True)
        path = settings.paths.raw_data / "generated.wav"
        waveform = np.sin(2 * np.pi * 440 * np.arange(76_800) / 48_000)
        pcm = (np.column_stack((waveform * 0.2, waveform * 0.1)) * 32_767).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(pcm.tobytes())
        consent = ConsentRecord(
            consent_id="synthetic-windows-001", status="granted", processing_scope="acoustic_only",
            device_authorized=True, granted_at=datetime.now(UTC),
        )
        loaded = load_audio(InputAudio("generated-001", path, consent), settings)
        prepared = prepare_signal(loaded, settings.audio)
        records = []
        for window in iter_windows(prepared):
            record = window.record
            records.append(record)
            assert window.samples.shape == (round(record.window_seconds * 16_000), 1)
            print(f"{record.window_seconds:g}s window: frames [{record.start_sample}, {record.end_sample}), "
                  f"{record.padding_samples} padding frames")
        assert [(r.start_sample, r.end_sample, r.padding_samples) for r in records[:3]] == [
            (0, 16_000, 0), (8_000, 24_000, 0), (16_000, 25_600, 6_400),
        ]
        assert len(records) == 5
        assert prepared.duration_seconds == 1.6
        print("Verified 5 windows across 1s, 5s, and 10s contexts. No prepared files were written.")
        print("Temporary source is removed on exit; no microphone or dataset download is used.")


if __name__ == "__main__":
    main()
