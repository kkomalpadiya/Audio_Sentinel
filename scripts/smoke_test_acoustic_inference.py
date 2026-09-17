"""Run the pinned local YAMNet over generated, prepared waveform windows."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_sentinel.acoustic_inference import infer_prepared_audio  # noqa: E402
from audio_sentinel.acoustic_loader import load_yamnet  # noqa: E402
from audio_sentinel.config import AudioSentinelSettings, AudioSettings, load_settings  # noqa: E402
from audio_sentinel.contracts import ConsentRecord, ConsentStatus, ProcessingScope  # noqa: E402
from audio_sentinel.interfaces import InputAudio  # noqa: E402
from audio_sentinel.pipeline import AudioPreparationService  # noqa: E402


def main() -> None:
    loaded = load_yamnet(load_settings(ROOT).paths)
    with TemporaryDirectory(prefix="audio-sentinel-yamnet-") as temporary:
        settings = AudioSentinelSettings.from_project_root(Path(temporary) / "project")
        settings.paths.raw_data.mkdir(parents=True)
        source = settings.paths.raw_data / "tone.wav"
        samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(25_600) / 16_000)
        sf.write(source, samples, 16_000, subtype="PCM_16")
        consent = ConsentRecord(
            consent_id="acoustic-inference-smoke",
            status=ConsentStatus.GRANTED,
            processing_scope=ProcessingScope.ACOUSTIC_ONLY,
            device_authorized=True,
            granted_at=datetime(2026, 9, 1, tzinfo=UTC),
        )
        bundle = AudioPreparationService(settings, "generated-tone").prepare(
            InputAudio("generated-tone", source, consent),
            audio_settings=AudioSettings(
                normalize_loudness=False,
                window_seconds=(1.0,),
                window_overlap_ratio=0.5,
            ),
            now=datetime(2026, 9, 7, tzinfo=UTC),
        )
        result = infer_prepared_audio(
            settings.paths,
            bundle.manifest_path.relative_to(settings.paths.interim_data),
            loaded,
            now=datetime(2026, 9, 7, tzinfo=UTC),
        )
        assert [item.scores.shape for item in result.windows] == [(2, 521), (2, 521), (1, 521)]
        assert all(np.isfinite(item.scores).all() for item in result.windows)
        assert all(np.all((0 <= item.scores) & (item.scores <= 1)) for item in result.windows)
        print(json.dumps(result.to_summary(), indent=2))


if __name__ == "__main__":
    main()
