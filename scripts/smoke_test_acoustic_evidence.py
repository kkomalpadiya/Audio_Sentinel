"""Run the real local YAMNet path through persisted A3.3 evidence JSON."""

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

from audio_sentinel.acoustic_aggregation import (  # noqa: E402
    AcousticAggregationSettings,
    aggregate_acoustic_events,
)
from audio_sentinel.acoustic_evidence import (  # noqa: E402
    load_acoustic_evidence,
    save_acoustic_evidence,
)
from audio_sentinel.acoustic_inference import infer_prepared_audio  # noqa: E402
from audio_sentinel.acoustic_loader import load_yamnet  # noqa: E402
from audio_sentinel.acoustic_model import LABEL_MAPPING  # noqa: E402
from audio_sentinel.config import AudioSentinelSettings, AudioSettings, load_settings  # noqa: E402
from audio_sentinel.contracts import ConsentRecord, ConsentStatus, ProcessingScope  # noqa: E402
from audio_sentinel.interfaces import InputAudio  # noqa: E402
from audio_sentinel.pipeline import AudioPreparationService  # noqa: E402


def main() -> None:
    loaded = load_yamnet(load_settings(ROOT).paths)
    now = datetime(2026, 9, 18, tzinfo=UTC)
    with TemporaryDirectory(prefix="audio-sentinel-evidence-") as temporary:
        settings = AudioSentinelSettings.from_project_root(Path(temporary) / "project")
        settings.paths.raw_data.mkdir(parents=True)
        source = settings.paths.raw_data / "tone.wav"
        samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(25_600) / 16_000)
        sf.write(source, samples, 16_000, subtype="PCM_16")
        consent = ConsentRecord(
            consent_id="acoustic-evidence-smoke",
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
            now=now,
        )
        raw = infer_prepared_audio(
            settings.paths,
            bundle.manifest_path.relative_to(settings.paths.interim_data),
            loaded,
            now=now,
        )
        # Zero is intentionally a structural-test threshold, not a recommended
        # detector threshold. It makes every mapped patch exercise persistence.
        aggregated = aggregate_acoustic_events(
            raw, AcousticAggregationSettings.uniform(0.0)
        )
        saved = save_acoustic_evidence(settings.paths, raw, aggregated, now=now)
        relative = saved.evidence_path.relative_to(settings.paths.processed_data)
        reloaded = load_acoustic_evidence(settings.paths, relative, now=now)
        assert reloaded == saved.evidence
        assert len(reloaded.events) == len(LABEL_MAPPING)
        assert all((item.start_sample, item.end_sample) == (0, 25_600) for item in reloaded.events)
        print(json.dumps({
            "status": "passed",
            "threshold_purpose": "structural persistence test only",
            "schema_version": reloaded.schema_version,
            "document_type": reloaded.document_type,
            "evidence_id": reloaded.evidence_id,
            "input_windows": reloaded.input_window_count,
            "input_patches": reloaded.input_patch_count,
            "mapped_events": len(reloaded.events),
            "source_verified_on_reload": True,
            "risk_decision_emitted": False,
        }, indent=2))


if __name__ == "__main__":
    main()
