"""Run A4.2 end to end over verified generated prepared-audio windows."""

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

from audio_sentinel.config import AudioSentinelSettings, AudioSettings, load_settings  # noqa: E402
from audio_sentinel.contracts import ConsentRecord, ConsentStatus, ProcessingScope  # noqa: E402
from audio_sentinel.interfaces import InputAudio  # noqa: E402
from audio_sentinel.pipeline import AudioPreparationService  # noqa: E402
from audio_sentinel.speech_contracts import SpeechEvidenceDocument, SpeechReliabilityPolicy  # noqa: E402
from audio_sentinel.speech_segments import extract_prepared_speech_segments  # noqa: E402
from audio_sentinel.vad import load_silero_vad  # noqa: E402


NOW = datetime(2026, 9, 19, 12, tzinfo=UTC)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    loaded = load_silero_vad(load_settings(ROOT).paths)
    with TemporaryDirectory(prefix="audio-sentinel-speech-segments-") as temporary:
        temporary_root = Path(temporary)
        settings = AudioSentinelSettings.from_project_root(temporary_root / "project")
        settings.paths.raw_data.mkdir(parents=True)
        source = settings.paths.raw_data / "generated.wav"
        sample_count = 25_600
        timeline = np.arange(sample_count, dtype=np.float64) / 16_000
        waveform = 0.18 * np.sin(2 * np.pi * 220 * timeline)
        waveform *= 0.6 + 0.4 * np.sin(2 * np.pi * 3 * timeline)
        sf.write(source, waveform, 16_000, subtype="PCM_16")
        consent = ConsentRecord(
            consent_id="speech-segment-smoke",
            status=ConsentStatus.GRANTED,
            processing_scope=ProcessingScope.ACOUSTIC_AND_SPEECH,
            device_authorized=True,
            granted_at=NOW,
        )
        bundle = AudioPreparationService(settings, "generated-speech-shape").prepare(
            InputAudio("speech-segment-smoke", source, consent),
            audio_settings=AudioSettings(
                normalize_loudness=False,
                window_seconds=(1.0, 5.0),
                window_overlap_ratio=0.5,
            ),
            now=NOW,
        )
        manifest_path = bundle.manifest_path.relative_to(settings.paths.interim_data)
        structural_policy = SpeechReliabilityPolicy(vad_speech_threshold=0.0)
        first = extract_prepared_speech_segments(
            settings.paths,
            manifest_path,
            loaded,
            reliability_policy=structural_policy,
            now=NOW,
        )
        second = extract_prepared_speech_segments(
            settings.paths,
            manifest_path,
            loaded,
            reliability_policy=structural_policy,
            now=NOW,
        )
        require(first.evidence == SpeechEvidenceDocument.model_validate_json(
            first.evidence.model_dump_json()
        ), "Speech evidence did not round-trip through its public contract")
        require(first.evidence.evidence_id == second.evidence.evidence_id,
                "Repeat extraction changed the semantic evidence identity")
        require(first.segments == second.segments,
                "Repeat extraction changed the segment inventory")
        require(first.selected_window_seconds == 1.0,
                "Extraction did not select the shortest complete window duration")
        require(len(first.windows) == 3 and first.input_frame_count == 83,
                "Unexpected prepared-window or VAD-frame inventory")
        require(len(first.segments) == 1,
                "A zero-threshold structural check should union the complete clip")
        require((first.segments[0].start_sample, first.segments[0].end_sample) == (0, sample_count),
                "Extracted absolute sample bounds differ from the complete clip")
        require(first.segments[0].end_seconds == 1.6,
                "Extracted seconds differ from exact sample conversion")
        report = {
            "status": "passed",
            "model_id": first.evidence.vad_model.model_id,
            "model_version": first.evidence.vad_model.model_version,
            "selected_window_seconds": first.selected_window_seconds,
            "input_window_count": len(first.windows),
            "input_frame_count": first.input_frame_count,
            "segment_count": len(first.segments),
            "segment_bounds_samples": [
                first.segments[0].start_sample,
                first.segments[0].end_sample,
            ],
            "segment_bounds_seconds": [
                first.segments[0].start_seconds,
                first.segments[0].end_seconds,
            ],
            "repeat_identity_match": True,
            "structural_vad_threshold": 0.0,
        }
    require(not temporary_root.exists(), "Temporary speech-segment files were not removed")
    report["temporary_files_removed"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
