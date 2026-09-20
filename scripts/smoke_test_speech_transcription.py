"""Run A4.3 end to end with the real pinned VAD and transcription models."""

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
from audio_sentinel.speech_transcription import orchestrate_speech_transcription  # noqa: E402
from audio_sentinel.transcription import load_transcription_model  # noqa: E402
from audio_sentinel.vad import load_silero_vad  # noqa: E402


NOW = datetime(2026, 9, 20, 12, tzinfo=UTC)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    project_paths = load_settings(ROOT).paths
    vad = load_silero_vad(project_paths)
    transcriber = load_transcription_model(project_paths)
    with TemporaryDirectory(prefix="audio-sentinel-speech-orchestration-") as temporary:
        temporary_root = Path(temporary)
        settings = AudioSentinelSettings.from_project_root(temporary_root / "project")
        settings.paths.raw_data.mkdir(parents=True)
        source = settings.paths.raw_data / "generated.wav"
        sample_count = 25_600
        timeline = np.arange(sample_count, dtype=np.float64) / 16_000
        waveform = 0.18 * np.sin(2 * np.pi * 220 * timeline)
        sf.write(source, waveform, 16_000, subtype="PCM_16")
        consent = ConsentRecord(
            consent_id="speech-orchestration-smoke",
            status=ConsentStatus.GRANTED,
            processing_scope=ProcessingScope.ACOUSTIC_AND_SPEECH,
            device_authorized=True,
            granted_at=NOW,
        )
        bundle = AudioPreparationService(settings, "generated-transcription-shape").prepare(
            InputAudio("speech-orchestration-smoke", source, consent),
            audio_settings=AudioSettings(
                normalize_loudness=False,
                window_seconds=(1.0,),
                window_overlap_ratio=0.5,
            ),
            now=NOW,
        )
        manifest_path = bundle.manifest_path.relative_to(settings.paths.interim_data)
        structural_policy = SpeechReliabilityPolicy(vad_speech_threshold=0.0)
        segmentation = extract_prepared_speech_segments(
            settings.paths,
            manifest_path,
            vad,
            reliability_policy=structural_policy,
            now=NOW,
        )
        first = orchestrate_speech_transcription(
            settings.paths, segmentation, transcriber, now=NOW
        )
        second = orchestrate_speech_transcription(
            settings.paths, segmentation, transcriber, now=NOW
        )
        require(
            first.evidence
            == SpeechEvidenceDocument.model_validate_json(first.evidence.model_dump_json()),
            "Transcribed speech evidence did not round-trip through its public contract",
        )
        require(first.evidence == second.evidence,
                "Repeat orchestration changed deterministic speech evidence")
        require(first.to_summary() == second.to_summary(),
                "Repeat orchestration changed policy-handling output")
        require(len(first.transcriptions) == len(segmentation.segments) == 1,
                "Expected one complete structural speech segment")
        require(all(
            item.assessment.downstream_text_allowed
            for item in first.transcriptions
            if item.segment_id in first.accepted_segment_ids
        ), "Downstream handoff contained text that policy did not accept")
        require(all(
            item.segment_id in first.accepted_segment_ids
            for item in first.downstream_transcripts
        ), "Accepted handoff inventory differs from assessment inventory")
        report = {
            "status": "passed",
            "vad_model_id": first.evidence.vad_model.model_id,
            "transcription_model_id": first.evidence.transcription_model.model_id,
            "segment_count": first.evidence.segment_count,
            "transcribed_segment_count": first.evidence.transcribed_segment_count,
            "accepted_segment_ids": first.accepted_segment_ids,
            "review_required_segment_ids": first.review_required_segment_ids,
            "rejected_segment_ids": first.rejected_segment_ids,
            "untranscribed_segment_ids": first.untranscribed_segment_ids,
            "repeat_identity_match": True,
            "structural_vad_threshold": 0.0,
        }
    require(not temporary_root.exists(), "Temporary orchestration files were not removed")
    report["temporary_files_removed"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
