from pathlib import Path

import pytest

from audio_sentinel.config import AudioSettings
from audio_sentinel.contracts import EventAnnotation, EventLabel, RiskLevel
from audio_sentinel.interfaces import (
    AcousticAnalysis,
    AcousticEventDetector,
    Alert,
    AlertPublisher,
    AudioPreprocessor,
    ConsensusDecision,
    ConsensusVerifier,
    InputAudio,
    LanguageAnalysis,
    LanguageAnalyzer,
    PreprocessedAudio,
    RiskAssessment,
    RiskAssessor,
    SpeechAnalysis,
    SpeechProcessor,
)


class StubPreprocessor:
    def preprocess(self, clip: InputAudio, settings: AudioSettings) -> PreprocessedAudio:
        return PreprocessedAudio(clip.clip_id, clip.audio_path, settings.target_sample_rate_hz, 1.0, clip.consent)


class StubDetector:
    def detect(self, audio: PreprocessedAudio) -> AcousticAnalysis:
        return AcousticAnalysis("stub-acoustic", ())


class StubSpeechProcessor:
    def analyze(self, audio: PreprocessedAudio) -> SpeechAnalysis:
        return SpeechAnalysis(False, None, None)


class StubLanguageAnalyzer:
    def analyze(self, speech: SpeechAnalysis) -> LanguageAnalysis:
        return LanguageAnalysis("stub-language", (), ())


class StubRiskAssessor:
    def assess(
        self,
        acoustic: AcousticAnalysis,
        speech: SpeechAnalysis | None,
        language: LanguageAnalysis | None,
    ) -> RiskAssessment:
        return RiskAssessment(0.0, RiskLevel.NONE, ())


class StubConsensusVerifier:
    def verify(
        self,
        assessment: RiskAssessment,
        acoustic: AcousticAnalysis,
        language: LanguageAnalysis | None,
    ) -> ConsensusDecision:
        return ConsensusDecision(True, assessment, ())


class StubAlertPublisher:
    def publish(self, alert: Alert) -> None:
        return None


def test_stubs_satisfy_all_phase_zero_interfaces() -> None:
    assert isinstance(StubPreprocessor(), AudioPreprocessor)
    assert isinstance(StubDetector(), AcousticEventDetector)
    assert isinstance(StubSpeechProcessor(), SpeechProcessor)
    assert isinstance(StubLanguageAnalyzer(), LanguageAnalyzer)
    assert isinstance(StubRiskAssessor(), RiskAssessor)
    assert isinstance(StubConsensusVerifier(), ConsensusVerifier)
    assert isinstance(StubAlertPublisher(), AlertPublisher)


def test_pipeline_data_types_preserve_the_clip_and_score_bounds(active_consent) -> None:
    clip = InputAudio("clip-001", Path("data/raw/example.wav"), active_consent)
    prepared = StubPreprocessor().preprocess(clip, AudioSettings())
    annotation = EventAnnotation(
        label=EventLabel.GUNSHOT,
        risk_level=RiskLevel.HIGH,
        start_seconds=0.0,
        end_seconds=0.2,
        confidence=0.9,
    )

    assert prepared.clip_id == clip.clip_id
    assert AcousticAnalysis("stub-acoustic", (annotation,)).events == (annotation,)

    with pytest.raises(ValueError, match="between 0 and 100"):
        RiskAssessment(101.0, RiskLevel.CRITICAL, ("out_of_range",))
