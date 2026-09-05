"""Implementation-independent contracts for the Audio Sentinel pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from audio_sentinel.config import AudioSettings
from audio_sentinel.contracts import ConsentRecord, EventAnnotation, RiskLevel


@dataclass(frozen=True)
class InputAudio:
    clip_id: str
    audio_path: Path
    consent: ConsentRecord


@dataclass(frozen=True)
class PreprocessedAudio:
    clip_id: str
    audio_path: Path
    sample_rate_hz: int
    duration_seconds: float
    consent: ConsentRecord


@dataclass(frozen=True)
class AcousticAnalysis:
    model_id: str
    events: tuple[EventAnnotation, ...]


@dataclass(frozen=True)
class SpeechAnalysis:
    speech_detected: bool
    transcript: str | None
    confidence: float | None


@dataclass(frozen=True)
class LanguageAnalysis:
    model_id: str
    indicators: tuple[str, ...]
    events: tuple[EventAnnotation, ...]


@dataclass(frozen=True)
class RiskAssessment:
    score: float
    risk_level: RiskLevel
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 100:
            raise ValueError("score must be between 0 and 100")


@dataclass(frozen=True)
class ConsensusDecision:
    approved: bool
    assessment: RiskAssessment
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class Alert:
    alert_id: str
    clip_id: str
    assessment: RiskAssessment
    reason_codes: tuple[str, ...]


@runtime_checkable
class AudioPreprocessor(Protocol):
    def preprocess(self, clip: InputAudio, settings: AudioSettings) -> PreprocessedAudio: ...


@runtime_checkable
class AcousticEventDetector(Protocol):
    def detect(self, audio: PreprocessedAudio) -> AcousticAnalysis: ...


@runtime_checkable
class SpeechProcessor(Protocol):
    def analyze(self, audio: PreprocessedAudio) -> SpeechAnalysis: ...


@runtime_checkable
class LanguageAnalyzer(Protocol):
    def analyze(self, speech: SpeechAnalysis) -> LanguageAnalysis: ...


@runtime_checkable
class RiskAssessor(Protocol):
    def assess(
        self,
        acoustic: AcousticAnalysis,
        speech: SpeechAnalysis | None,
        language: LanguageAnalysis | None,
    ) -> RiskAssessment: ...


@runtime_checkable
class ConsensusVerifier(Protocol):
    def verify(
        self,
        assessment: RiskAssessment,
        acoustic: AcousticAnalysis,
        language: LanguageAnalysis | None,
    ) -> ConsensusDecision: ...


@runtime_checkable
class AlertPublisher(Protocol):
    def publish(self, alert: Alert) -> None: ...
