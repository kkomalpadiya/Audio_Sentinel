"""One-call offline audio preparation, with the shared preprocessor interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from audio_sentinel.audio_loader import load_audio
from audio_sentinel.audio_transforms import prepare_signal
from audio_sentinel.config import AudioSentinelSettings, AudioSettings
from audio_sentinel.contracts import EventAnnotation
from audio_sentinel.interfaces import InputAudio, PreprocessedAudio
from audio_sentinel.persistence import PersistedAudio, save_prepared_audio


@dataclass(frozen=True)
class WindowPlan:
    short_seconds: float = 1.0
    medium_seconds: float = 5.0
    long_seconds: float = 10.0


DEFAULT_WINDOW_PLAN = WindowPlan()


@dataclass(frozen=True)
class AudioPreparationService:
    """Prepare clips from one documented dataset using fixed project settings.

    Construction performs no I/O. Each call owns its intermediate arrays and
    returns only file-backed results; no last-result state is kept on the service.
    Stage exceptions, including their stable error codes, propagate unchanged.
    """

    settings: AudioSentinelSettings
    source_dataset: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_dataset, str) or not self.source_dataset.strip() or len(self.source_dataset) > 128:
            raise ValueError("source_dataset must be a nonblank name of at most 128 characters")

    def prepare(
        self, clip: InputAudio, *, annotations: Sequence[EventAnnotation] = (),
        audio_settings: AudioSettings | None = None, now: datetime | None = None,
    ) -> PersistedAudio:
        """Load, transform, window, and save one authorized local recording.

        Relative input paths are relative to settings.paths.raw_data. An audio
        override applies to every stage of this call, leaving the service defaults
        intact. Omit now in production so each stage checks the live clock.
        """
        effective = self.settings if audio_settings is None else self.settings.model_copy(update={"audio": audio_settings})
        loaded = load_audio(clip, effective, now=now)
        prepared = prepare_signal(loaded, effective.audio, now=now)
        del loaded  # The source array is no longer needed while saving windows.
        return save_prepared_audio(
            prepared, effective.paths, source_dataset=self.source_dataset,
            annotations=annotations, policy=effective.persistence, now=now,
        )

    def preprocess(self, clip: InputAudio, settings: AudioSettings) -> PreprocessedAudio:
        """Implement AudioPreprocessor using the caller's audio recipe."""
        return self.prepare(clip, audio_settings=settings).audio
