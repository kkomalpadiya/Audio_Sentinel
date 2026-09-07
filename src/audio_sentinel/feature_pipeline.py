"""A2.3 one-call preparation, feature generation, and verified storage."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
from numbers import Integral
from typing import Sequence

from audio_sentinel.audio_loader import validate_processing_consent
from audio_sentinel.config import AudioSentinelSettings, AudioSettings
from audio_sentinel.contracts import EventAnnotation
from audio_sentinel.feature_persistence import (
    FeaturePersistenceSettings, SavedLogMel, load_log_mel, save_window_log_mel,
)
from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.log_mel import estimate_log_mel_working_bytes
from audio_sentinel.persistence import PersistedAudio
from audio_sentinel.pipeline import AudioPreparationService


class FeaturePipelineError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class PreparedAudioFeatures:
    """Complete ordered inventory; carries file references, never feature arrays."""

    prepared: PersistedAudio
    features: tuple[SavedLogMel, ...]

    @property
    def reused(self) -> bool:
        return self.prepared.reused and all(feature.reused for feature in self.features)

    def to_summary(self) -> dict[str, object]:
        return {
            "clip_id": self.prepared.manifest.clip.clip_id,
            "audio_path": str(self.prepared.audio.audio_path),
            "preparation_manifest_path": str(self.prepared.manifest_path),
            "feature_count": len(self.features),
            "reused": self.reused,
            "features": [
                {"window_id": item.metadata.source.window.window_id,
                 "feature_path": str(item.feature_path), "metadata_path": str(item.metadata_path),
                 "shape": list(item.metadata.shape), "reused": item.reused}
                for item in self.features
            ],
        }


def _clock(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(UTC)


def _manifest_hash(prepared: PersistedAudio, policy: FeaturePersistenceSettings) -> str:
    if prepared.manifest_path.stat().st_size > policy.max_metadata_bytes:
        raise FeaturePipelineError("metadata_too_large", "Preparation manifest exceeds feature metadata limit.")
    with prepared.manifest_path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class AudioFeatureService:
    """Prepare one recording and save a verified feature bundle for every window.

    Errors propagate without returning a partial result. Valid completed bundles
    remain available for retry; this is not a transaction over the whole clip.
    """

    settings: AudioSentinelSettings
    source_dataset: str
    feature_policy: FeaturePersistenceSettings = field(default_factory=FeaturePersistenceSettings)
    max_total_feature_bytes: int = 1_073_741_824

    def __post_init__(self) -> None:
        # Share dataset validation with the preparation service, without doing I/O.
        AudioPreparationService(self.settings, self.source_dataset)
        if (isinstance(self.max_total_feature_bytes, bool)
                or not isinstance(self.max_total_feature_bytes, Integral)
                or self.max_total_feature_bytes <= 0):
            raise ValueError("max_total_feature_bytes must be a positive integer")

    def prepare(
        self, clip: InputAudio, *, annotations: Sequence[EventAnnotation] = (),
        audio_settings: AudioSettings | None = None, log_mel_settings: LogMelSettings | None = None,
        now: datetime | None = None,
    ) -> PreparedAudioFeatures:
        """Process one authorized clip; per-call recipes leave defaults unchanged."""
        audio = AudioSettings.model_validate((audio_settings or self.settings.audio).model_dump())
        recipe = LogMelSettings.model_validate((log_mel_settings or self.settings.log_mel).model_dump())
        policy = FeaturePersistenceSettings.model_validate(self.feature_policy.model_dump())
        if not audio.convert_to_mono:
            raise FeaturePipelineError("mono_required", "The integrated feature pipeline requires mono conversion enabled.")
        if audio.target_sample_rate_hz != recipe.sample_rate_hz:
            raise FeaturePipelineError("sample_rate_mismatch", "Preparation and Log-Mel sample rates must match.")
        consent = validate_processing_consent(clip.consent, _clock(now))
        prepared = AudioPreparationService(self.settings, self.source_dataset).prepare(
            clip, annotations=annotations, audio_settings=audio, now=now,
        )
        relative_manifest = prepared.manifest_path.relative_to(self.settings.paths.interim_data)
        digest = _manifest_hash(prepared, policy)
        # Inspect only the actual inventory: a drop-tail short clip can have none.
        estimated_total = 0
        for window in prepared.manifest.windows:
            length = window.end_sample - window.start_sample + window.padding_samples
            shape = recipe.expected_shape(length)
            if length * 4 > policy.max_decoded_bytes:
                raise FeaturePipelineError("decoded_audio_too_large", "A planned window exceeds the decoded feature-input budget.")
            if estimate_log_mel_working_bytes(length, recipe) > policy.max_working_bytes:
                raise FeaturePipelineError("working_memory_exceeded", "A planned window exceeds the generator workspace budget.")
            # Header/metadata allowance, including variable source paths and recipe.
            estimate = (shape[0] * shape[1] * 4 + 10_000
                        + len(window.model_dump_json().encode())
                        + len(recipe.model_dump_json().encode())
                        + len(relative_manifest.as_posix().encode()))
            if estimate > policy.max_output_bytes:
                raise FeaturePipelineError("output_too_large", "A planned feature bundle exceeds its output budget.")
            estimated_total += estimate
        if estimated_total > self.max_total_feature_bytes:
            raise FeaturePipelineError("total_output_too_large", "Estimated feature inventory exceeds max_total_feature_bytes.")

        completed: list[SavedLogMel] = []
        used_bytes = 0
        for window in prepared.manifest.windows:
            validate_processing_consent(consent, _clock(now))
            feature = save_window_log_mel(
                self.settings.paths, relative_manifest, window.window_id, recipe, policy=policy, now=now,
            )
            feature.metadata.source.validate_against(prepared.manifest)
            if (feature.metadata.source.window != window
                    or feature.metadata.source.preparation_manifest_sha256 != digest):
                raise FeaturePipelineError("source_changed", "Feature source differs from this preparation run.")
            used_bytes += feature.feature_path.stat().st_size + feature.metadata_path.stat().st_size
            if used_bytes > self.max_total_feature_bytes:
                raise FeaturePipelineError("total_output_too_large", "Actual feature inventory exceeds max_total_feature_bytes.")
            completed.append(feature)

        # An earlier bundle might have changed while a later window was processed.
        # Reload and release one array at a time before returning the whole inventory.
        for feature in completed:
            verified = load_log_mel(
                self.settings.paths, feature.metadata_path.relative_to(self.settings.paths.processed_data),
                policy=policy, now=now,
            )
            if verified.metadata != feature.metadata:
                raise FeaturePipelineError("output_changed", "Feature metadata changed before the pipeline completed.")
            del verified
        validate_processing_consent(consent, _clock(now))
        if _manifest_hash(prepared, policy) != digest:
            raise FeaturePipelineError("source_changed", "Preparation manifest changed before the pipeline completed.")
        return PreparedAudioFeatures(prepared, tuple(completed))
