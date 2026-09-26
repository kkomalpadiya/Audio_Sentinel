"""A8.1 tests for one-call offline recorded-clip evaluation."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio_sentinel.acoustic_aggregation import AcousticAggregationSettings
from audio_sentinel.acoustic_inference import AcousticInferenceError, expected_yamnet_patches
from audio_sentinel.acoustic_loader import (
    AcousticModelMetadata,
    LoadedAcousticModel,
    TensorContract,
)
from audio_sentinel.acoustic_model import LABEL_MAPPING, YAMNET, load_class_map
from audio_sentinel.config import AudioSettings
from audio_sentinel.consensus_contracts import ConsensusOutcome
from audio_sentinel.contracts import (
    ConsentRecord,
    ConsentStatus,
    EventLabel,
    ProcessingScope,
)
from audio_sentinel.evaluator import (
    OfflineClipEvaluator,
    OfflineEvaluationResult,
    OfflineEvaluatorError,
    OfflineEvaluatorModels,
    OfflineEvaluatorPolicies,
)
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.risk_contracts import RiskInputStatus
from audio_sentinel.transcription import (
    WHISPER_TINY_EN,
    LoadedTranscriptionModel,
    TranscriptionArtifactMetadata,
    TranscriptionModelMetadata,
)
from audio_sentinel.vad import (
    SILERO_VAD,
    LoadedVadModel,
    VadModelMetadata,
    VadTensorContract,
)


NOW = datetime(2026, 9, 27, 12, tzinfo=UTC)


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def numpy(self) -> np.ndarray:
        return self.values


class FakeAcousticModel:
    def __init__(self, *, label: EventLabel | None, score: float = 0.9) -> None:
        self.label = label
        self.score = score
        self.calls = 0

    def __call__(self, waveform: np.ndarray):
        self.calls += 1
        patches = expected_yamnet_patches(len(waveform))
        frames = YAMNET.patch_frames + (patches - 1) * (YAMNET.patch_frames // 2)
        scores = np.zeros((patches, YAMNET.num_classes), dtype=np.float32)
        if self.label is not None:
            scores[:, LABEL_MAPPING[self.label][0].index] = np.float32(self.score)
        return tuple(
            FakeTensor(item)
            for item in (
                scores,
                np.zeros((patches, 1024), dtype=np.float32),
                np.zeros((frames, YAMNET.mel_bands), dtype=np.float32),
            )
        )


class InvalidAcousticModel(FakeAcousticModel):
    def __call__(self, waveform: np.ndarray):
        outputs = list(super().__call__(waveform))
        outputs[0] = FakeTensor(np.zeros((1, 1), dtype=np.float32))
        return tuple(outputs)


def loaded_acoustic(model: object) -> LoadedAcousticModel:
    metadata = AcousticModelMetadata(
        model_id=YAMNET.model_id,
        model_version="1",
        model_handle=YAMNET.model_handle,
        model_path="yamnet/1",
        artifact_sha256=YAMNET.artifact_sha256,
        class_map_sha256=YAMNET.class_map_sha256,
        label_mapping_version="1.0",
        runtime_distribution=YAMNET.runtime_distribution,
        runtime_version=YAMNET.runtime_version,
        exported_with_tensorflow="2.3.0",
        exported_with_tensorflow_git="test",
        signature_name="serving_default",
        input=TensorContract("waveform", (None,), "float32"),
        outputs=(
            TensorContract("output_0", (None, 521), "float32"),
            TensorContract("output_1", (None, 1024), "float32"),
            TensorContract("output_2", (None, 64), "float32"),
        ),
        num_classes=521,
    )
    return LoadedAcousticModel(metadata, load_class_map(), model)


class FakeVadSession:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    def run(self, _outputs, feeds):
        self.calls += 1
        scores = np.full(len(feeds["input"]), self.score, dtype=np.float32)
        state = np.zeros((1, 1, SILERO_VAD.state_size), dtype=np.float32)
        return scores, state, state.copy()


def loaded_vad(session: object) -> LoadedVadModel:
    metadata = VadModelMetadata(
        model_id=SILERO_VAD.model_id,
        model_version=SILERO_VAD.model_version,
        model_source=SILERO_VAD.model_source,
        model_path="silero-vad/6",
        artifact_sha256=SILERO_VAD.artifact_sha256,
        source_distribution=SILERO_VAD.source_distribution,
        source_distribution_version=SILERO_VAD.source_distribution_version,
        runtime_distribution=SILERO_VAD.runtime_distribution,
        runtime_version=SILERO_VAD.runtime_version,
        providers=("CPUExecutionProvider",),
        inputs=(
            VadTensorContract("input", (None, 576), "tensor(float)"),
            VadTensorContract("h", (1, 1, 128), "tensor(float)"),
            VadTensorContract("c", (1, 1, 128), "tensor(float)"),
        ),
        outputs=(
            VadTensorContract("speech_probs", (None,), "tensor(float)"),
            VadTensorContract("hn", (1, 1, 128), "tensor(float)"),
            VadTensorContract("cn", (1, 1, 128), "tensor(float)"),
        ),
        sample_rate_hz=16_000,
        frame_samples=512,
        context_samples=64,
    )
    return LoadedVadModel(metadata, session)


class FakeTranscriber:
    def __init__(self, text: str, confidence: float = 0.9) -> None:
        self.text = text
        self.confidence = confidence
        self.calls = 0

    def transcribe(self, samples: np.ndarray, **_kwargs):
        self.calls += 1
        duration = len(samples) / 16_000
        chunks = (
            SimpleNamespace(
                id=0,
                start=0.0,
                end=duration,
                text=self.text,
                tokens=(1, 2, 3),
                avg_logprob=math.log(self.confidence),
                no_speech_prob=0.05,
            ),
        )
        info = SimpleNamespace(
            language="en", language_probability=1.0, duration=duration
        )
        return iter(chunks), info


def loaded_transcriber(model: object) -> LoadedTranscriptionModel:
    spec = WHISPER_TINY_EN
    metadata = TranscriptionModelMetadata(
        model_id=spec.model_id,
        model_version=spec.model_version,
        model_source=spec.model_source,
        model_path="faster-whisper-tiny.en/1",
        source_repository=spec.source_repository,
        source_revision=spec.source_revision,
        artifact_sha256=spec.artifact_sha256,
        artifacts=tuple(
            TranscriptionArtifactMetadata(item.filename, item.size_bytes, item.sha256)
            for item in spec.artifacts
        ),
        runtime_distribution="faster-whisper",
        runtime_version=spec.faster_whisper_version,
        ctranslate2_version=spec.ctranslate2_version,
        tokenizers_version=spec.tokenizers_version,
        device="cpu",
        compute_type="int8_float32",
        language="en",
        sample_rate_hz=16_000,
    )
    return LoadedTranscriptionModel(metadata, model)


def consent(scope: ProcessingScope) -> ConsentRecord:
    return ConsentRecord(
        consent_id=f"consent-{scope.value}",
        status=ConsentStatus.GRANTED,
        processing_scope=scope,
        device_authorized=True,
        granted_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


def input_clip(settings, scope: ProcessingScope, *, clip_id: str = "evaluation-clip") -> InputAudio:
    settings.paths.raw_data.mkdir(parents=True, exist_ok=True)
    path = settings.paths.raw_data / f"{clip_id}.wav"
    timeline = np.arange(12_000, dtype=np.float64) / 16_000
    sf.write(path, 0.1 * np.sin(2 * np.pi * 220 * timeline), 16_000, subtype="PCM_16")
    return InputAudio(clip_id, path, consent(scope))


def policies() -> OfflineEvaluatorPolicies:
    return OfflineEvaluatorPolicies(
        acoustic_aggregation=AcousticAggregationSettings.uniform(0.5)
    )


def evaluator(
    settings,
    *,
    acoustic_label: EventLabel | None = EventLabel.EXPLOSION,
    acoustic_score: float = 0.9,
    speech: bool = True,
    transcript: str = "I will kill you",
    acoustic_model: object | None = None,
) -> tuple[OfflineClipEvaluator, object, object, object]:
    acoustic = acoustic_model or FakeAcousticModel(
        label=acoustic_label, score=acoustic_score
    )
    vad = FakeVadSession(0.9)
    transcriber = FakeTranscriber(transcript)
    models = OfflineEvaluatorModels(
        acoustic=loaded_acoustic(acoustic),
        vad=loaded_vad(vad) if speech else None,
        transcription=loaded_transcriber(transcriber) if speech else None,
    )
    service = OfflineClipEvaluator(
        settings=settings,
        source_dataset="synthetic-evaluator-tests",
        models=models,
        policies=policies(),
    )
    return service, acoustic, vad, transcriber


def test_runs_complete_authorized_clip_to_local_alert_candidate(temporary_settings):
    service, acoustic, vad, transcriber = evaluator(temporary_settings)
    result = service.evaluate(
        input_clip(temporary_settings, ProcessingScope.ACOUSTIC_AND_SPEECH),
        audio_settings=AudioSettings(
            normalize_loudness=False,
            window_seconds=(1.0,),
            window_overlap_ratio=0.5,
        ),
        now=NOW,
    )

    assert isinstance(result, OfflineEvaluationResult)
    assert acoustic.calls == vad.calls == transcriber.calls == 1
    assert result.prepared.manifest.clip.clip_id == "evaluation-clip"
    assert result.acoustic_evidence.evidence_path.is_file()
    assert result.speech_segmentation is not None
    assert result.speech is not None and result.speech.evidence.segment_count == 1
    assert result.language is not None and result.language.evidence.finding_count >= 1
    assert result.risk_assessment.score == 92
    assert result.decision.outcome is ConsensusOutcome.ALERT
    assert result.decision.alert_candidate is True
    assert result.decision.review_required is True
    assert result.to_summary()["notification_sent"] is False


def test_acoustic_only_scope_skips_speech_and_cannot_alert(temporary_settings):
    service, _acoustic, vad, transcriber = evaluator(
        temporary_settings, speech=False
    )
    result = service.evaluate(
        input_clip(temporary_settings, ProcessingScope.ACOUSTIC_ONLY),
        audio_settings=AudioSettings(
            normalize_loudness=False, window_seconds=(1.0,)
        ),
        now=NOW,
    )

    assert vad.calls == transcriber.calls == 0
    assert result.speech_segmentation is result.speech is result.language is None
    assert result.risk_inputs.speech.status is RiskInputStatus.NOT_PERMITTED
    assert result.risk_inputs.language.status is RiskInputStatus.NOT_PERMITTED
    assert result.decision.outcome is ConsensusOutcome.REVIEW
    assert result.decision.alert_candidate is False
    assert result.to_summary()["speech_evidence_id"] is None


def test_no_vad_speech_still_completes_with_no_accepted_text(temporary_settings):
    service, acoustic, _vad, transcriber = evaluator(
        temporary_settings, acoustic_label=None, transcript="harmless words"
    )
    quiet_vad = FakeVadSession(0.1)
    service = replace(
        service,
        models=OfflineEvaluatorModels(
            acoustic=loaded_acoustic(acoustic),
            vad=loaded_vad(quiet_vad),
            transcription=loaded_transcriber(transcriber),
        ),
    )
    result = service.evaluate(
        input_clip(temporary_settings, ProcessingScope.ACOUSTIC_AND_SPEECH),
        audio_settings=AudioSettings(
            normalize_loudness=False, window_seconds=(1.0,)
        ),
        now=NOW,
    )

    assert transcriber.calls == 0
    assert result.speech is not None and result.speech.evidence.segment_count == 0
    assert result.language is not None and result.language.evidence.finding_count == 0
    assert result.risk_inputs.language.status is RiskInputStatus.NO_ACCEPTED_TEXT
    assert result.risk_assessment.score == 0
    assert result.decision.outcome is ConsensusOutcome.NO_ACTION


def test_speech_scope_requires_both_models_before_writing_outputs(temporary_settings):
    service, *_ = evaluator(temporary_settings, speech=False)
    clip = input_clip(temporary_settings, ProcessingScope.ACOUSTIC_AND_SPEECH)

    with pytest.raises(OfflineEvaluatorError) as captured:
        service.evaluate(clip, now=NOW)

    assert captured.value.code == "speech_models_required"
    assert not temporary_settings.paths.interim_data.exists()


def test_model_bundle_rejects_incomplete_speech_pair(temporary_settings):
    with pytest.raises(ValueError, match="both be supplied"):
        OfflineEvaluatorModels(
            acoustic=loaded_acoustic(FakeAcousticModel(label=None)),
            vad=loaded_vad(FakeVadSession(0.1)),
        )


@pytest.mark.parametrize(
    "audio_settings",
    (
        AudioSettings(target_sample_rate_hz=8_000),
        AudioSettings(convert_to_mono=False),
    ),
)
def test_incompatible_audio_recipe_fails_before_persistence(
    temporary_settings, audio_settings
):
    service, *_ = evaluator(temporary_settings)
    clip = input_clip(temporary_settings, ProcessingScope.ACOUSTIC_AND_SPEECH)

    with pytest.raises(OfflineEvaluatorError) as captured:
        service.evaluate(clip, audio_settings=audio_settings, now=NOW)

    assert captured.value.code == "incompatible_audio_settings"
    assert not temporary_settings.paths.interim_data.exists()


def test_stage_specific_error_propagates_with_original_code(temporary_settings):
    invalid = InvalidAcousticModel(label=EventLabel.EXPLOSION)
    service, *_ = evaluator(temporary_settings, acoustic_model=invalid)

    with pytest.raises(AcousticInferenceError) as captured:
        service.evaluate(
            input_clip(temporary_settings, ProcessingScope.ACOUSTIC_AND_SPEECH),
            audio_settings=AudioSettings(
                normalize_loudness=False, window_seconds=(1.0,)
            ),
            now=NOW,
        )

    assert captured.value.code == "invalid_output"


def test_fixed_time_repeat_is_semantically_deterministic_and_reuses_outputs(
    temporary_settings,
):
    service, *_ = evaluator(temporary_settings)
    clip = input_clip(temporary_settings, ProcessingScope.ACOUSTIC_AND_SPEECH)
    recipe = AudioSettings(normalize_loudness=False, window_seconds=(1.0,))

    first = service.evaluate(clip, audio_settings=recipe, now=NOW)
    second = service.evaluate(clip, audio_settings=recipe, now=NOW)

    assert second.reused is True
    assert first.risk_assessment == second.risk_assessment
    assert first.agreement == second.agreement
    assert first.decision == second.decision
    assert first.to_summary() == second.to_summary() | {
        "prepared_reused": False,
        "acoustic_evidence_reused": False,
    }


def test_summary_is_json_ready_and_excludes_transcript_text(temporary_settings):
    phrase = "I will kill you"
    service, *_ = evaluator(temporary_settings, transcript=phrase)
    result = service.evaluate(
        input_clip(temporary_settings, ProcessingScope.ACOUSTIC_AND_SPEECH),
        audio_settings=AudioSettings(
            normalize_loudness=False, window_seconds=(1.0,)
        ),
        now=NOW,
    )

    encoded = json.dumps(result.to_summary(), sort_keys=True)
    assert phrase not in encoded
    assert str(result.prepared.manifest_path) not in encoded
    assert result.decision.decision_id in encoded


def test_result_and_policy_objects_are_immutable(temporary_settings):
    service, *_ = evaluator(temporary_settings, speech=False)
    result = service.evaluate(
        input_clip(temporary_settings, ProcessingScope.ACOUSTIC_ONLY),
        audio_settings=AudioSettings(
            normalize_loudness=False, window_seconds=(1.0,)
        ),
        now=NOW,
    )

    with pytest.raises(FrozenInstanceError):
        result.speech = None
    with pytest.raises(Exception):
        service.policies.consensus = None


def test_invalid_clip_type_has_stable_safe_failure(temporary_settings):
    service, *_ = evaluator(temporary_settings, speech=False)

    with pytest.raises(OfflineEvaluatorError) as captured:
        service.evaluate(object())  # type: ignore[arg-type]

    assert captured.value.code == "invalid_clip"
    assert str(captured.value) == "Evaluation requires one InputAudio record."

