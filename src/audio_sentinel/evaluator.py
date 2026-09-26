"""A8.1 one-call offline evaluation for one authorized recorded clip."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from audio_sentinel.acoustic_aggregation import (
    AcousticAggregationResult,
    AcousticAggregationSettings,
    aggregate_acoustic_events,
)
from audio_sentinel.acoustic_evidence import (
    AcousticEvidencePolicy,
    SavedAcousticEvidence,
    save_acoustic_evidence,
)
from audio_sentinel.acoustic_inference import (
    AcousticInferenceResult,
    AcousticInferenceSettings,
    infer_prepared_audio,
)
from audio_sentinel.acoustic_loader import LoadedAcousticModel
from audio_sentinel.acoustic_model import YAMNET
from audio_sentinel.config import AudioSentinelSettings, AudioSettings
from audio_sentinel.consensus_contracts import (
    ConsensusDecisionDocument,
    ConsensusPolicy,
)
from audio_sentinel.consensus_rules import (
    ConsensusAgreementEvaluation,
    LoadedAgreementRuleSet,
    evaluate_evidence_agreement,
)
from audio_sentinel.consensus_service import FinalVerificationService
from audio_sentinel.contracts import EventAnnotation, ProcessingScope
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.language_analysis import (
    LanguageAnalysisResult,
    LanguageAnalysisSettings,
    analyze_accepted_transcripts,
)
from audio_sentinel.language_rules import LoadedLanguageRuleSet
from audio_sentinel.persistence import PersistedAudio
from audio_sentinel.pipeline import AudioPreparationService
from audio_sentinel.risk_contracts import (
    RiskAssessmentDocument,
    RiskInputSet,
    RiskSource,
)
from audio_sentinel.risk_integration import integrate_risk_inputs
from audio_sentinel.risk_scoring import LoadedRiskRuleSet, score_risk
from audio_sentinel.speech_contracts import SpeechReliabilityPolicy
from audio_sentinel.speech_segments import (
    SpeechSegmentationResult,
    SpeechSegmentationSettings,
    extract_prepared_speech_segments,
)
from audio_sentinel.speech_transcription import (
    SpeechOrchestrationSettings,
    SpeechTranscriptionResult,
    orchestrate_speech_transcription,
)
from audio_sentinel.transcription import (
    LoadedTranscriptionModel,
    TranscriptionSettings,
)
from audio_sentinel.vad import LoadedVadModel, VadInferenceSettings


class OfflineEvaluatorError(RuntimeError):
    """Stable evaluator-level failure code plus a safe explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class OfflineEvaluatorPolicies(BaseModel):
    """Explicit stage policies used by one complete offline evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    acoustic_aggregation: AcousticAggregationSettings
    acoustic_inference: AcousticInferenceSettings = Field(
        default_factory=AcousticInferenceSettings
    )
    acoustic_evidence: AcousticEvidencePolicy = Field(
        default_factory=AcousticEvidencePolicy
    )
    speech_segmentation: SpeechSegmentationSettings = Field(
        default_factory=SpeechSegmentationSettings
    )
    speech_reliability: SpeechReliabilityPolicy = Field(
        default_factory=SpeechReliabilityPolicy
    )
    vad_inference: VadInferenceSettings = Field(default_factory=VadInferenceSettings)
    speech_orchestration: SpeechOrchestrationSettings = Field(
        default_factory=SpeechOrchestrationSettings
    )
    transcription: TranscriptionSettings = Field(
        default_factory=TranscriptionSettings
    )
    language_analysis: LanguageAnalysisSettings = Field(
        default_factory=LanguageAnalysisSettings
    )
    consensus: ConsensusPolicy = Field(default_factory=ConsensusPolicy)


@dataclass(frozen=True)
class OfflineEvaluatorModels:
    """Already-loaded, locally verified models required by the evaluator."""

    acoustic: LoadedAcousticModel
    vad: LoadedVadModel | None = field(default=None, repr=False)
    transcription: LoadedTranscriptionModel | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (self.vad is None) != (self.transcription is None):
            raise ValueError(
                "vad and transcription models must either both be supplied or both be omitted"
            )

    @property
    def speech_available(self) -> bool:
        return self.vad is not None and self.transcription is not None


@dataclass(frozen=True)
class OfflineEvaluatorRuleSets:
    """Optional explicitly trusted rule artifacts; built-ins are the default."""

    language: LoadedLanguageRuleSet | None = None
    risk: LoadedRiskRuleSet | None = None
    agreement: LoadedAgreementRuleSet | None = None


@dataclass(frozen=True)
class OfflineEvaluationResult:
    """Complete in-process handoff; it is not the versioned B8.1 report."""

    prepared: PersistedAudio
    acoustic_inference: AcousticInferenceResult
    acoustic_aggregation: AcousticAggregationResult
    acoustic_evidence: SavedAcousticEvidence
    speech_segmentation: SpeechSegmentationResult | None
    speech: SpeechTranscriptionResult | None
    language: LanguageAnalysisResult | None
    risk_inputs: RiskInputSet
    risk_assessment: RiskAssessmentDocument
    agreement: ConsensusAgreementEvaluation
    decision: ConsensusDecisionDocument

    @property
    def clip_id(self) -> str:
        return self.prepared.manifest.clip.clip_id

    @property
    def reused(self) -> bool:
        return self.prepared.reused and self.acoustic_evidence.reused

    def to_summary(self) -> dict[str, object]:
        """Return a privacy-minimized diagnostic summary without transcript text."""

        return {
            "clip_id": self.clip_id,
            "processing_scope": self.prepared.manifest.clip.consent.processing_scope.value,
            "prepared_reused": self.prepared.reused,
            "acoustic_evidence_reused": self.acoustic_evidence.reused,
            "acoustic_evidence_id": self.acoustic_evidence.evidence.evidence_id,
            "acoustic_event_count": len(self.acoustic_evidence.evidence.events),
            "speech_evidence_id": (
                None if self.speech is None else self.speech.evidence.evidence_id
            ),
            "speech_segment_count": (
                None if self.speech is None else self.speech.evidence.segment_count
            ),
            "language_evidence_id": (
                None if self.language is None else self.language.evidence.evidence_id
            ),
            "language_finding_count": (
                None if self.language is None else self.language.evidence.finding_count
            ),
            "assessment_id": self.risk_assessment.assessment_id,
            "risk_score": self.risk_assessment.score,
            "risk_severity": self.risk_assessment.severity.value,
            "agreement_id": self.agreement.evaluation_id,
            "decision_id": self.decision.decision_id,
            "outcome": self.decision.outcome.value,
            "review_required": self.decision.review_required,
            "alert_candidate": self.decision.alert_candidate,
            "notification_sent": False,
        }


@dataclass(frozen=True)
class OfflineClipEvaluator:
    """Run the existing deterministic stages for one authorized local recording.

    The evaluator creates no notification and treats an ``alert`` decision only as
    a local candidate. Stage-specific exceptions propagate unchanged so callers
    retain the original stable error code and failure boundary.
    """

    settings: AudioSentinelSettings
    source_dataset: str
    models: OfflineEvaluatorModels
    policies: OfflineEvaluatorPolicies
    rule_sets: OfflineEvaluatorRuleSets = field(
        default_factory=OfflineEvaluatorRuleSets
    )

    def __post_init__(self) -> None:
        AudioPreparationService(self.settings, self.source_dataset)
        validated = OfflineEvaluatorPolicies.model_validate(
            self.policies.model_dump(mode="python")
        )
        object.__setattr__(self, "policies", validated)

    def evaluate(
        self,
        clip: InputAudio,
        *,
        annotations: Sequence[EventAnnotation] = (),
        audio_settings: AudioSettings | None = None,
        now: datetime | None = None,
    ) -> OfflineEvaluationResult:
        """Evaluate one clip from preparation through the Phase 7 decision gate."""

        if not isinstance(clip, InputAudio):
            raise OfflineEvaluatorError(
                "invalid_clip", "Evaluation requires one InputAudio record."
            )
        effective_audio = AudioSettings.model_validate(
            (audio_settings or self.settings.audio).model_dump(mode="python")
        )
        if (
            effective_audio.target_sample_rate_hz != YAMNET.sample_rate_hz
            or not effective_audio.convert_to_mono
        ):
            raise OfflineEvaluatorError(
                "incompatible_audio_settings",
                "End-to-end evaluation requires mono audio prepared at 16 kHz.",
            )
        if (
            clip.consent.processing_scope is ProcessingScope.ACOUSTIC_AND_SPEECH
            and not self.models.speech_available
        ):
            raise OfflineEvaluatorError(
                "speech_models_required",
                "The authorized processing scope requires local VAD and transcription models.",
            )

        prepared = AudioPreparationService(
            self.settings, self.source_dataset
        ).prepare(
            clip,
            annotations=annotations,
            audio_settings=effective_audio,
            now=now,
        )
        manifest_path = self._relative_manifest_path(prepared)

        acoustic_inference = infer_prepared_audio(
            self.settings.paths,
            manifest_path,
            self.models.acoustic,
            policy=self.policies.acoustic_inference,
            now=now,
        )
        acoustic_aggregation = aggregate_acoustic_events(
            acoustic_inference, self.policies.acoustic_aggregation
        )
        acoustic_evidence = save_acoustic_evidence(
            self.settings.paths,
            acoustic_inference,
            acoustic_aggregation,
            policy=self.policies.acoustic_evidence,
            now=now,
        )

        speech_segmentation: SpeechSegmentationResult | None = None
        speech: SpeechTranscriptionResult | None = None
        language: LanguageAnalysisResult | None = None
        scope = prepared.manifest.clip.consent.processing_scope
        if scope is ProcessingScope.ACOUSTIC_AND_SPEECH:
            if self.models.vad is None or self.models.transcription is None:
                raise OfflineEvaluatorError(
                    "speech_models_required",
                    "The authorized processing scope requires local VAD and transcription models.",
                )
            speech_segmentation = extract_prepared_speech_segments(
                self.settings.paths,
                manifest_path,
                self.models.vad,
                reliability_policy=self.policies.speech_reliability,
                settings=self.policies.speech_segmentation,
                inference_settings=self.policies.vad_inference,
                now=now,
            )
            speech = orchestrate_speech_transcription(
                self.settings.paths,
                speech_segmentation,
                self.models.transcription,
                settings=self.policies.speech_orchestration,
                transcription_settings=self.policies.transcription,
                now=now,
            )
            language = analyze_accepted_transcripts(
                speech,
                rule_set=self.rule_sets.language,
                settings=self.policies.language_analysis,
                now=now,
            )

        source = RiskSource(
            clip_id=prepared.manifest.clip.clip_id,
            consent_id=prepared.manifest.clip.consent.consent_id,
            processing_scope=scope,
            sample_rate_hz=prepared.manifest.clip.sample_rate_hz,
            num_samples=prepared.manifest.num_frames,
        )
        risk_inputs = integrate_risk_inputs(
            source,
            acoustic=acoustic_evidence.evidence,
            speech=None if speech is None else speech.evidence,
            language=None if language is None else language.evidence,
        )
        risk_assessment = score_risk(
            risk_inputs, rule_set=self.rule_sets.risk, now=now
        )
        agreement = evaluate_evidence_agreement(
            risk_assessment,
            rule_set=self.rule_sets.agreement,
            now=now,
        )
        decision = FinalVerificationService(
            policy=self.policies.consensus,
            trusted_rule_set=self.rule_sets.agreement,
        ).decide(risk_assessment, agreement, now=now)

        return OfflineEvaluationResult(
            prepared=prepared,
            acoustic_inference=acoustic_inference,
            acoustic_aggregation=acoustic_aggregation,
            acoustic_evidence=acoustic_evidence,
            speech_segmentation=speech_segmentation,
            speech=speech,
            language=language,
            risk_inputs=risk_inputs,
            risk_assessment=risk_assessment,
            agreement=agreement,
            decision=decision,
        )

    def _relative_manifest_path(self, prepared: PersistedAudio) -> Path:
        try:
            return prepared.manifest_path.relative_to(
                self.settings.paths.interim_data
            )
        except ValueError as error:
            raise OfflineEvaluatorError(
                "invalid_prepared_output",
                "Prepared audio manifest is outside the configured interim directory.",
            ) from error

