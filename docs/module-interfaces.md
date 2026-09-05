# Phase 0 Module Interfaces

These interfaces are the stable handoff points for the first implementation pass. They intentionally define responsibilities without choosing model libraries, storage systems, or alert transports.

| Stage | Input | Output | Contract |
| --- | --- | --- | --- |
| Preprocessing | `InputAudio` + `AudioSettings` | `PreprocessedAudio` | `AudioPreprocessor` |
| Acoustic detection | `PreprocessedAudio` | `AcousticAnalysis` | `AcousticEventDetector` |
| Speech processing | `PreprocessedAudio` | `SpeechAnalysis` | `SpeechProcessor` |
| Language analysis | `SpeechAnalysis` | `LanguageAnalysis` | `LanguageAnalyzer` |
| Risk scoring | Acoustic, optional speech and language analyses | `RiskAssessment` | `RiskAssessor` |
| Consensus | Risk, acoustic, optional language analyses | `ConsensusDecision` | `ConsensusVerifier` |
| Alert delivery | `Alert` | None | `AlertPublisher` |

## Consent Rule

Only invoke `SpeechProcessor` and `LanguageAnalyzer` when the clip consent allows speech processing. Preprocessors and acoustic detectors must preserve the clip identifier and consent record through their outputs. Persisted manifests remain governed by the v1 data contract and must not include transcript text.

## Implementation Rule

Concrete implementations belong in later phases. They must implement the matching protocol from `audio_sentinel.interfaces` and return its declared data types. This lets a baseline acoustic model, a transcription provider, and an alert transport be replaced independently.
