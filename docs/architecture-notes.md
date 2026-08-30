# Architecture Notes

## Confirmed agent plan

### Prebuilt agents

1. Speech Recognition Agent
   - Use a pretrained ASR system such as Faster-Whisper.
2. Acoustic Event Detection Agent
   - Use a pretrained acoustic model such as YAMNet or PANNs.
3. Language Understanding Agent
   - Use a pretrained text model or rules-plus-model pipeline to analyze intent, keywords, negation, and harmless context.

### Custom components

1. Risk Assessment Agent
   - Our own scoring formula or trainable model that converts the agent outputs into a calibrated risk score.
2. Verification / Consensus Agent
   - Cross-checks the outputs of the other agents and applies consensus rules before alerting.

## Practical build order

1. Offline audio ingestion.
2. Preprocessing and windowing.
3. Acoustic baseline with a pretrained model.
4. Speech gate and transcription.
5. Language analysis.
6. Risk scoring.
7. Consensus logic.
8. Alert formatting and evidence packaging.

## Important design note

The risk agent and consensus layer need project-specific labels even if the upstream agents are pretrained. That is why public datasets alone will not finish the project.

