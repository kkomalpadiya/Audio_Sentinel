# Dataset Sources for Panel Review

This document lists the datasets we can justify for the current system design. The goal is not to download everything, but to select sources that map clearly to the agents in the architecture.

## Recommended starter set

| Dataset | Official source | Why we need it | Main agent support | Notes |
|---|---|---|---|---|
| AudioSet | https://research.google.com/audioset/ | Large sound-event ontology and transfer-learning reference covering many environmental classes | Acoustic Event Detection Agent | Good reference source, but not the first dataset to operationalize locally because it is large and tied to YouTube clips |
| FSD50K | https://fsannotator.upf.edu/fsd/release/FSD50K/ | Open sound-event dataset with broad environmental labels | Acoustic Event Detection Agent | Strong starter dataset for open, reproducible experimentation |
| UrbanSound8K | https://urbansounddataset.weebly.com/urbansound8k.html | Includes classes such as `gun_shot` and `siren` | Acoustic Event Detection Agent | Useful for quick targeted baseline experiments |
| ESC-50 | https://github.com/karolpiczak/esc-50 | Small curated environmental benchmark | Acoustic Event Detection Agent | Best for quick smoke tests and comparisons |
| MUSAN | https://www.openslr.org/17/ | Speech, music, and noise corpus useful for robustness and VAD-style experiments | Preprocessing / Speech Gate | Helpful for noise augmentation and speech-vs-noise checks |
| Common Voice | https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk | Open speech corpus for general ASR evaluation | Speech Recognition Agent | Good for general speech robustness, not threat-specific language |
| RAVDESS | https://zenodo.org/records/1188976 | Emotional speech data with fear, anger, and related vocal expressions | Language Understanding / Distress Cue Support | Supports distress-style vocal emotion cues, not real-world threat dialogue |

## What each dataset does not solve

### Public datasets do not fully cover:

- suspicious spoken threat phrases
- harmless phrases containing dangerous words
- negation cases such as "do not shoot"
- realistic CCTV-distance emergency conversations
- your final risk score labels
- your final consensus decision labels

## Therefore we also need a custom dataset

### Custom project dataset purpose

We should create a small project-specific labeled set for:

- threat phrases
- distress phrases
- harmless-context speech
- negation examples
- fused examples with acoustic and language outputs
- final risk-score and alert-decision labels

## Suggested panel explanation

You can explain the data plan like this:

We use established public datasets for broad environmental sound recognition, speech robustness, and baseline evaluation. Because our final system combines multiple pretrained agents and makes a project-specific threat decision, we also prepare a custom labeled dataset for risk scoring and consensus validation. This lets us use recognized benchmarks while still training the parts that are unique to our system.

## Current shortlist

### Download first

1. FSD50K
2. UrbanSound8K
3. ESC-50
4. MUSAN

### Add after that

1. Common Voice
2. AudioSet metadata/reference materials
3. RAVDESS

### Build ourselves

1. Custom threat-speech and consensus-label dataset

