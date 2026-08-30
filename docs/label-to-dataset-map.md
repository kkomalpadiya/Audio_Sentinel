# Label to Dataset Map

This maps the threat categories in the project to realistic public data sources and shows where public data is still insufficient.

## Project labels

| Project label | Public dataset support | Notes |
|---|---|---|
| Gunshot / Explosion | UrbanSound8K, FSD50K, AudioSet reference | Public support exists, though real surveillance conditions may still differ |
| Scream / Distress | FSD50K, AudioSet reference, RAVDESS for vocal distress cues | Public support is partial; acted emotion is not the same as real emergency distress |
| Glass Breaking / Impact | FSD50K, AudioSet reference, ESC-50 | Public support exists for acoustic detection |
| Siren / Crowd Panic | UrbanSound8K, FSD50K, AudioSet reference | Siren is covered better than crowd panic |
| Normal Environmental Sound | FSD50K, ESC-50, AudioSet reference, MUSAN noise | Good public coverage |
| Speech Present / No Speech | MUSAN, Common Voice | Suitable for gating and robustness checks |
| Threat Phrase | Custom dataset needed | Public datasets are not specific enough |
| Weapon Reference | Custom dataset needed | Requires project-specific text labels |
| Harmless Context / Negation | Custom dataset needed | Needed for phrases such as "do not shoot" or movie dialogue |
| Final Risk Score | Custom dataset needed | This is your own project outcome |
| Final Consensus Decision | Custom dataset needed | This belongs to your architecture, not to public benchmarks |

## Bottom line

Public datasets are enough to begin the acoustic and speech branches, but not enough to finish the final risk and consensus stages. Those final layers need a custom labeled dataset built around your threat-definition policy.

