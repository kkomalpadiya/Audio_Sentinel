# Dataset Intake Checklist

For every dataset we include, record:

1. Official source URL
2. Citation or paper
3. License
4. Which agent it supports
5. Which labels from our project it covers
6. Gaps it does not cover
7. Local storage path inside `data/raw/`

## Folder rule

Downloaded datasets should go only into:

- `data/raw/`

Derived clips go into:

- `data/interim/`

Features and training-ready tables go into:

- `data/processed/`

## Panel-safe explanation

We are using a mix of:

- public benchmark datasets for general acoustic and speech capability
- pretrained models for baseline agent behavior
- a custom labeled dataset for project-specific threat scoring and consensus validation

