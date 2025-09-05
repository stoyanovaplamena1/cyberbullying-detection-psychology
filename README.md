# Cyberbullying & Toxicity Detection (TF–IDF + Linear Models)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/stoyanovaplamena1/cyberbullying-detection-psychology)

This repo tracks a simple, reproducible pipeline for:
- **Cyberbullying (tweets)** — binary + multiclass (by `cyberbullying_type`)
- **Wikipedia/Jigsaw toxicity** — multilabel (toxic, severe_toxic, obscene, threat, insult, identity_hate)

The goal: a transparent TF–IDF baseline with a few lightweight psycholinguistic features, solid splits, and clean evaluation. If the linear stack plateaus, we escalate (calibration, late fusion with sentence embeddings, then transformers).

---

## Clone the repo

```bash
git clone https://github.com/stoyanovaplamena1/cyberbullying-detection-psychology.git
cd cyberbullying-detection-psychology
