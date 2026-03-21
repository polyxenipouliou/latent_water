# Latent Water 🌊
### Comparing and Traversing Neural Audio Latent Spaces

**Polyxeni Pouliou** · Universitat Pompeu Fabra · [polyxeni.pouliou01@estudiant.upf.edu](mailto:polyxeni.pouliou01@estudiant.upf.edu)

---

## Overview

This project compares the latent spaces of two neural audio models — **EnCodec** and **RAVE** — and uses them as creative material for an audiovisual installation on the theme of washing.

We encode three water-themed pieces into both models, compare their representations using PCA, CKA, and MMD, and develop three artistic operations: temporal reordering, barycentric blending, and cross-space resynthesis.

---

## Dataset

Three water-themed pieces, 30-second segments extracted from the middle of each:

| # | Piece | Artist | Year |
|---|-------|--------|------|
| audio001–003 | *Jeux d'eau* | Maurice Ravel | 1901 |
| audio004–006 | *Wasserklavier* | Luciano Berio | 1965 |
| audio007–009 | *Ocean Eyes* | Billie Eilish & blackbear | 2016 |

Resampled versions: 24kHz for EnCodec (audio004–006), 44100Hz for RAVE (audio007–009).

---

## Models

**EnCodec** (`facebook/encodec_24khz`)
- Residual Vector Quantization (RVQ), 8 quantizers, 6kbps
- Latent shape: `(2250, 128)` per 30-second segment

**RAVE** (`vintage.ts` — pretrained on vintage instruments)
- Variational Autoencoder (VAE) + adversarial fine-tuning
- Latent shape: `(646, 16)` per 30-second segment

---

## Analysis

| Method | Purpose |
|--------|---------|
| PCA | 2D visualization of both latent spaces |
| CKA | Similarity index between EnCodec and RAVE representations |
| MMD | Distributional distance within and across spaces |

**Key finding:** EnCodec and RAVE encode fundamentally different aspects of audio. EnCodec shows limited discriminative power across styles (all MMD = 0.0040), while RAVE captures finer timbral distinctions between pieces.

---

## Artistic Extension

All outputs are numbered sequentially and available in the `/audio` folder.

### Temporal Reordering via Latent Similarity (audio010–015)
Audio divided into 15 chunks (~2s each), reordered by greedy nearest-neighbor algorithm in the EnCodec latent space.

### Barycentric Blending and Latent Reordering (audio016–018)
Latent vectors from all three pieces blended using Dirichlet-sampled weights, then reordered with the same NN algorithm.

| File | Ravel | Berio | Billie |
|------|-------|-------|--------|
| audio017 | 0.31 | 0.45 | 0.25 |
| audio018 | 0.02 | 0.60 | 0.38 |

### Cross-Space Resynthesis (audio019–022)
Temporal structure from EnCodec latent space of one piece applied to RAVE latents of a different piece, decoded by RAVE. Combines timbral identity from one model with temporal structure from another.

---

## Repository Structure

```
latent_water/
├── audio/                  # all audio outputs (001–022)
├── figures/                # PCA plots and other visualizations
├── latex/                  # ISMIR 2026 paper source
│   ├── ISMIR2026_fixed.tex
│   └── references.bib
├── utils/
│   └── ecdc_utils.py       # EnCodec utility functions
├── notebook.ipynb          # main Jupyter notebook
└── README.md
```

---

## Requirements

```bash
pip install torch torchaudio
pip install transformers
pip install audiocraft
pip install librosa soundfile
pip install scikit-learn matplotlib numpy
```

RAVE model: download `vintage.ts` from [acids-ircam.github.io/rave_models_download](https://acids-ircam.github.io/rave_models_download)

---

## References

- Défossez et al. (2022). *High Fidelity Neural Audio Compression.* arXiv:2210.13438
- Caillon & Esling (2021). *RAVE: A Variational Autoencoder for Fast and High-Quality Neural Audio Synthesis.* arXiv:2111.05011
- Kornblith et al. (2019). *Similarity of Neural Network Representations Revisited.* ICML.
- Raghu et al. (2017). *SVCCA.* NeurIPS.
- Bourriaud (2002). *Relational Aesthetics.* Les presses du réel.
- Wyse et al. (2022). *Sound Model Factory.* EvoStar.
