# longitudinal-ubi-switchtab

> **Longitudinal UBI Underwriting Framework via SwitchTab Pretraining,
> Explainable AI, and LLM-Based Risk Narrative Generation**

A full end-to-end research pipeline for automobile insurance
underwriting using longitudinal panel data, self-supervised
representation learning, and AI governance components.

---

## Overview

This repository contains the complete experimental codebase
accompanying an anonymized academic manuscript submitted for
peer review.

The framework addresses a core limitation of existing auto
insurance underwriting models — their reliance on static,
cross-sectional features — by introducing a longitudinal
sequence modeling approach that captures temporal risk
dynamics at the individual policyholder level.

### Core Methodological Contribution

We extend **SwitchTab** (AAAI 2024), a self-supervised
tabular representation learning method, from its original
random-sample pairing to **same-policyholder adjacent-year
pairing (t, t+1)**. Under this design:

- **Mutual representation**: time-invariant individual risk
  trait (persistent driving behavior)
- **Salient representation**: year-specific risk deviation
  signal (transient risk fluctuation)

This is, to our knowledge, the first application of SwitchTab
to longitudinal insurance panel data.

---

## Research Pipeline

```
00_data_collection        → Data acquisition (3 sources)
01_eda_preprocessing      → EDA, derived features, split
02_sequence_builder       → Transformer-ready sequences
03_transformer_model      → Baseline Transformer (raw features)
04_switchtab_pretrain     → SwitchTab Phase1 + Phase2
05_transformer_switchtab  → Transformer (SwitchTab embeddings)
06_baseline_models        → LightGBM / LSTM comparison (7 models)
07_shap_analysis          → SHAP importance + fairness audit G1–G4
08_llm_narrative          → Rule-based + LLM narrative + LLM-as-Judge
```

---

## Datasets

Three publicly available datasets are used.
**No proprietary or personally identifiable data is included.**

| Dataset | Source | Role | Rows |
|---|---|---|---|
| `fremotor2freq9907b` | CASdatasets (R pkg) | Longitudinal panel — core modeling | 366,354 |
| `freMTPL2freq/sev` | HuggingFace (`mabilton/fremtpl2`) | Rate simulation reference | 678,013 |
| `car-insurance-data` | Kaggle (`sagnik1511`) | Pipeline prototype validation | 10,000 |

> **Note:** `fremotor2freq9907b` is downloaded automatically
> via the R script `get_casdatasets.R` included in this
> repository. An R installation (≥ 4.0) is required.

---

## Model Comparison Results

Seven models evaluated on the same test split
(IDpol-level, no leakage):

| Model | AUC-ROC | Best F1 |
|---|---|---|
| LightGBM-Raw [A] | **0.7687** | 0.3796 |
| Transformer-Raw [F] | 0.7680 | 0.3780 |
| Transformer-STab [G] | 0.7680 | 0.3790 |
| LightGBM-STab [B] | 0.7675 | 0.3797 |
| LSTM-STab [D] | 0.7671 | 0.3793 |
| SwitchTab Standalone [E] | 0.7668 | **0.3822** |
| LSTM-Raw [C] | 0.7581 | 0.3724 |

> SwitchTab Standalone achieves the best F1 score across all models,
> while maintaining competitive AUC-ROC performance.

---

## Requirements

### Python Environment

```bash
# Create conda environment
conda create -n ubi_switchtab python=3.10
conda activate ubi_switchtab

# Install dependencies
pip install -r requirements.txt
```

### R Environment (for CASdatasets only)

```r
# Run once inside R (≥ 4.0)
install.packages("CASdatasets",
  repos = "https://dutangc.perso.math.cnrs.fr/RRepository/pub/",
  type  = "source")
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

> `.env` is listed in `.gitignore` and is **never committed**.

---

## Repository Structure

```
longitudinal-ubi-switchtab/
│
├── notebooks/
│   ├── 00_data_collection.ipynb
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_sequence_builder.ipynb
│   ├── 03_transformer_model.ipynb
│   ├── 04_switchtab_pretrain.ipynb
│   ├── 05_transformer_switchtab.ipynb
│   ├── 06_baseline_models.ipynb
│   ├── 07_shap_analysis.ipynb
│   └── 08_llm_narrative.ipynb
│
├── get_casdatasets.R          # R script for CASdatasets download
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore
└── README.md
```

---

## Execution Order

Run notebooks sequentially from `00` to `08`.
Each notebook saves intermediate outputs to `data/`
which are loaded by subsequent notebooks.

```bash
# Recommended: run in order
jupyter notebook notebooks/00_data_collection.ipynb
jupyter notebook notebooks/01_eda_preprocessing.ipynb
# ... continue through 08
```

> **GPU note:** Notebook `04_switchtab_pretrain.ipynb`
> (Phase 1: 200 epochs, Phase 2: 50 epochs) requires
> a CUDA-enabled GPU. All other notebooks run on CPU.
> Tested on NVIDIA RTX 4060 Ti 16GB with CUDA 12.4.

---

## Key Components

### SwitchTab Longitudinal Extension (`04`)

Original SwitchTab pairs random samples. This study
constrains pairing to **same-IDpol, adjacent-year records**,
giving the mutual/salient decomposition an explicit
temporal interpretation:

```python
# Adjacent-year pairing (t, t+1)
if group.loc[i+1, "Year"] == group.loc[i, "Year"] + 1:
    pairs_x1.append(group.loc[i,   FEATURE_COLS].values)
    pairs_x2.append(group.loc[i+1, FEATURE_COLS].values)
```

### SHAP-Based Risk Grading (`07`)

Five-tier risk grading (G1\_Safe → G5\_Critical) derived
from IDpol-level mean SHAP values, with monotone positive
rate validation:

| Grade | n | Positive Rate |
|---|---|---|
| G1\_Safe | 2,142 | 7.1% |
| G2\_Caution | 2,142 | 15.5% |
| G3\_Risk | 2,141 | 31.9% |
| G4\_HighRisk | 2,142 | 50.8% |
| G5\_Critical | 2,142 | 71.9% |

### Fairness Audit G1–G4 (`07`)

Four fairness metrics computed across observation cohorts:

| Metric | Gap | Status |
|---|---|---|
| G1 Demographic Parity | 0.2165 | ⚠️ Structural bias* |
| G2 Equal Opportunity (TPR) | 0.0412 | ✅ Within threshold |
| G3 Predictive Parity | 0.3012 | ⚠️ Structural bias* |
| G4 High-Risk Grade Rate | 5.17%p | ⚠️ Structural bias* |

> *Gaps attributed to observation period differences across
> cohorts, not discriminatory model behavior.

### LLM Risk Narrative (`08`)

Two-stage narrative pipeline:

1. **Rule-based** (all 10,709 policyholders, zero cost)
2. **LLM enrichment** via GPT-4o-mini (sampled subset)
3. **LLM-as-a-Judge** quality evaluation
   (Accuracy / Utility / Fairness: 5.00 / 5.00 / 5.00)

---

## Reproducibility

All random seeds are fixed (`seed=42`).
IDpol-level train/val/test split (70/15/15) is saved as
pickle files and reused across all notebooks to guarantee
identical evaluation conditions.

---

## License

This repository is released for academic review purposes.
Author information is withheld pending peer review.

---

## Citation

```
[Citation information withheld for anonymous peer review]
```

---

## Acknowledgements

- **CASdatasets**: Dutang & Charpentier (R package)
- **SwitchTab**: Han et al., AAAI 2024
- **ts3l**: TabularS3L library (pip install ts3l)
- **freMTPL2**: Available via HuggingFace (mabilton/fremtpl2)
