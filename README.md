# Constrained-Variable Motor Insurance Underwriting: A Framework of Strategies by Data Constraint Type

> **Anonymized repository.** This repository has been anonymized for double-blind peer review. Author names, affiliations, and the target venue are withheld. All local file paths, environment names, and machine-specific details from the original development environment have been generalized to repository-relative paths.

## Overview

This repository contains the full analysis pipeline, notebooks, and results for a study of how motor insurers with variable-constrained data can adopt strategies suited to the specific type of constraint they face, rather than assuming a single "more data is better" prescription.

Instead of asking "which model performs best," the study treats three publicly available motor insurance datasets as parallel case studies, each representing a different kind of data constraint:

| Case | Dataset | Constraint profile | Central question |
|---|---|---|---|
| **A** | `fremotor2freq9907b` (CASdatasets) | Few static variables, but long per-policyholder history (up to 9 years) | Can longitudinal history compensate for thin static features — and if so, what part of that history actually matters? |
| **B** | `car_insurance` (Kaggle) | Rich demographic/credit variables, but no longitudinal history (cross-sectional) | Does demographic richness substitute for longitudinal depth? |
| **C** | `freMTPL2` (CASdatasets) | More static variables than Case A, but no reliable multi-year re-observation of the same policyholder | Can a single compressed history proxy (a bonus-malus-style variable) substitute for genuine multi-year history? |

Each case is evaluated with a matched pair (or set) of models — a variable-scope baseline vs. an extended version — under a consistent evaluation protocol, then cross-checked with a second, architecturally distinct model family to confirm the finding is not an artifact of one algorithm.

## Key findings

- **Elapsed time alone is not the mechanism.** In Case A, adding a simple "years observed" feature to the model barely moved performance (Δ AUC < 0.001). Adding features that *summarize what happened* during that history (cumulative claim rate, prior-claim flags, etc.) produced a statistically significant improvement (paired bootstrap, p < 0.001), reproduced independently across two architecturally different model families.
- **A single compressed history proxy can be highly efficient.** In Case C, the bonus-malus variable — which compresses multi-year claim history into a single score — improved performance by a statistically significant margin comparable to or exceeding the *entire* multi-feature summary used in Case A.
- **Demographic access can outweigh longitudinal depth.** In Case B, demographic variables alone outperformed the best longitudinal model from Case A, without any statistically significant benefit from adding further behavioral history.
- **Governance costs may be a bigger constraint than predictive performance.** A fairness audit using a real protected attribute (available only in Case B) produced defensible, interpretable results. The same audit, performed on Case A using a non-protected proxy cohort (a vehicle characteristic, since no protected attribute exists in that dataset), surfaced a statistically confirmed pattern — after decomposing a global calibration bias from group-specific effects — in which the model systematically amplifies real risk differences across the proxy groups. This suggests that thin-variable environments may be harder to govern responsibly than they are to predict accurately.
- Full methodological detail, including a documented case where a look-ahead leakage bug was found and corrected (an initial run showing an implausible AUC of ~0.99 was traced to a missing causal attention mask and corrected to ~0.79, subsequently cross-validated against an independent tree-based model), is in `docs/research_plan.md`.

## Repository structure

```
.
├── notebooks/
│   ├── 00_data_collection.ipynb                    # acquires all three source datasets
│   ├── 01_eda_preprocessing.ipynb                   # Case A: EDA, longitudinal split construction
│   ├── 02_sequence_builder.ipynb                    # Case A: train/val/test split (shared across all Case A models)
│   ├── 03a_lightgbm_static.ipynb                    # Case A, Model ①
│   ├── 03b_lightgbm_longitudinal.ipynb               # Case A, Model ②
│   ├── 03_transformer_longitudinal.ipynb             # Case A, Model ④ (causal-mask corrected)
│   ├── 03c_lightgbm_aggregate.ipynb                  # Case A, Model ③c
│   ├── 04_aggregate_features.ipynb                   # Case A: engineered longitudinal summary features
│   ├── 04a_transformer_static.ipynb                  # Case A, Model ③ (causal-mask corrected)
│   ├── 04b_transformer_aggregate.ipynb               # Case A, Model ④b (causal-mask corrected)
│   ├── 05_case_b_car_insurance_eda.ipynb             # Case B: preprocessing
│   ├── 06a_case_b_lightgbm_demographic.ipynb         # Case B, Model B1
│   ├── 06b_case_b_lightgbm_full.ipynb                # Case B, Model B2
│   ├── 07_case_c_fremtpl2_eda.ipynb                  # Case C: preprocessing
│   ├── 08a_case_c_lightgbm_static.ipynb              # Case C, Model C1
│   ├── 08b_case_c_lightgbm_bonusmalus.ipynb          # Case C, Model C2
│   ├── 09_governance_shap.ipynb                      # SHAP analysis, all three cases
│   ├── 10_governance_fairness_case_a.ipynb           # fairness audit, Case A (proxy cohort)
│   ├── 11_governance_fairness_case_b.ipynb           # fairness audit, Case B (protected attribute)
│   ├── 12_paper_assets_export.ipynb                  # exports all publication tables/figures
│   ├── 13_statistical_significance.ipynb             # paired bootstrap significance testing
│   ├── 14a_case_b_mlp.ipynb                          # Case B architecture contrast (MLP)
│   ├── 14b_case_c_mlp.ipynb                          # Case C architecture contrast (MLP)
│   └── 15_case_a_vehtype_diagnosis.ipynb             # global/local calibration bias decomposition follow-up
├── outputs/
│   ├── tables/       # all result tables referenced in the paper (CSV)
│   └── figures/       # all figures referenced in the paper (PNG)
├── docs/
│   └── research_plan.md   # full versioned research log (design decisions, results, revisions)
└── references.bib         # bibliography (BibTeX), including the theoretical framework citations
```

**Note on data:** raw source data is not redistributed in this repository. `fremotor2freq9907b` and `freMTPL2` are distributed via the [CASdatasets](https://dutangc.perso.math.cnrs.fr/RRepository/pub/) R package; `car_insurance` is distributed via [Kaggle](https://www.kaggle.com/) (`sagnik1511/car-insurance-data`). `notebooks/00_data_collection.ipynb` reproduces acquisition of all three from their public sources — see the notebook for exact retrieval steps.

## Reproducing the analysis

Notebooks are numbered in execution order. Within a case, models sharing a preprocessing notebook (e.g. all Case A LightGBM/Transformer variants) use an identical train/validation/test split, recorded once and reused, so that results are directly comparable across models within a case.

1. Run `00_data_collection.ipynb` to acquire the three source datasets.
2. For Case A: run `01` → `02`, then any of `03a`/`03b`/`03c`/`03`/`04a`/`04b` in any order (each is self-contained given `01`–`02`'s outputs). `04` (aggregate feature engineering) must run before `03c` and `04b`.
3. For Case B: run `05`, then `06a`/`06b`.
4. For Case C: run `07`, then `08a`/`08b`.
5. Governance and follow-up analyses (`09`–`11`, `15`) require the relevant case's model(s) to already be trained and saved.
6. `12`–`14b` are post-hoc analyses that read saved model/prediction artifacts rather than retraining; they can be run once the corresponding case's models exist.

## Methodological notes

- All pairwise model comparisons that inform a claim of "improvement" are backed by paired bootstrap significance testing (1,000 resamples, 95% CI) rather than point-estimate comparison alone — see `13_statistical_significance.ipynb` and `docs/research_plan.md` §4.8.
- Architecture-selection decisions (e.g., using a single tree-based model for two of the three cases) are justified with an explicit contrast-architecture check rather than assumed from a single case's result — see `14a`/`14b` and `docs/research_plan.md` §4.6.
- A documented instance of a methodological bug (missing causal attention mask causing look-ahead leakage) and its correction is preserved in the notebook history and research log rather than silently fixed, per the project's emphasis on reproducibility over presentation — see `docs/research_plan.md` §4.5.
- The theoretical framework connecting the empirical findings to classical actuarial credibility theory (Bühlmann, Bühlmann–Straub, and their modern deep-learning extensions) is developed in `docs/research_plan.md` §2.2 and `references.bib`.

## Status

This is an active research repository; see `docs/research_plan.md` for the versioned log of design decisions, completed analyses, and remaining work.

## License

*(To be added prior to public release. Code and analysis are made available for review purposes during the anonymized review period.)*
