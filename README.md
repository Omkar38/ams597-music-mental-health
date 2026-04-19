# The Sound of Sadness
### Music Listening Behaviour and Mental Health: A Statistical Analysis
**AMS 597 Statistical Computing — Spring 2026 | Stony Brook University**

*Aarushi Bahri · Jai Nilesh Vasi · Omkar Vilas Lashkare · Prerana Somani · Sanket Jaysing Bendale*

---

## Overview

This project investigates whether music-listening intensity and genre preferences are
statistically associated with self-reported mental health indicators (anxiety, depression,
insomnia, OCD). We use a two-pronged approach:

1. **XGBoost (ML)** to identify which variables are predictively important
2. **Classical statistical inference** (Wilcoxon / ANOVA / MLR) to determine whether those
   variables are genuinely significant after controlling for confounders

**Dataset**: [MXMH Survey Results](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)
— 736 respondents · 33 variables · collected Aug–Nov 2022

---

## Key Results

| Model | RMSE | MAE | R² | Dataset |
|-------|------|-----|----|---------|
| XGBoost | 2.26 | 1.85 | **0.40** | 20% holdout |
| Multiple Linear Regression | 2.94 | 2.53 | 0.054 | Full training set |

*Depression scale is 0–10. XGBoost explains 40% of held-out variance vs 5.4% for MLR.*

| Hypothesis | Test | Result | Decision |
|------------|------|--------|----------|
| H1: Instrumentalist → Anxiety | Wilcoxon rank-sum | W=56527, p=0.405, r=−0.031 | Fail to Reject H₀ |
| H2: Genre → Depression | One-Way ANOVA + Kruskal-Wallis | F(12,703)=1.80, p=0.045, η²=0.030 | **Reject H₀** |
| H3: log_hours → Depression (MLR) | Multiple Linear Regression | β=0.535, p=0.008 | **Reject H₀** |
| H3: frequency_metal → Depression (MLR) | Multiple Linear Regression | β=0.425, p<0.001 | **Reject H₀** |

> **Core finding**: XGBoost correctly identified genuine statistical signals — the
> top music-behaviour features survive classical inference even after controlling for age.

---

## Repository Structure

```
.
├── report.Rmd                            ← MAIN REPORT — compile this for submission
├── Coding_preprocessing_fixed.Rmd       ← Preprocessing detail (Prerana Somani)
│
├── 00_run_all.R                          ← Master runner: sources all pipeline stages
├── 01_data_cleaning.R                    ← Stage 1: Load, clean, impute, EDA
├── 02_xgboost_feature_importance.R       ← Stage 2: XGBoost, feature importance, CV
├── 03_assumption_checking.R              ← Stage 3: Shapiro-Wilk, Q-Q, skewness
├── 04_hypothesis_tests.R                 ← Stage 4: Wilcoxon / ANOVA / KW / MLR
├── 05_conclusion.R                       ← Stage 5: Summary table, accuracy, conclusion
│
├── mxmh_survey_results.csv              ← Raw survey data (source: Kaggle)
├── music_mental_health_survey_results.ipynb  ← Exploratory notebook
│
├── STAT PIPELINE.pdf                     ← Pipeline reference document
├── stat computing project.pdf            ← Project requirements
└── README.md
```

**For submission**: compile `report.Rmd` → it produces the full self-contained PDF/HTML
report covering all data cleaning, EDA, assumption checking, hypothesis tests, and conclusions.

---

## Research Questions

| # | Research Question | Method | Language |
|---|-------------------|--------|----------|
| RQ1 | Which music-listening features best predict depression? | XGBoost + 5-fold CV | R |
| RQ2 | Do instrumentalists differ in anxiety? Does genre predict depression? | Wilcoxon rank-sum · ANOVA + Kruskal-Wallis + Tukey HSD | R |
| RQ3 | Do XGBoost-flagged features remain significant after controlling for age? | Multiple Linear Regression + VIF + residual diagnostics | R |

---

## Pipeline

```
Raw CSV  →  Stage 1: Clean & EDA
             • check.names=FALSE preserves column names
             • BPM: regex extraction, range filter [40,250], median imputation
             • Frequency cols → ordered factor (0–3)
             • Boolean Yes/No → 0/1
             • log(hours+1) for right-skew correction
             • Correlation heatmap, distribution plots
             └─ df_clean.rds

         →  Stage 2: XGBoost Feature Importance         [RQ1]
             • 80/20 train/test split (seed = 2026)
             • early_stopping_rounds = 15
             • Accuracy: RMSE, MAE, R² on held-out test
             • 5-fold CV on training set only
             └─ top_features.rds, xgb_metrics.rds

         →  Stage 3: Assumption Checking
             • Shapiro-Wilk for all key variables
             • Q-Q plots, density comparisons

         →  Stage 4: Hypothesis Tests                   [RQ2, RQ3]
             • H1: Wilcoxon rank-sum (instrumentalist → anxiety)
             • H2: ANOVA + Levene + Kruskal-Wallis + Tukey HSD (genre → depression)
             • H3: MLR + VIF + residual diagnostics (XGBoost audit)

         →  Stage 5: Conclusion
             • Model accuracy comparison table (XGBoost vs MLR)
             • XGBoost importance vs MLR significance plot
             • Full written conclusion with actual p-values
```

---

## Setup

### Prerequisites

- R ≥ 4.1 · RStudio (recommended)

### Install Packages

```r
install.packages(c(
  "dplyr", "stringr", "ggplot2", "reshape2", "gridExtra",
  "xgboost", "Matrix", "Ckmeans.1d.dp",
  "moments", "car", "knitr", "kableExtra", "rmarkdown"
), repos = "https://cran.r-project.org")
```

---

## How to Run

### Option 1 — Compile the full report (recommended for submission)

```r
# In RStudio: open report.Rmd, then click Knit
# Or from the console:
setwd("path/to/project")
rmarkdown::render("report.Rmd", output_format = "pdf_document")
```

### Option 2 — Run the pipeline scripts individually

```r
setwd("path/to/project")   # must contain mxmh_survey_results.csv
source("00_run_all.R")     # runs all 5 stages in order (~60 sec total)
```

> All scripts use `set.seed(2026)` and require the working directory to be
> set to the project folder before running.

---

## Outputs

| Stage / File | Key Outputs |
|------|-------------|
| `report.Rmd` | Self-contained reproducible PDF/HTML report |
| Stage 01 | `df_clean.rds`, `df_clean.csv`, 5 EDA plots |
| Stage 02 | `xgb_metrics.rds`, `top_features.rds`, `xgb_feature_importance.csv`, 2 plots |
| Stage 03 | 8 Q-Q / density / histogram plots |
| Stage 04 | 4 hypothesis-test plots |
| Stage 05 | `results_summary_table.csv`, `model_accuracy_comparison.csv`, comparison plot |

Generated outputs (`.rds`, `.png`, `.csv`, `.bin`) are in `.gitignore` —
reproduce them by running `00_run_all.R`.

---

## Bugs Fixed in This Branch (`Preprocessed-code`)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `hours_per_day` not found | `read.csv` converts spaces to `.` before column cleaning, stripping underscores | Added `check.names = FALSE` to `read.csv` |
| xgboost crash | `watchlist` renamed to `evals` in xgboost ≥ 2.0 | Renamed parameter |
| CV crash | `nrounds = best_iteration` could be 0 or NULL | Guarded with `max(best_iteration, 10L)` |
| `r_rb = NA` | Named numeric from `wilcox.test$statistic` causes silent NA in arithmetic | Wrapped in `as.numeric()` |

---

## Methodology Notes

- **`music_effects` excluded from XGBoost**: direct self-report of music impact → data leakage
- **Genre filter n > 15**: too few observations make ANOVA cells unreliable
- **Dynamic feature for H3**: top non-demographic feature from XGBoost auto-selected — no hardcoding
- **Kruskal-Wallis alongside ANOVA**: H2 residuals are non-normal; both tests reported and checked for agreement
- **H2 borderline**: p = 0.045 is marginally significant; the corroborating Kruskal-Wallis strengthens the conclusion

---

*AMS 597 Statistical Computing — Spring 2026 · Stony Brook University*
