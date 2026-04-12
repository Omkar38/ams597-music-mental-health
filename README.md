# The Sound of Sadness
### Music Listening Behaviour and Mental Health: A Statistical Analysis
**AMS 597 Statistical Computing — Spring 2026 Group Project**
Stony Brook University | Instructor: Silvia Sharna

---

## Overview

This project investigates whether music-listening intensity and genre preferences are
statistically associated with self-reported mental health indicators (anxiety, depression,
insomnia, OCD). We use a two-pronged approach: a machine-learning model (XGBoost) to
identify which variables are predictively important, followed by classical statistical
inference to determine whether those variables are genuinely significant after controlling
for confounders.

**Dataset:** [MXMH Survey Results](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)
(736 respondents · 33 variables · collected Aug–Nov 2022)

---

## Research Questions

| # | Research Question | Method | Language |
|---|-------------------|--------|----------|
| RQ1 | Which music-listening features best predict depression scores? | XGBoost (ML) + 5-fold CV | R |
| RQ2 | Do instrumentalists differ from non-instrumentalists in anxiety? Does favourite genre predict depression? | Wilcoxon rank-sum test · One-way ANOVA + Kruskal-Wallis + Tukey HSD | R |
| RQ3 | Do the XGBoost-flagged features remain statistically significant after controlling for age? | Multiple Linear Regression | R |

---

## Repository Structure

```
.
├── 00_run_all.R                      ← Master runner: sources all stages in order
├── 01_data_cleaning.R                ← Stage 1: Load, clean, impute, EDA, save df_clean.rds
├── 02_xgboost_feature_importance.R   ← Stage 2: XGBoost training, feature importance, CV
├── 03_assumption_checking.R          ← Stage 3: Shapiro-Wilk, Q-Q plots, skewness/kurtosis
├── 04_hypothesis_tests.R             ← Stage 4: Wilcoxon / ANOVA / Kruskal-Wallis / MLR
├── 05_conclusion.R                   ← Stage 5: Summary table, comparison plot, conclusion
├── mxmh_survey_results.csv           ← Raw survey data (source: Kaggle)
└── README.md
```

---

## Pipeline at a Glance

```
Raw CSV
  │
  ▼
Stage 1 ── Data Cleaning & EDA
  │         • BPM extraction & median imputation
  │         • Frequency columns → ordered factor (0–3)
  │         • Boolean Yes/No → 0/1
  │         • Log-transform hours_per_day
  │         • Correlation heatmap (EDA)
  │         └─ saves: df_clean.rds
  │
  ▼
Stage 2 ── XGBoost Feature Importance          (RQ1)
  │         • 80/20 train/test split
  │         • Early stopping (rounds = 15)
  │         • Feature importance by Gain
  │         • 5-fold CV on training set
  │         └─ saves: top_features.rds, xgb_feature_importance.csv
  │
  ▼
Stage 3 ── Assumption Checking
  │         • Shapiro-Wilk normality tests
  │         • Q-Q plots for all key variables
  │         • Density comparison: raw vs log-transformed hours
  │
  ▼
Stage 4 ── Statistical Hypothesis Tests        (RQ2, RQ3)
  │         • H1: Wilcoxon rank-sum (instrumentalist → anxiety)
  │         • H2: One-way ANOVA + Kruskal-Wallis + Tukey HSD (genre → depression)
  │         • H3: Multiple Linear Regression (XGBoost audit)
  │
  ▼
Stage 5 ── Conclusion & Summary
            • Results summary table
            • XGBoost importance vs. MLR significance comparison plot
            • Written conclusion
```

---

## Setup & Requirements

### Prerequisites

- R ≥ 4.1
- RStudio (recommended) or any R environment

### Install Packages

Run once in the R console:

```r
install.packages(c(
  "dplyr", "stringr", "ggplot2", "reshape2",
  "xgboost", "Matrix", "Ckmeans.1d.dp",
  "moments", "car", "knitr"
))
```

All scripts also auto-install missing packages on first run.

---

## How to Run

### Option 1 — Run everything at once (recommended)

1. Open `00_run_all.R` in RStudio
2. **Session → Set Working Directory → To Source File Location**
3. Click **Source** (or press `Ctrl+Shift+S`)

This runs all five stages in order and prints timing for each.

### Option 2 — Run stages individually

Set your working directory to the project folder first, then source each file in order:

```r
setwd("path/to/project")        # adjust to your path

source("01_data_cleaning.R")
source("02_xgboost_feature_importance.R")
source("03_assumption_checking.R")
source("04_hypothesis_tests.R")
source("05_conclusion.R")
```

> **Important:** All scripts must be run from the same working directory as
> `mxmh_survey_results.csv`. Each stage depends on the `.rds` files produced
> by the previous stage.

---

## Outputs

Each stage writes clearly named files to the working directory:

| Stage | Output Files |
|-------|-------------|
| 01 | `df_clean.rds`, `df_clean.csv`, `plot_01a_depression_dist.png`, `plot_01b_anxiety_dist.png`, `plot_01c_hours_dist.png`, `plot_01d_log_hours_dist.png`, `plot_01e_correlation_heatmap.png` |
| 02 | `xgb_feature_importance.csv`, `top_features.rds`, `xgb_depression_model.bin`, `plot_02a_xgb_importance.png`, `plot_02b_xgb_cv_curve.png` |
| 03 | `plot_03a_qq_*.png` (6 files), `plot_03b_hours_density_compare.png`, `plot_03c_depression_hist_normal.png`, `plot_03d_anxiety_hist_normal.png` |
| 04 | `plot_04a_h1_anxiety_violin.png`, `plot_04b_h2_depression_by_genre.png`, `plot_04c_h3_regression_diagnostics.png`, `plot_04d_h3_coef_plot.png` |
| 05 | `results_summary_table.csv`, `plot_05a_xgb_vs_mlr_comparison.png` |

---

## Methodology Summary

### Data Cleaning (Stage 1)

- **BPM**: Extracted leading integer using regex; values outside 40–250 BPM set to `NA`
  and imputed with the column median (robust to outliers)
- **Frequency columns**: Converted from text (`"Never"`, `"Rarely"`, `"Sometimes"`,
  `"Very frequently"`) to ordered factors, then to integers 0–3 for modelling
- **Boolean columns**: `Yes`/`No` recoded to `1`/`0`
- **Rows dropped**: Those missing `depression`, `anxiety`, `age`, or `hours_per_day`
- **Derived variable**: `log_hours = log(hours_per_day + 1)` to correct right skew

### Machine Learning (Stage 2)

XGBoost (`reg:squarederror`) trained on an 80/20 train/test split with early stopping.
Feature importance is measured by **Gain** (reduction in squared-error loss per split).
5-fold cross-validation is performed on the training set only.

### Hypothesis Tests (Stage 4)

| Hypothesis | IV | DV | Test | Why |
|------------|----|----|------|-----|
| H1 | Instrumentalist (0/1) | Anxiety (0–10) | Wilcoxon rank-sum | Anxiety is bounded integers, confirmed non-normal |
| H2 | Favourite genre | Depression (0–10) | ANOVA + Kruskal-Wallis + Tukey HSD | Kruskal-Wallis as non-parametric check; Tukey HSD for pairwise comparisons |
| H3 | `log_hours` + top XGBoost music feature + age | Depression | Multiple Linear Regression | Audits whether ML-flagged variables survive classical inference |

Effect sizes reported: rank-biserial *r* (H1), η² (H2), R² (H3).

---

## Key Design Decisions

- **`music_effects` excluded from XGBoost**: This column asks respondents whether music
  improved/worsened their mood — including it would be data leakage
- **Genre filter (n > 15)**: Genres with too few respondents produce unreliable ANOVA
  cells and are excluded from H2
- **Dynamic feature for H3**: The top non-demographic feature from XGBoost's ranked list
  is used automatically — no hardcoding
- **Kruskal-Wallis alongside ANOVA**: Since ANOVA residuals are likely non-normal
  (bounded integer outcome), both tests are reported and checked for agreement
- **`set.seed(2026)`**: Applied globally for full reproducibility

---

## Reproducibility

All analyses are run in R (≥ 4.1). The global seed is `set.seed(2026)`.
Running `source("00_run_all.R")` from the project directory should reproduce every
plot, table, and printed result exactly.

---

## Notes on the Dataset

- **Source**: Kaggle — [catherinerasgaitis/mxmh-survey-results](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)
- **Collection period**: August 27 – November 9, 2022 (online survey)
- **Sample**: 736 anonymised respondents
- **Key variables**: Age, Hours per day, Primary streaming service, Favourite genre,
  16 genre-frequency columns (ordinal), BPM (self-reported, messy), Anxiety (0–10),
  Depression (0–10), Insomnia (0–10), OCD (0–10), Music effects (Improve/No effect/Worsen)

---

*AMS 597 Statistical Computing — Spring 2026*
*Stony Brook University*
