# =============================================================================
# STAGE 5: Conclusion & Final Report
# The Sound of Sadness: ML vs. Statistical Inference in Music-Induced Mental Health
# =============================================================================
# This script re-loads all saved outputs from Stages 1–4 and prints a
# structured written conclusion for your report.  It also produces a final
# summary table and a combined comparison figure (XGBoost importance vs.
# regression significance) to use directly in your write-up.
# Run AFTER all previous stages.
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
pkgs <- c("dplyr", "ggplot2", "knitr")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos='https://cran.r-project.org')
}

library(dplyr)
library(ggplot2)
library(knitr)

# ── 1. Load Artefacts from Previous Stages ───────────────────────────────────
if (!file.exists("df_clean.rds")) stop("Run 01_data_cleaning.R first.")

df_clean        <- readRDS("df_clean.rds")
top_features    <- if (file.exists("top_features.rds")) readRDS("top_features.rds") else NULL
xgb_importance  <- if (file.exists("xgb_feature_importance.csv"))
                     read.csv("xgb_feature_importance.csv") else NULL

cat("Data and artefacts loaded.\n\n")

# ── 2. Re-run Final Regression to Gather Numbers ─────────────────────────────
# (Avoids carrying objects between scripts; takes only a few milliseconds)
# Reproduce the same dynamic feature selection logic used in Stage 4.
demo_cols <- c("age", "log_hours", "hours_per_day", "bpm_clean",
               "anxiety", "insomnia", "ocd")

if (!is.null(top_features)) {
  music_feature <- top_features[
    top_features %in% names(df_clean) &
    !top_features %in% demo_cols
  ][1]
} else {
  music_feature <- NULL
}

if (is.null(music_feature) || is.na(music_feature)) {
  music_feature <- names(df_clean)[grepl("rock", names(df_clean), ignore.case = TRUE)][1]
}

if (is.ordered(df_clean[[music_feature]])) df_clean[[music_feature]] <- as.integer(df_clean[[music_feature]])

final_model <- lm(
  as.formula(paste("depression ~ log_hours +", music_feature, "+ age")),
  data = df_clean
)
mlr_sum  <- summary(final_model)
mlr_coef <- as.data.frame(mlr_sum$coefficients)
mlr_coef$term <- rownames(mlr_coef)

# ── 3. Re-run Hypothesis Tests for Numbers ────────────────────────────────────
df_clean <- df_clean %>%
  mutate(instrumentalist_f = factor(instrumentalist, levels = c(0,1),
                                    labels = c("Non-Instrumentalist", "Instrumentalist")))

wilcox_result <- wilcox.test(anxiety ~ instrumentalist_f, data = df_clean,
                              exact = FALSE, conf.int = TRUE)

df_anova <- df_clean %>% group_by(fav_genre) %>% filter(n() > 15) %>% ungroup()
anova_model <- aov(depression ~ fav_genre, data = df_anova)
anova_p     <- summary(anova_model)[[1]][["Pr(>F)"]][1]
eta_sq      <- sum((fitted(anova_model) - mean(df_anova$depression))^2) /
               (sum((fitted(anova_model) - mean(df_anova$depression))^2) +
                sum(anova_model$residuals^2))

# ── 4. Summary Results Table ─────────────────────────────────────────────────
summary_table <- data.frame(
  Hypothesis = c(
    "H1: Musician's Effect",
    "H2: Genre Correlation",
    "H3: XGBoost Audit – log_hours",
    paste0("H3: XGBoost Audit – ", music_feature)
  ),
  Test = c(
    "Wilcoxon rank-sum",
    "One-way ANOVA",
    "Multiple Linear Regression",
    "Multiple Linear Regression"
  ),
  Statistic = c(
    sprintf("W = %.0f", wilcox_result$statistic),
    sprintf("F(%d,%d) = %.2f",
            summary(anova_model)[[1]]$Df[1],
            summary(anova_model)[[1]]$Df[2],
            summary(anova_model)[[1]]$`F value`[1]),
    sprintf("β = %.3f", mlr_coef["log_hours", "Estimate"]),
    sprintf("β = %.3f", mlr_coef[grep(music_feature, mlr_coef$term, fixed=TRUE), "Estimate"])
  ),
  p_value = c(
    round(wilcox_result$p.value, 4),
    round(anova_p, 4),
    round(mlr_coef["log_hours", "Pr(>|t|)"], 4),
    round(mlr_coef[grep(music_feature, mlr_coef$term, fixed=TRUE), "Pr(>|t|)"], 4)
  ),
  Effect_Size = c(
    sprintf("r_rb = %.3f",
            ((wilcox_result$statistic - sum(df_clean$instrumentalist==0)*sum(df_clean$instrumentalist==1)/2) /
             sqrt(sum(df_clean$instrumentalist==0)*sum(df_clean$instrumentalist==1)*(sum(df_clean$instrumentalist==0)+sum(df_clean$instrumentalist==1)+1)/12)) /
            sqrt(nrow(df_clean))),
    sprintf("η² = %.4f", eta_sq),
    sprintf("R² = %.3f", mlr_sum$r.squared),
    sprintf("R² = %.3f", mlr_sum$r.squared)
  ),
  Decision = c(
    ifelse(wilcox_result$p.value < 0.05, "Reject H0 ✓", "Fail to Reject H0"),
    ifelse(anova_p < 0.05,               "Reject H0 ✓", "Fail to Reject H0"),
    ifelse(mlr_coef["log_hours", "Pr(>|t|)"] < 0.05, "Reject H0 ✓", "Fail to Reject H0"),
    ifelse(mlr_coef[grep(music_feature, mlr_coef$term, fixed=TRUE), "Pr(>|t|)"] < 0.05,
           "Reject H0 ✓", "Fail to Reject H0")
  )
)

cat("╔═══════════════════════════════════════════════════════════════╗\n")
cat("║  RESULTS SUMMARY TABLE                                       ║\n")
cat("╚═══════════════════════════════════════════════════════════════╝\n\n")
print(summary_table, row.names = FALSE)
cat("\n")

write.csv(summary_table, "results_summary_table.csv", row.names = FALSE)
cat("Saved: results_summary_table.csv\n\n")

# ── 5. XGBoost vs Regression Comparison Plot ─────────────────────────────────
if (!is.null(xgb_importance)) {
  # Normalise XGBoost gain to 0–1
  xgb_top <- xgb_importance %>%
    slice(1:10) %>%
    mutate(
      Gain_norm = Gain / max(Gain),
      source    = "XGBoost Gain (normalised)"
    )

  # Get MLR coefficients (absolute t-statistic as "importance")
  mlr_imp <- mlr_coef %>%
    filter(term != "(Intercept)") %>%
    transmute(
      Feature   = term,
      Gain_norm = abs(`t value`) / max(abs(`t value`)),
      source    = "MLR |t-statistic| (normalised)"
    )

  # Find features present in both
  common_features <- intersect(xgb_top$Feature, mlr_imp$Feature)

  if (length(common_features) > 0) {
    compare_df <- bind_rows(
      xgb_top %>% filter(Feature %in% common_features) %>%
        select(Feature, Gain_norm, source),
      mlr_imp %>% filter(Feature %in% common_features) %>%
        select(Feature, Gain_norm, source)
    )

    p_compare <- ggplot(compare_df,
                        aes(x = reorder(Feature, Gain_norm), y = Gain_norm,
                            fill = source)) +
      geom_col(position = "dodge") +
      coord_flip() +
      scale_fill_manual(values = c("#4e79a7", "#e15759")) +
      labs(
        title    = "XGBoost Importance vs. MLR Significance",
        subtitle = "Features ranked by normalised importance in each method",
        x = NULL, y = "Normalised Score (0–1)", fill = NULL
      ) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom",
            plot.title = element_text(face = "bold"))

    ggsave("plot_05a_xgb_vs_mlr_comparison.png", p_compare, width = 8, height = 5)
    cat("Plot saved: plot_05a_xgb_vs_mlr_comparison.png\n\n")
  } else {
    cat("No common features between XGBoost top-10 and MLR predictors to compare.\n\n")
  }
}

# ── 6. Written Conclusion Template ───────────────────────────────────────────
cat("╔═══════════════════════════════════════════════════════════════╗\n")
cat("║  WRITTEN CONCLUSION  (fill in your observed values below)    ║\n")
cat("╚═══════════════════════════════════════════════════════════════╝\n\n")

h1_p   <- wilcox_result$p.value
h1_dec <- ifelse(h1_p < 0.05, "significant", "not significant")

h2_dec <- ifelse(anova_p < 0.05, "significant", "not significant")

lh_p   <- mlr_coef["log_hours", "Pr(>|t|)"]
rk_p   <- mlr_coef[grep(music_feature, mlr_coef$term, fixed=TRUE), "Pr(>|t|)"]
lh_sig <- ifelse(lh_p  < 0.05, "statistically significant", "NOT statistically significant")
rk_sig <- ifelse(rk_p  < 0.05, "statistically significant", "NOT statistically significant")

cat(sprintf('
 ─────────────────────────────────────────────────────────────────
 5. CONCLUSION
 ─────────────────────────────────────────────────────────────────

 This project investigated whether music-consumption variables flagged
 as important by an XGBoost black-box model retain genuine statistical
 significance when subjected to rigorous classical inference.

 Hypothesis 1 — The Musician\'s Effect
 A Wilcoxon rank-sum test revealed that the difference in anxiety scores
 between instrumentalists and non-instrumentalists was %s
 (W = %.0f, p = %.4f).  %s

 Hypothesis 2 — The Genre Correlation
 A one-way ANOVA found that mean depression scores were %s
 across favourite-genre groups (F(%d, %d) = %.2f, p = %.4f, η² = %.4f).
 %s

 Hypothesis 3 — Auditing XGBoost
 After controlling for Age in a multiple linear regression, log-transformed
 daily listening hours was %s (β = %.3f, p = %.4f), and
 rock-listening frequency was %s (β = %.3f, p = %.4f).
 The overall model explained %.1f%% of variance in depression (R² = %.3f).

 Overall Interpretation
 %s

 These findings illustrate a core limitation of black-box ML models:
 feature importance reflects predictive correlation, not causal or even
 independently significant relationships.  Age, as a demographic confounder,
 may absorb variance that XGBoost attributes to listening behaviour.
 Future work should use causal inference frameworks (e.g., propensity score
 matching) or longitudinal designs to establish direction of effect.
 ─────────────────────────────────────────────────────────────────\n',

 h1_dec, wilcox_result$statistic, h1_p,
 ifelse(h1_p < 0.05,
   "We reject H0: instrumentalists report statistically different anxiety levels.",
   "We fail to reject H0: no significant anxiety difference was detected."),

 h2_dec,
 summary(anova_model)[[1]]$Df[1], summary(anova_model)[[1]]$Df[2],
 summary(anova_model)[[1]]$`F value`[1], anova_p, eta_sq,
 ifelse(anova_p < 0.05,
   "We reject H0: at least one genre group differs in depression (Tukey HSD identifies specific pairs).",
   "We fail to reject H0: genre preference does not significantly predict depression."),

 lh_sig, mlr_coef["log_hours", "Estimate"], lh_p,
 rk_sig, mlr_coef[grep(music_feature, mlr_coef$term, fixed=TRUE), "Estimate"], rk_p,
 mlr_sum$r.squared * 100, mlr_sum$r.squared,

 ifelse(lh_p >= 0.05 & rk_p >= 0.05,
   "XGBoost over-valued listening hours and rock frequency as predictors of depression.\n Once Age is controlled for, these features lose statistical significance,\n suggesting they were acting as proxies for age-related variance.",
   "At least one XGBoost feature retains significance after controlling for Age,\n suggesting it has an independent, statistically meaningful relationship with depression\n beyond what is explained by demographic factors alone.")
))

cat("═══════════════════════════════════════════════\n")
cat("  Stage 5 complete.  Outputs:\n")
cat("    results_summary_table.csv\n")
cat("    plot_05a_xgb_vs_mlr_comparison.png\n")
cat("    (conclusion text printed above)\n")
cat("═══════════════════════════════════════════════\n")
