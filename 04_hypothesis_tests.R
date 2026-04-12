# =============================================================================
# STAGE 4: Statistical Inference – Three Hypothesis Tests
# The Sound of Sadness: ML vs. Statistical Inference in Music-Induced Mental Health
# =============================================================================
#
# Hypothesis 1 (Musician's Effect)
#   H0: No difference in anxiety between instrumentalists and non-instrumentalists
#   Test: Wilcoxon rank-sum (non-parametric, since anxiety is ordinal/skewed)
#
# Hypothesis 2 (Genre Correlation)
#   H0: Mean depression scores are equal across all primary genre groups
#   Test: One-way ANOVA + Tukey HSD post-hoc
#
# Hypothesis 3 (Auditing XGBoost)
#   H0: XGBoost top features are NOT significant after controlling for Age
#   Test: Multiple Linear Regression (depression ~ log_hours + frequency_rock + age)
#
# Run AFTER 01_data_cleaning.R and 02_xgboost_feature_importance.R
# (requires df_clean.rds)
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
pkgs <- c("dplyr", "ggplot2", "car")   # car: leveneTest
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

library(dplyr)
library(ggplot2)
library(car)

# ── 1. Load Clean Data ────────────────────────────────────────────────────────
if (!file.exists("df_clean.rds")) {
  stop("'df_clean.rds' not found. Please run 01_data_cleaning.R first.")
}
df_clean <- readRDS("df_clean.rds")
cat("Clean data loaded:", nrow(df_clean), "rows\n\n")

# ═════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 1 – The Musician's Effect
# IV: instrumentalist (0/1)   DV: anxiety (0–10)
# ═════════════════════════════════════════════════════════════════════════════
cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║  HYPOTHESIS 1: The Musician's Effect (Instrumentalist)      ║\n")
cat("╚══════════════════════════════════════════════════════════════╝\n\n")

# Ensure instrumentalist is a clear factor
df_clean <- df_clean %>%
  mutate(instrumentalist_f = factor(instrumentalist,
                                    levels = c(0, 1),
                                    labels = c("Non-Instrumentalist", "Instrumentalist")))

# --- Descriptive stats ---
h1_desc <- df_clean %>%
  group_by(instrumentalist_f) %>%
  summarise(
    n      = n(),
    mean   = round(mean(anxiety, na.rm = TRUE), 3),
    median = median(anxiety, na.rm = TRUE),
    sd     = round(sd(anxiety, na.rm = TRUE), 3),
    .groups = "drop"
  )
cat("Descriptive Statistics – Anxiety by Instrumentalist Status:\n")
print(h1_desc)
cat("\n")

# --- Wilcoxon rank-sum test ---
# Preferred over t-test because anxiety scores are bounded integers
# (not truly continuous) and Shapiro-Wilk in Stage 3 likely flags non-normality.
wilcox_result <- wilcox.test(anxiety ~ instrumentalist_f,
                              data      = df_clean,
                              exact     = FALSE,   # avoid exact p with ties
                              conf.int  = TRUE)
cat("Wilcoxon Rank-Sum Test Result:\n")
print(wilcox_result)
cat("\n")

# Effect size: rank-biserial correlation r = Z / sqrt(N)
# wilcox.test does not return Z directly; compute from W
n1 <- sum(df_clean$instrumentalist == 0, na.rm = TRUE)
n2 <- sum(df_clean$instrumentalist == 1, na.rm = TRUE)
# Expected W under H0
W_obs  <- wilcox_result$statistic
mu_W   <- n1 * n2 / 2
sigma_W <- sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
Z_val  <- (W_obs - mu_W) / sigma_W
r_rb   <- Z_val / sqrt(n1 + n2)
cat(sprintf("Effect size (rank-biserial r) = %.3f\n", r_rb))
cat("  Interpretation: |r| < 0.1 = negligible, 0.1–0.3 = small,\n")
cat("                  0.3–0.5 = medium, > 0.5 = large\n\n")

# --- Visualisation ---
p_h1 <- ggplot(df_clean, aes(x = instrumentalist_f, y = anxiety,
                               fill = instrumentalist_f)) +
  geom_violin(alpha = 0.6, colour = NA) +
  geom_boxplot(width = 0.15, outlier.shape = NA,
               colour = "black", fill = "white", alpha = 0.7) +
  scale_fill_manual(values = c("#4e79a7", "#e15759")) +
  labs(
    title    = "Hypothesis 1: Anxiety by Instrumentalist Status",
    subtitle = sprintf("Wilcoxon rank-sum: W = %.0f, p = %.4f, r = %.3f",
                       W_obs, wilcox_result$p.value, r_rb),
    x = NULL, y = "Anxiety Score (0–10)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title    = element_text(face = "bold"),
        plot.subtitle = element_text(colour = "grey40"))

ggsave("plot_04a_h1_anxiety_violin.png", p_h1, width = 6, height = 5)
cat("Plot saved: plot_04a_h1_anxiety_violin.png\n\n")

# ═════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 2 – The Genre Correlation (ANOVA)
# IV: fav_genre   DV: depression (0–10)
# ═════════════════════════════════════════════════════════════════════════════
cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║  HYPOTHESIS 2: Genre Correlation (One-Way ANOVA)            ║\n")
cat("╚══════════════════════════════════════════════════════════════╝\n\n")

# Filter genres with sufficient sample size for reliable ANOVA cells
df_anova <- df_clean %>%
  group_by(fav_genre) %>%
  filter(n() > 15) %>%
  ungroup()

genres_kept <- unique(df_anova$fav_genre)
cat("Genres retained (n > 15):\n")
print(sort(genres_kept))
cat("Total rows in ANOVA dataset:", nrow(df_anova), "\n\n")

# --- Levene's Test (equal variances assumption) ---
levene_result <- leveneTest(depression ~ factor(fav_genre), data = df_anova)
cat("Levene's Test for Homogeneity of Variance:\n")
print(levene_result)
cat("(If p < 0.05 → variances are unequal → consider Welch ANOVA)\n\n")

# --- One-way ANOVA ---
anova_model <- aov(depression ~ fav_genre, data = df_anova)
cat("ANOVA Summary:\n")
print(summary(anova_model))
cat("\n")

# Eta-squared (effect size)
ss_total  <- sum(anova_model$residuals^2) + sum((fitted(anova_model) - mean(df_anova$depression))^2)
ss_effect <- sum((fitted(anova_model) - mean(df_anova$depression))^2)
eta_sq    <- ss_effect / (ss_effect + sum(anova_model$residuals^2))
cat(sprintf("Eta-squared (η²) = %.4f\n", eta_sq))
cat("  Interpretation: η² ≈ 0.01 = small, 0.06 = medium, 0.14 = large\n\n")

# --- Residual normality check ---
shapiro_anova_resid <- shapiro.test(
  sample(residuals(anova_model), min(5000, length(residuals(anova_model))))
)
cat("Shapiro-Wilk on ANOVA residuals:", sprintf("W = %.4f, p = %.4e\n\n",
    shapiro_anova_resid$statistic, shapiro_anova_resid$p.value))

# --- Kruskal-Wallis fallback (non-parametric ANOVA equivalent) ---------------
# If ANOVA residuals are non-normal (p < 0.05 on Shapiro-Wilk), the Kruskal-
# Wallis test is the appropriate non-parametric alternative.  We run it
# regardless and note whether the two tests agree.
kw_result <- kruskal.test(depression ~ factor(fav_genre), data = df_anova)
cat("Kruskal-Wallis Test (non-parametric ANOVA alternative):\n")
print(kw_result)
cat(sprintf("  → ANOVA and Kruskal-Wallis %s (both %s at α = 0.05)\n\n",
            ifelse(sign(anova_p - 0.05) == sign(kw_result$p.value - 0.05),
                   "AGREE", "DISAGREE"),
            ifelse(kw_result$p.value < 0.05, "significant", "NOT significant")))

# --- Tukey HSD post-hoc (only if ANOVA is significant) ---
anova_p <- summary(anova_model)[[1]][["Pr(>F)"]][1]

if (!is.na(anova_p) && anova_p < 0.05) {
  cat("ANOVA is significant → running Tukey HSD post-hoc test:\n")
  tukey_result <- TukeyHSD(anova_model)
  tukey_df <- as.data.frame(tukey_result$fav_genre)
  tukey_df$comparison <- rownames(tukey_df)
  tukey_sig <- tukey_df %>% filter(`p adj` < 0.05) %>% arrange(`p adj`)
  cat("Significant pairwise differences (p.adj < 0.05):\n")
  print(tukey_sig[, c("comparison", "diff", "lwr", "upr", "p adj")])
  cat("\n")
} else {
  cat("ANOVA is NOT significant (p ≥ 0.05) → no post-hoc test needed.\n\n")
}

# --- Visualisation: Box plots by genre ---
genre_order <- df_anova %>%
  group_by(fav_genre) %>%
  summarise(med = median(depression, na.rm = TRUE)) %>%
  arrange(med) %>%
  pull(fav_genre)

p_h2 <- ggplot(df_anova,
               aes(x = factor(fav_genre, levels = genre_order),
                   y = depression,
                   fill = factor(fav_genre, levels = genre_order))) +
  geom_boxplot(outlier.alpha = 0.4, outlier.size = 1) +
  scale_fill_viridis_d(option = "D", guide = "none") +
  labs(
    title    = "Hypothesis 2: Depression Scores by Favourite Genre",
    subtitle = sprintf("One-way ANOVA: F(%d, %d) = %.2f, p = %.4f, η² = %.4f",
                       summary(anova_model)[[1]]$Df[1],
                       summary(anova_model)[[1]]$Df[2],
                       summary(anova_model)[[1]]$`F value`[1],
                       anova_p, eta_sq),
    x = "Favourite Genre", y = "Depression Score (0–10)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x   = element_text(angle = 45, hjust = 1, size = 10),
    plot.title    = element_text(face = "bold"),
    plot.subtitle = element_text(colour = "grey40")
  )

ggsave("plot_04b_h2_depression_by_genre.png", p_h2, width = 10, height = 6)
cat("Plot saved: plot_04b_h2_depression_by_genre.png\n\n")

# ═════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 3 – Auditing XGBoost (Multiple Linear Regression)
# We take the top music-behaviour feature from XGBoost (saved in Stage 2),
# then test whether it survives after controlling for Age.
# Response: depression
# ═════════════════════════════════════════════════════════════════════════════
cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║  HYPOTHESIS 3: Auditing XGBoost (Multiple Linear Regression)║\n")
cat("╚══════════════════════════════════════════════════════════════╝\n\n")

# ── Identify top music-behaviour feature from XGBoost output ─────────────────
# Demographic columns we want to exclude from being chosen as "the music feature"
demo_cols <- c("age", "log_hours", "hours_per_day", "bpm_clean",
               "anxiety", "insomnia", "ocd")

# Load the ranked feature list produced by Stage 2 (if available)
if (file.exists("top_features.rds")) {
  top_features <- readRDS("top_features.rds")
  # Pick the highest-ranked feature that (a) exists in df_clean and
  # (b) is NOT a demographic / outcome variable
  music_feature <- top_features[
    top_features %in% names(df_clean) &
    !top_features %in% demo_cols
  ][1]
} else {
  music_feature <- NULL
}

# Fallback: use rock frequency column if XGBoost list is unavailable
if (is.null(music_feature) || is.na(music_feature)) {
  music_feature <- names(df_clean)[grepl("rock", names(df_clean), ignore.case = TRUE)][1]
  cat("top_features.rds not found – falling back to:", music_feature, "\n")
} else {
  cat("Top music feature selected from XGBoost ranking:", music_feature, "\n")
}

# Convert ordered factor to integer (0–3) if needed
if (is.ordered(df_clean[[music_feature]])) {
  df_clean[[music_feature]] <- as.integer(df_clean[[music_feature]])
}

# Construct formula
formula_mlr <- as.formula(paste("depression ~ log_hours +", music_feature, "+ age"))
cat("Regression formula:", deparse(formula_mlr), "\n\n")

# --- Fit the model ---
final_model <- lm(formula_mlr, data = df_clean)

cat("Multiple Linear Regression Summary:\n")
print(summary(final_model))
cat("\n")

# --- Variance Inflation Factors (multicollinearity check) ---
cat("Variance Inflation Factors (VIF):\n")
print(vif(final_model))
cat("  (VIF > 5 suggests concerning multicollinearity; > 10 is severe)\n\n")

# --- Confidence intervals for coefficients ---
cat("95% Confidence Intervals for Coefficients:\n")
print(confint(final_model))
cat("\n")

# --- Residual Diagnostics ─────────────────────────────────────────────────
# Four standard diagnostic plots
png("plot_04c_h3_regression_diagnostics.png", width = 900, height = 750, res = 100)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))
plot(final_model, which = 1:4)
dev.off()
cat("Plot saved: plot_04c_h3_regression_diagnostics.png\n\n")

# --- Coefficient Plot (Forest Plot style) ------------------------------------
coef_df <- as.data.frame(summary(final_model)$coefficients)
coef_df$term <- rownames(coef_df)
coef_df <- coef_df %>%
  filter(term != "(Intercept)") %>%
  mutate(
    lower = Estimate - 1.96 * `Std. Error`,
    upper = Estimate + 1.96 * `Std. Error`,
    sig   = ifelse(`Pr(>|t|)` < 0.05, "Significant (p < 0.05)", "Not significant")
  )

p_h3_coef <- ggplot(coef_df, aes(x = Estimate, y = term, colour = sig)) +
  geom_point(size = 4) +
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2, linewidth = 1) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = c("Significant (p < 0.05)" = "#e15759",
                                  "Not significant"         = "#4e79a7")) +
  labs(
    title    = "Hypothesis 3: MLR Coefficient Plot",
    subtitle = sprintf("R² = %.3f | Adj. R² = %.3f | n = %d",
                       summary(final_model)$r.squared,
                       summary(final_model)$adj.r.squared,
                       nrow(df_clean)),
    x = "Estimate (95% CI)", y = NULL, colour = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold"),
    plot.subtitle = element_text(colour = "grey40"),
    legend.position = "bottom"
  )

ggsave("plot_04d_h3_coef_plot.png", p_h3_coef, width = 7, height = 5)
cat("Plot saved: plot_04d_h3_coef_plot.png\n\n")

# ─── Summary of Decisions ─────────────────────────────────────────────────────
cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║  DECISION SUMMARY                                           ║\n")
cat("╚══════════════════════════════════════════════════════════════╝\n")

h1_decision <- ifelse(wilcox_result$p.value < 0.05, "REJECT H0", "FAIL TO REJECT H0")
h2_decision <- ifelse(!is.na(anova_p) && anova_p < 0.05, "REJECT H0", "FAIL TO REJECT H0")

mlr_coefs <- summary(final_model)$coefficients
h3_log_hours_p <- mlr_coefs[grep("log_hours",    rownames(mlr_coefs)), "Pr(>|t|)"]
h3_music_p     <- mlr_coefs[grep(music_feature,  rownames(mlr_coefs), fixed = TRUE), "Pr(>|t|)"]
h3_decision <- ifelse(
  (length(h3_log_hours_p) > 0 && h3_log_hours_p < 0.05) |
  (length(h3_music_p)     > 0 && h3_music_p     < 0.05),
  "REJECT H0 (features remain significant after controlling for Age)",
  "FAIL TO REJECT H0 (features are NOT significant after controlling for Age)"
)

cat(sprintf("\n  H1 (Wilcoxon):  p = %.4f → %s\n", wilcox_result$p.value, h1_decision))
cat(sprintf("  H2 (ANOVA):     p = %.4f → %s\n", anova_p, h2_decision))
cat(sprintf("  H3 (MLR):       %s\n", h3_decision))
cat("\n")

cat("═══════════════════════════════════════════════\n")
cat("  Stage 4 complete.  Outputs:\n")
cat("    plot_04a_h1_anxiety_violin.png\n")
cat("    plot_04b_h2_depression_by_genre.png\n")
cat("    plot_04c_h3_regression_diagnostics.png\n")
cat("    plot_04d_h3_coef_plot.png\n")
cat("═══════════════════════════════════════════════\n")
