# =============================================================================
# STAGE 3: Checking Statistical Assumptions (Normality)
# The Sound of Sadness: ML vs. Statistical Inference in Music-Induced Mental Health
# =============================================================================
# Before running any parametric tests in Stage 4, we must verify whether the
# key continuous variables meet the normality assumption.  This script runs
# Shapiro-Wilk tests, draws Q-Q plots, and checks skewness / kurtosis.
# For variables that violate normality, we confirm whether log-transformation
# (already computed in Stage 1) improves the distribution sufficiently.
# Run AFTER 01_data_cleaning.R (requires df_clean.rds).
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
pkgs <- c("dplyr", "ggplot2", "moments")   # moments: skewness / kurtosis
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos='https://cran.r-project.org')
}

library(dplyr)
library(ggplot2)
library(moments)

set.seed(2026)   # reproducibility for any random sampling (e.g., Shapiro-Wilk subsampling)

# ── 1. Load Clean Data ────────────────────────────────────────────────────────
if (!file.exists("df_clean.rds")) {
  stop("'df_clean.rds' not found. Please run 01_data_cleaning.R first.")
}
df_clean <- readRDS("df_clean.rds")
cat("Clean data loaded:", nrow(df_clean), "rows\n\n")

# ── Helper: Normality Report ──────────────────────────────────────────────────
normality_report <- function(x, label) {
  # Shapiro-Wilk is only valid for n ≤ 5000; subsample if needed
  n    <- length(na.omit(x))
  sw_x <- if (n > 5000) sample(na.omit(x), 5000) else na.omit(x)

  sw   <- shapiro.test(sw_x)
  skew <- skewness(na.omit(x))
  kurt <- kurtosis(na.omit(x))

  cat(sprintf("  %-30s  W = %.4f  p = %.4e  skew = %+.2f  kurt = %.2f  n = %d\n",
              label, sw$statistic, sw$p.value, skew, kurt, n))

  invisible(list(W = sw$statistic, p = sw$p.value, skew = skew, kurt = kurt, n = n))
}

# ── 2. Shapiro-Wilk & Descriptive Stats for Key Variables ────────────────────
cat("── Normality Check (Shapiro-Wilk) ──────────────────────────────────────\n")
cat("  (p < 0.05 → reject normality)\n\n")

vars_to_check <- list(
  hours_raw  = df_clean$hours_per_day,
  log_hours  = df_clean$log_hours,
  depression = df_clean$depression,
  anxiety    = df_clean$anxiety,
  insomnia   = df_clean$insomnia,
  ocd        = df_clean$ocd,
  age        = df_clean$age,
  bpm        = df_clean$bpm_clean
)

results <- lapply(names(vars_to_check), function(nm) {
  normality_report(vars_to_check[[nm]], nm)
})
names(results) <- names(vars_to_check)

cat("\n")

# ── 3. Q-Q Plots for All Variables ────────────────────────────────────────────
qq_plot <- function(x, label) {
  df_qq <- data.frame(
    sample = sort(na.omit(x)),
    theoretical = qnorm(ppoints(sum(!is.na(x))))
  )
  ggplot(df_qq, aes(x = theoretical, y = sample)) +
    geom_point(alpha = 0.4, colour = "#4e79a7", size = 1.2) +
    geom_abline(slope  = sd(na.omit(x)),
                intercept = mean(na.omit(x)),
                colour = "red", linewidth = 1) +
    labs(title = paste("Q-Q Plot:", label),
         x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal(base_size = 12)
}

plots_qq <- list(
  qq_plot(df_clean$hours_per_day, "Hours per Day (raw)"),
  qq_plot(df_clean$log_hours,     "Hours per Day (log-transformed)"),
  qq_plot(df_clean$depression,    "Depression Score"),
  qq_plot(df_clean$anxiety,       "Anxiety Score"),
  qq_plot(df_clean$age,           "Age"),
  qq_plot(df_clean$bpm_clean,     "BPM (cleaned)")
)

labels_qq <- c("hours_raw", "log_hours", "depression", "anxiety", "age", "bpm")

for (i in seq_along(plots_qq)) {
  fname <- paste0("plot_03a_qq_", labels_qq[i], ".png")
  ggsave(fname, plots_qq[[i]], width = 5, height = 4)
  cat("Saved:", fname, "\n")
}
cat("\n")

# ── 4. Density Plots: Raw vs Log-Transformed Hours ────────────────────────────
p_density_compare <- ggplot(df_clean) +
  geom_density(aes(x = hours_per_day, fill = "Raw hours"),
               alpha = 0.5, colour = NA) +
  geom_density(aes(x = log_hours, fill = "log(hours + 1)"),
               alpha = 0.5, colour = NA) +
  scale_fill_manual(values = c("Raw hours" = "#e15759",
                                "log(hours + 1)" = "#59a14f")) +
  labs(title = "Hours per Day: Raw vs. Log-Transformed",
       x = "Value", y = "Density", fill = NULL) +
  theme_minimal(base_size = 13)

ggsave("plot_03b_hours_density_compare.png", p_density_compare, width = 7, height = 4)
cat("Saved: plot_03b_hours_density_compare.png\n\n")

# ── 5. Histograms for Depression & Anxiety ────────────────────────────────────
p_dep_hist <- ggplot(df_clean, aes(x = depression)) +
  geom_histogram(binwidth = 1, fill = "#4e79a7", colour = "white") +
  stat_function(
    fun  = function(x) dnorm(x, mean(df_clean$depression), sd(df_clean$depression)) *
                       nrow(df_clean),
    colour = "red", linewidth = 1, linetype = "dashed"
  ) +
  labs(title = "Depression: Observed vs. Normal Curve",
       x = "Depression Score (0–10)", y = "Count") +
  theme_minimal(base_size = 13)

p_anx_hist <- ggplot(df_clean, aes(x = anxiety)) +
  geom_histogram(binwidth = 1, fill = "#f28e2b", colour = "white") +
  stat_function(
    fun  = function(x) dnorm(x, mean(df_clean$anxiety), sd(df_clean$anxiety)) *
                       nrow(df_clean),
    colour = "red", linewidth = 1, linetype = "dashed"
  ) +
  labs(title = "Anxiety: Observed vs. Normal Curve",
       x = "Anxiety Score (0–10)", y = "Count") +
  theme_minimal(base_size = 13)

ggsave("plot_03c_depression_hist_normal.png", p_dep_hist, width = 6, height = 4)
ggsave("plot_03d_anxiety_hist_normal.png",    p_anx_hist, width = 6, height = 4)
cat("Saved: plot_03c_depression_hist_normal.png\n")
cat("Saved: plot_03d_anxiety_hist_normal.png\n\n")

# ── 6. Summary & Interpretation ───────────────────────────────────────────────
cat("── Interpretation Guide ────────────────────────────────────────────────\n")
cat("
Based on the Shapiro-Wilk results above:

 hours_per_day  → Typically RIGHT-SKEWED (p < 0.05 → not normal).
                   Use log_hours in parametric tests (Stage 4, Hypothesis 3).

 depression     → Likely non-normal (discrete 0–10 integers, often left-skewed).
                   Large n partially justifies t-tests (CLT), but ANOVA residuals
                   should be checked post-hoc (see Stage 4).

 anxiety        → Similar to depression; bounded integers, rarely perfectly normal.
                   Wilcoxon rank-sum test is used in Hypothesis 1 (non-parametric).

 age            → Often approximately normal in survey data; check the Q-Q plot.

 bpm_clean      → Should be roughly bell-shaped after cleaning; verify Q-Q plot.

 Recommendation:
   H1 (Instrumentalist → Anxiety):  Use Wilcoxon (non-parametric) ✓ already planned
   H2 (Genre → Depression):         ANOVA + check residual normality (Levene test)
   H3 (MLR audit):                  Use log_hours; check residual diagnostics
\n")

cat("═══════════════════════════════════════════════\n")
cat("  Stage 3 complete.  Outputs:\n")
cat("    plot_03a_qq_*.png  (Q-Q plots for each variable)\n")
cat("    plot_03b_hours_density_compare.png\n")
cat("    plot_03c_depression_hist_normal.png\n")
cat("    plot_03d_anxiety_hist_normal.png\n")
cat("═══════════════════════════════════════════════\n")
