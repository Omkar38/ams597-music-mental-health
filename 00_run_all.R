# =============================================================================
# 00_run_all.R  —  Master Runner
# The Sound of Sadness: ML vs. Statistical Inference in Music-Induced Mental Health
# =============================================================================
# Sources all five pipeline stages in order from this file's directory.
# Run this single script to reproduce every output in the project.
#
# Usage (RStudio):
#   1. Open this file in RStudio
#   2. Session > Set Working Directory > To Source File Location
#   3. Click "Source" (or Ctrl+Shift+S)
#
# Usage (terminal):
#   Rscript --vanilla 00_run_all.R
# =============================================================================

# ── Set working directory to this script's own location ──────────────────────
# Works when sourced from RStudio *or* run via Rscript from any directory.
this_dir <- tryCatch(
  dirname(rstudioapi::getSourceEditorContext()$path),  # RStudio path
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- args[grepl("--file=", args)]
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg)))
    else getwd()   # fallback: current directory
  }
)
setwd(this_dir)
cat("Working directory set to:", getwd(), "\n\n")

# ── Global seed (ensures full reproducibility across all stages) ──────────────
set.seed(2026)

# ── Stage timing helper ───────────────────────────────────────────────────────
run_stage <- function(script, label) {
  cat(rep("═", 60), "\n", sep = "")
  cat("  Running", label, "\n")
  cat(rep("═", 60), "\n", sep = "")
  t_start <- proc.time()["elapsed"]
  source(script, local = FALSE)
  elapsed <- round(proc.time()["elapsed"] - t_start, 1)
  cat(sprintf("\n  ✓ %s finished in %.1f s\n\n", label, elapsed))
}

# ── Run pipeline ──────────────────────────────────────────────────────────────
run_stage("01_data_cleaning.R",              "Stage 1: Data Cleaning & EDA")
run_stage("02_xgboost_feature_importance.R", "Stage 2: XGBoost Feature Importance")
run_stage("03_assumption_checking.R",        "Stage 3: Normality Assumption Checks")
run_stage("04_hypothesis_tests.R",           "Stage 4: Statistical Hypothesis Tests")
run_stage("05_conclusion.R",                 "Stage 5: Conclusion & Summary")

cat(rep("═", 60), "\n", sep = "")
cat("  ALL STAGES COMPLETE\n")
cat("  Check the working directory for all output plots and tables.\n")
cat(rep("═", 60), "\n", sep = "")
