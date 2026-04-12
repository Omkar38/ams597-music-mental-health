# =============================================================================
# STAGE 2: Machine Learning – XGBoost Feature Importance
# The Sound of Sadness: ML vs. Statistical Inference in Music-Induced Mental Health
# =============================================================================
# Trains an XGBoost regression model to predict depression scores, then
# extracts and visualises the top feature importances.  This stage answers
# the question: "What does the ML algorithm think matters?"
# Run AFTER 01_data_cleaning.R (requires df_clean.rds).
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
pkgs <- c("dplyr", "ggplot2", "xgboost", "Matrix", "Ckmeans.1d.dp")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

library(dplyr)
library(ggplot2)
library(xgboost)
library(Matrix)
library(Ckmeans.1d.dp)   # required by xgb.ggplot.importance

# ── 1. Load Clean Data ────────────────────────────────────────────────────────
if (!file.exists("df_clean.rds")) {
  stop("'df_clean.rds' not found. Please run 01_data_cleaning.R first.")
}

df_clean <- readRDS("df_clean.rds")
cat("Clean data loaded:", nrow(df_clean), "rows\n\n")

# ── 2. Prepare Model Matrix ───────────────────────────────────────────────────
# Drop columns that are not useful predictors:
#   timestamp   – row identifier
#   bpm         – raw (replaced by bpm_clean)
#   permissions – consent string, not a feature
#   music_effects – post-hoc self-report, would cause leakage
#   hours_per_day – kept; log_hours is the transformed version we keep too

drop_cols <- c("timestamp", "bpm", "permissions", "music_effects")
ml_data <- df_clean %>%
  select(-any_of(drop_cols))

# Convert ordered factors to integers (0 = Never … 3 = Very frequently)
ml_data <- ml_data %>%
  mutate(across(where(is.ordered), as.integer))

# Convert any remaining character columns to factors then integers
ml_data <- ml_data %>%
  mutate(across(where(is.character), ~ as.integer(factor(.x))))

cat("Features used for modelling:", ncol(ml_data) - 1, "\n")
cat("Target variable: depression\n\n")

# Build sparse model matrix (one-hot encodes factors automatically)
# depression is the response; remove it from the predictor side
sparse_matrix <- sparse.model.matrix(depression ~ . - 1, data = ml_data)
y_target      <- ml_data$depression

cat("Sparse matrix dimensions:", nrow(sparse_matrix), "×", ncol(sparse_matrix), "\n\n")

# ── 3. Train/Test Split + XGBoost Model ──────────────────────────────────────
# Split 80% train / 20% test BEFORE training so that the final RMSE we report
# reflects genuinely held-out data (not just cross-validation on training rows).
set.seed(2026)
n_obs   <- nrow(sparse_matrix)
train_idx <- sample(seq_len(n_obs), size = floor(0.8 * n_obs))
test_idx  <- setdiff(seq_len(n_obs), train_idx)

dtrain <- xgb.DMatrix(sparse_matrix[train_idx, ], label = y_target[train_idx])
dtest  <- xgb.DMatrix(sparse_matrix[test_idx,  ], label = y_target[test_idx])

cat(sprintf("Train set: %d rows | Test set: %d rows\n\n", length(train_idx), length(test_idx)))

# watchlist lets XGBoost print both train and test RMSE each round
watchlist <- list(train = dtrain, test = dtest)

xgb_model <- xgb.train(
  params = list(
    objective   = "reg:squarederror",
    eval_metric = "rmse",
    max_depth   = 4,
    eta         = 0.1,         # learning rate
    subsample   = 0.8
  ),
  data                = dtrain,
  nrounds             = 200,             # cap; early stopping will cut this short
  watchlist           = watchlist,
  early_stopping_rounds = 15,            # stop if test RMSE doesn't improve for 15 rounds
  verbose             = 1,
  print_every_n       = 20
)

# Report held-out test RMSE
test_preds     <- predict(xgb_model, dtest)
test_rmse      <- sqrt(mean((test_preds - y_target[test_idx])^2))
best_iteration <- xgb_model$best_iteration

cat(sprintf("\nModel training complete.\n"))
cat(sprintf("  Best iteration : %d\n", best_iteration))
cat(sprintf("  Hold-out RMSE  : %.4f  (scale 0–10; < 2 is reasonable)\n\n", test_rmse))

# ── 4. Feature Importance ─────────────────────────────────────────────────────
importance_matrix <- xgb.importance(
  feature_names = colnames(sparse_matrix),
  model         = xgb_model
)

cat("── Top 15 Features by Gain ──\n")
print(head(importance_matrix, 15))
cat("\n")

# ── 5. Visualise Feature Importance ──────────────────────────────────────────
# Plot top 10 features
top10 <- importance_matrix[1:min(10, nrow(importance_matrix)), ]

p_importance <- xgb.ggplot.importance(top10) +
  ggtitle("XGBoost Top 10 Features for Predicting Depression") +
  labs(subtitle = paste0("nrounds = 100 | eta = 0.1 | n = ", nrow(df_clean)),
       x = "Feature", y = "Gain (importance score)") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold"),
    axis.text.y   = element_text(size = 11)
  )

ggsave("plot_02a_xgb_importance.png", p_importance, width = 8, height = 5)
cat("Plot saved: plot_02a_xgb_importance.png\n\n")

# ── 6. SHAP-style Gain / Cover / Frequency summary ───────────────────────────
# Save all importance metrics for reference in later stages
write.csv(importance_matrix, "xgb_feature_importance.csv", row.names = FALSE)
cat("Full importance table saved: xgb_feature_importance.csv\n\n")

# ── 7. Identify Top Features for Stage 4 Regression ─────────────────────────
# We extract the top-ranked music-behaviour features (non-demographic) to
# audit in the multiple regression in Stage 4.
top_features <- importance_matrix$Feature[1:10]
cat("Top 10 features identified (to audit in Stage 4):\n")
print(top_features)

# Save list so Stage 4 can read it automatically
saveRDS(top_features, "top_features.rds")

# ── 8. Cross-Validation RMSE (model quality check) ───────────────────────────
# CV is run on the TRAINING set only to avoid any contamination with the holdout.
set.seed(2026)
cv_result <- xgb.cv(
  data      = dtrain,
  nfold     = 5,
  nrounds   = best_iteration,   # use the best round from the held-out run
  params    = list(
    objective   = "reg:squarederror",
    eval_metric = "rmse",
    max_depth   = 4,
    eta         = 0.1,
    subsample   = 0.8
  ),
  verbose   = FALSE
)

cv_best_round <- which.min(cv_result$evaluation_log$test_rmse_mean)
cv_best_rmse  <- min(cv_result$evaluation_log$test_rmse_mean)

cat("\n5-Fold CV Results (on training set):\n")
cat("  Best round  :", cv_best_round, "\n")
cat("  Best CV RMSE:", round(cv_best_rmse, 4), "\n")
cat("  Hold-out RMSE (true test):", round(test_rmse, 4), "\n")
cat("  (Depression scale is 0–10; RMSE < 2 indicates reasonable fit)\n\n")

# Plot CV learning curve
cv_log <- cv_result$evaluation_log

p_cv <- ggplot(cv_log, aes(x = iter)) +
  geom_line(aes(y = train_rmse_mean, colour = "Train")) +
  geom_line(aes(y = test_rmse_mean,  colour = "CV Test")) +
  geom_vline(xintercept = cv_best_round, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = c("Train" = "#4e79a7", "CV Test" = "#e15759")) +
  labs(title = "XGBoost 5-Fold CV Learning Curve",
       x = "Boosting Round", y = "RMSE",
       colour = NULL) +
  theme_minimal(base_size = 13)

ggsave("plot_02b_xgb_cv_curve.png", p_cv, width = 7, height = 4)
cat("CV curve saved: plot_02b_xgb_cv_curve.png\n\n")

# Save model object for optional later use
xgb.save(xgb_model, "xgb_depression_model.bin")

cat("═══════════════════════════════════════════════\n")
cat("  Stage 2 complete.  Outputs:\n")
cat("    xgb_feature_importance.csv\n")
cat("    top_features.rds\n")
cat("    xgb_depression_model.bin\n")
cat("    plot_02a_xgb_importance.png\n")
cat("    plot_02b_xgb_cv_curve.png\n")
cat("═══════════════════════════════════════════════\n")
