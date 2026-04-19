# =============================================================================
# STAGE 1: Data Loading & Cleaning
# The Sound of Sadness: ML vs. Statistical Inference in Music-Induced Mental Health
# =============================================================================
# This script loads the raw survey data, cleans messy columns (especially BPM),
# standardizes column names, handles missing values, and saves a clean dataset
# for all downstream stages.
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
if (!requireNamespace("dplyr",   quietly = TRUE)) install.packages("dplyr", repos='https://cran.r-project.org')
if (!requireNamespace("stringr", quietly = TRUE)) install.packages("stringr", repos='https://cran.r-project.org')
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2", repos='https://cran.r-project.org')

library(dplyr)
library(stringr)
library(ggplot2)

# ── Global seed (set here too so Stage 1 is reproducible when run standalone) ─
set.seed(2026)

# ── 1. Load Raw Data ──────────────────────────────────────────────────────────
# Place mxmh_survey_results.csv in the same folder as this script,
# or update the path below.
raw_path <- "mxmh_survey_results.csv"

if (!file.exists(raw_path)) {
  stop("Cannot find '", raw_path, "'.\n",
       "Please put the CSV in the same folder as this script and re-run.")
}

df_raw <- read.csv(raw_path, stringsAsFactors = FALSE, check.names = FALSE)
cat("── Raw data loaded ──\n")
cat("Rows:", nrow(df_raw), " | Columns:", ncol(df_raw), "\n\n")

# ── 2. Standardise Column Names ───────────────────────────────────────────────
# Lowercase, replace spaces with underscores, strip non-alphanumeric characters
# (except underscores) so names are safe to use in R formulas.
names(df_raw) <- tolower(
  gsub("[^[:alnum:]_]", "",
       gsub("\\s+", "_", names(df_raw)))
)

cat("Cleaned column names:\n")
print(names(df_raw))
cat("\n")

# ── 3. Inspect Raw BPM Column ─────────────────────────────────────────────────
# BPM often contains strings ("120 BPM"), ranges ("100-120"), or impossible
# values (e.g., 0, 999).  We extract the first integer and validate range.
cat("── BPM column: sample of raw values ──\n")
print(head(df_raw$bpm, 20))
cat("\nUnique non-numeric BPM values:\n")
non_num <- df_raw$bpm[is.na(suppressWarnings(as.numeric(df_raw$bpm)))]
print(unique(non_num))
cat("\n")

# ── 4. Clean & Impute BPM ────────────────────────────────────────────────────
df_clean <- df_raw %>%
  mutate(
    # Extract the leading integer from whatever text is in BPM
    bpm_clean = as.numeric(str_extract(bpm, "\\d+")),
    # Treat values outside human-music range as invalid → NA
    bpm_clean = ifelse(bpm_clean >= 40 & bpm_clean <= 250, bpm_clean, NA),
    # Impute NAs with the column median (robust to outliers)
    bpm_clean = ifelse(
      is.na(bpm_clean),
      median(bpm_clean, na.rm = TRUE),
      bpm_clean
    )
  )

cat("BPM NAs after extraction (before imputation):",
    sum(is.na(df_raw$bpm)), "\n")
cat("BPM NAs after imputation:", sum(is.na(df_clean$bpm_clean)), "\n\n")

# ── 5. Frequency Columns → Ordered Factor ────────────────────────────────────
# Frequency columns use text levels; convert to ordered factors so they can
# be coerced to integers (0–3) for modelling.
freq_levels <- c("Never", "Rarely", "Sometimes", "Very frequently")

freq_cols <- names(df_clean)[str_detect(names(df_clean), "^frequency")]

df_clean <- df_clean %>%
  mutate(across(
    all_of(freq_cols),
    ~ factor(.x, levels = freq_levels, ordered = TRUE)
  ))

cat("Frequency columns converted to ordered factor (", length(freq_cols), "cols ):\n")
print(freq_cols)
cat("\n")

# ── 6. Boolean / Binary Columns ───────────────────────────────────────────────
# Convert "Yes"/"No" text columns to 1/0 integers
bool_cols <- c("while_working", "instrumentalist", "composer",
               "exploratory", "foreign_languages")

# Only process columns that actually exist in the dataset
bool_cols <- intersect(bool_cols, names(df_clean))

df_clean <- df_clean %>%
  mutate(across(
    all_of(bool_cols),
    ~ as.integer(.x == "Yes")
  ))

cat("Boolean columns recoded to 0/1:", paste(bool_cols, collapse = ", "), "\n\n")

# ── 7. Drop Rows with Missing Target / Key Predictor Variables ────────────────
rows_before <- nrow(df_clean)

df_clean <- df_clean %>%
  filter(
    !is.na(depression),
    !is.na(anxiety),
    !is.na(age),
    !is.na(hours_per_day)
  )

rows_after <- nrow(df_clean)
cat("Rows removed due to NA in key columns:",
    rows_before - rows_after, "\n")
cat("Final clean dataset: ", rows_after, "rows\n\n")

# ── 8. Derived Variables ──────────────────────────────────────────────────────
# Log-transform hours_per_day to correct right-skew (used in regression later)
df_clean <- df_clean %>%
  mutate(log_hours = log(hours_per_day + 1))

cat("Derived variable 'log_hours' = log(hours_per_day + 1) added.\n\n")

# ── 9. Summary of Clean Dataset ───────────────────────────────────────────────
cat("── Summary of key variables ──\n")
df_clean %>%
  select(age, hours_per_day, log_hours, bpm_clean, anxiety, depression,
         insomnia, ocd) %>%
  summary() %>%
  print()

# ── 10. Exploratory Plots ─────────────────────────────────────────────────────
# Distribution of the two main outcome variables
p1 <- ggplot(df_clean, aes(x = depression)) +
  geom_histogram(binwidth = 1, fill = "#4e79a7", colour = "white") +
  labs(title = "Distribution of Depression Scores",
       x = "Depression (0–10)", y = "Count") +
  theme_minimal()

p2 <- ggplot(df_clean, aes(x = anxiety)) +
  geom_histogram(binwidth = 1, fill = "#f28e2b", colour = "white") +
  labs(title = "Distribution of Anxiety Scores",
       x = "Anxiety (0–10)", y = "Count") +
  theme_minimal()

p3 <- ggplot(df_clean, aes(x = hours_per_day)) +
  geom_histogram(binwidth = 1, fill = "#76b7b2", colour = "white") +
  labs(title = "Distribution of Daily Listening Hours",
       x = "Hours per Day", y = "Count") +
  theme_minimal()

p4 <- ggplot(df_clean, aes(x = log_hours)) +
  geom_histogram(bins = 30, fill = "#59a14f", colour = "white") +
  labs(title = "Log-Transformed Daily Listening Hours",
       x = "log(Hours + 1)", y = "Count") +
  theme_minimal()

# Save plots
ggsave("plot_01a_depression_dist.png",    p1, width = 6, height = 4)
ggsave("plot_01b_anxiety_dist.png",       p2, width = 6, height = 4)
ggsave("plot_01c_hours_dist.png",         p3, width = 6, height = 4)
ggsave("plot_01d_log_hours_dist.png",     p4, width = 6, height = 4)
cat("Plots saved: plot_01a–01d\n\n")

# ── 11. Correlation Heatmap (EDA) ────────────────────────────────────────────
# Shows pairwise Pearson correlations between all key numeric variables.
# Useful for spotting multicollinearity before regression and for EDA storytelling.
if (!requireNamespace("reshape2", quietly = TRUE)) install.packages("reshape2", repos='https://cran.r-project.org')
library(reshape2)

numeric_vars <- df_clean %>%
  select(age, hours_per_day, log_hours, bpm_clean,
         anxiety, depression, insomnia, ocd)

cor_matrix <- cor(numeric_vars, use = "pairwise.complete.obs")

# Melt to long format for ggplot
cor_long <- melt(cor_matrix)
names(cor_long) <- c("Var1", "Var2", "Correlation")

p_heatmap <- ggplot(cor_long, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.2f", Correlation)),
            size = 3, colour = "black") +
  scale_fill_gradient2(
    low  = "#d73027", mid = "white", high = "#4575b4",
    midpoint = 0, limits = c(-1, 1),
    name = "Pearson r"
  ) +
  labs(
    title    = "Correlation Heatmap – Key Numeric Variables",
    subtitle = "Anxiety, Depression, Insomnia, OCD tend to cluster positively",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x  = element_text(angle = 45, hjust = 1),
    plot.title   = element_text(face = "bold"),
    panel.grid   = element_blank()
  )

ggsave("plot_01e_correlation_heatmap.png", p_heatmap, width = 7, height = 6)
cat("Plot saved: plot_01e_correlation_heatmap.png\n\n")

# ── 12. Save Clean Data ───────────────────────────────────────────────────────
saveRDS(df_clean, "df_clean.rds")
write.csv(df_clean, "df_clean.csv", row.names = FALSE)

cat("═══════════════════════════════════════════════\n")
cat("  Stage 1 complete.  Clean data saved to:\n")
cat("    df_clean.rds  (for downstream R scripts)\n")
cat("    df_clean.csv  (human-readable)\n")
cat("═══════════════════════════════════════════════\n")
