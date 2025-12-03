# ==============================================================================
# NFL PREDICTION PIPELINE - MASTER RUNNER
# ==============================================================================

library(here)

cat("\n=== NFL PREDICTION PIPELINE ===\n\n")

# Choose what to run (comment out what you don't want)
RUN_DATA_COLLECTION <- FALSE
RUN_FEATURE_ENGINEERING <- TRUE
RUN_UPDATE_CURRENT <- TRUE
RUN_MODEL_TRAINING <- TRUE
RUN_PREDICTIONS <- TRUE

# ==============================================================================
# RUN PIPELINE
# ==============================================================================

if (RUN_DATA_COLLECTION) {
  cat("Step 1: Collecting data...\n")
  source(here("scripts", "01_data_collection.R"))
  cat("\n✓ Data collection complete\n\n")
}

if (RUN_FEATURE_ENGINEERING) {
  cat("Step 2: Engineering features...\n")
  source(here("scripts", "02_feature_engineering.R"))
  cat("\n✓ Feature engineering complete\n\n")
}

if (RUN_UPDATE_CURRENT) {
  cat("Step 3: Updating current season...\n")
  source(here("scripts", "update_current_season.R"))
  cat("\n✓ Current season updated\n\n")
}

if (RUN_MODEL_TRAINING) {
  cat("Step 4: Training models...\n")
  source(here("scripts", "03_model_training.R"))
  cat("\n✓ Model training complete\n\n")
}

if (RUN_PREDICTIONS) {
  cat("Step 5: Loading prediction system...\n")
  source(here("scripts", "04_predictions.R"))
  cat("\n✓ Prediction system ready\n\n")
}

cat("=== PIPELINE COMPLETE ===\n")