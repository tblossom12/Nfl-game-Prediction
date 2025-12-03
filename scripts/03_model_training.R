# ==============================================================================
# NFL PREDICTION MODEL - MODEL TRAINING WITH FEATURE SELECTION
# Script: 03_model_training.R
# Purpose: Train models with proper temporal splits and feature experimentation
# ==============================================================================

library(dplyr)
library(readr)
library(here)
library(caret)
library(randomForest)
library(xgboost)
library(glmnet)
library(yardstick)
library(ggplot2)

ROLLING_WINDOW = 7

if (!exists("slice", mode = "function")) {
  slice <- dplyr::slice
}

# Set up paths
data_processed_path <- here("data", "processed")
models_path <- here("models")
if (!dir.exists(models_path)) dir.create(models_path, recursive = TRUE)

set.seed(42)

print("=== LOADING PROCESSED DATA ===")
modeling_data <- read_rds(here(data_processed_path, "modeling_data.rds"))
print(paste("✓ Loaded", nrow(modeling_data), "games"))

# ==============================================================================
# FEATURE SET CONFIGURATION
# ==============================================================================

print("=== CONFIGURING FEATURE SETS ===")

# Define different feature sets for experimentation
FEATURE_SETS <- list(
  
  # Minimal set - just the basics
  minimal = c(
    "epa_advantage",
    "success_rate_advantage"
  ),
  
  # Core set - EPA + situational
  core = c(
    "epa_advantage",
    "success_rate_advantage", 
    "rush_advantage",
    "pass_advantage",
    "red_zone_advantage",
    "third_down_advantage"
  ),
  
  # Extended set - add team strengths
  extended = c(
    "epa_advantage",
    "success_rate_advantage",
    "rush_advantage",
    "pass_advantage",
    "red_zone_advantage",
    "third_down_advantage",
    "home_net_epa",
    "away_net_epa"
  ),
  
  # Full set - everything that exists
  full = c(
    "epa_advantage",
    "success_rate_advantage",
    "rush_advantage",
    "pass_advantage",
    "red_zone_advantage",
    "third_down_advantage",
    "home_net_epa",
    "away_net_epa"
  ),
  
  forward_best = c(
    "divisional_game",
    "away_net_success",
    "home_net_epa",
    "third_down_advantage",
    "pressure_advantage",
    "red_zone_advantage"
  )
)

# Choose which feature set to use
ACTIVE_FEATURE_SET <- "extended"  # Change this to experiment!
features_to_use <- FEATURE_SETS[[ACTIVE_FEATURE_SET]]

print(paste("Using feature set:", ACTIVE_FEATURE_SET))
print(paste("Number of features:", length(features_to_use)))
cat("\nFeatures:\n")
cat(paste("-", features_to_use), sep = "\n")

# ==============================================================================
# TEMPORAL DATA SPLITTING
# ==============================================================================

print("\n=== CREATING TEMPORAL DATA SPLITS ===")

# Split by season (proper temporal split)
train_data <- modeling_data %>% filter(season <= 2022)
val_data <- modeling_data %>% filter(season == 2023)
test_data <- modeling_data %>% filter(season == 2024)

print(paste("Training set (2020-2022):", nrow(train_data), "games"))
print(paste("Validation set (2023):   ", nrow(val_data), "games"))
print(paste("Test set (2024):         ", nrow(test_data), "games"))

# Verify no data leakage
print("\nVerifying temporal integrity...")
print(paste("Latest training game:", max(train_data$season), "Week", max(train_data$week[train_data$season == max(train_data$season)])))
print(paste("Earliest val game:    ", min(val_data$season), "Week", min(val_data$week[val_data$season == min(val_data$season)])))
print(paste("Earliest test game:   ", min(test_data$season), "Week", min(test_data$week[test_data$season == min(test_data$season)])))

# Prepare feature matrices
X_train <- train_data[, features_to_use, drop = FALSE]
y_train <- train_data$home_win
X_val <- val_data[, features_to_use, drop = FALSE]
y_val <- val_data$home_win
X_test <- test_data[, features_to_use, drop = FALSE]
y_test <- test_data$home_win

# ==============================================================================
# MODEL TRAINING
# ==============================================================================

print("\n=== TRAINING MODELS ===")

model_results <- list()
model_objects <- list()

# ---------------------------------------------------------------------------
# 1. LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
print("\n[1/4] Training Logistic Regression...")

# Prepare numeric matrices
numeric_features <- features_to_use[!features_to_use %in% c("divisional_game", "season_phase")]
X_train_numeric <- X_train[, numeric_features, drop = FALSE]
X_val_numeric <- X_val[, numeric_features, drop = FALSE]
X_test_numeric <- X_test[, numeric_features, drop = FALSE]

X_train_matrix <- as.matrix(X_train_numeric)
X_val_matrix <- as.matrix(X_val_numeric)
X_test_matrix <- as.matrix(X_test_numeric)

# Train elastic net
logistic_model <- cv.glmnet(
  x = X_train_matrix,
  y = y_train,
  family = "binomial",
  alpha = 0.5,
  nfolds = 5,
  type.measure = "class"
)

pred_logistic_val <- predict(logistic_model, X_val_matrix, type = "response", s = "lambda.min")[,1]
pred_logistic_test <- predict(logistic_model, X_test_matrix, type = "response", s = "lambda.min")[,1]
pred_logistic_val_class <- factor(ifelse(pred_logistic_val > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))
pred_logistic_test_class <- factor(ifelse(pred_logistic_test > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))

model_objects$logistic <- logistic_model
model_results$logistic <- list(
  val_prob = pred_logistic_val,
  val_class = pred_logistic_val_class,
  test_prob = pred_logistic_test,
  test_class = pred_logistic_test_class
)

print("✓ Logistic Regression complete")

# ---------------------------------------------------------------------------
# 2. RANDOM FOREST
# ---------------------------------------------------------------------------
print("\n[2/4] Training Random Forest...")

rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = max(2, floor(sqrt(length(features_to_use)))),
  importance = TRUE
)

pred_rf_val <- predict(rf_model, X_val, type = "prob")[, "Win"]
pred_rf_val_class <- predict(rf_model, X_val, type = "class")
pred_rf_test <- predict(rf_model, X_test, type = "prob")[, "Win"]
pred_rf_test_class <- predict(rf_model, X_test, type = "class")

model_objects$random_forest <- rf_model
model_results$random_forest <- list(
  val_prob = pred_rf_val,
  val_class = pred_rf_val_class,
  test_prob = pred_rf_test,
  test_class = pred_rf_test_class
)

print("✓ Random Forest complete")

# ---------------------------------------------------------------------------
# 3. XGBOOST
# ---------------------------------------------------------------------------
print("\n[3/4] Training XGBoost...")

# Convert factors to numeric for XGBoost
X_train_xgb <- X_train
X_val_xgb <- X_val
X_test_xgb <- X_test

if ("divisional_game" %in% names(X_train_xgb)) {
  X_train_xgb$divisional_game <- as.numeric(X_train_xgb$divisional_game)
  X_val_xgb$divisional_game <- as.numeric(X_val_xgb$divisional_game)
  X_test_xgb$divisional_game <- as.numeric(X_test_xgb$divisional_game)
}

dtrain <- xgb.DMatrix(data = as.matrix(X_train_xgb), label = as.numeric(y_train) - 1)
dval <- xgb.DMatrix(data = as.matrix(X_val_xgb), label = as.numeric(y_val) - 1)
dtest <- xgb.DMatrix(data = as.matrix(X_test_xgb), label = as.numeric(y_test) - 1)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  seed = 42
)

xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 20,
  verbose = 0
)

pred_xgb_val <- predict(xgb_model, dval)
pred_xgb_val_class <- factor(ifelse(pred_xgb_val > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))
pred_xgb_test <- predict(xgb_model, dtest)
pred_xgb_test_class <- factor(ifelse(pred_xgb_test > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))

model_objects$xgboost <- xgb_model
model_results$xgboost <- list(
  val_prob = pred_xgb_val,
  val_class = pred_xgb_val_class,
  test_prob = pred_xgb_test,
  test_class = pred_xgb_test_class
)

print("✓ XGBoost complete")

# ---------------------------------------------------------------------------
# 4. ENSEMBLE
# ---------------------------------------------------------------------------
print("\n[4/4] Creating Ensemble...")

ensemble_val <- (pred_logistic_val + pred_rf_val + pred_xgb_val) / 3
ensemble_test <- (pred_logistic_test + pred_rf_test + pred_xgb_test) / 3
ensemble_val_class <- factor(ifelse(ensemble_val > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))
ensemble_test_class <- factor(ifelse(ensemble_test > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))

model_results$ensemble <- list(
  val_prob = ensemble_val,
  val_class = ensemble_val_class,
  test_prob = ensemble_test,
  test_class = ensemble_test_class
)

print("✓ Ensemble complete")

# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

print("\n=== EVALUATING MODELS ===")

evaluate_model <- function(actual, predicted_prob, predicted_class, model_name, dataset_type) {
  accuracy <- mean(actual == predicted_class)
  auc <- yardstick::roc_auc_vec(actual, predicted_prob)
  log_loss <- yardstick::mn_log_loss_vec(actual, predicted_prob)
  
  cm <- table(Predicted = predicted_class, Actual = actual)
  precision <- cm[2,2] / sum(cm[2,])
  recall <- cm[2,2] / sum(cm[,2])
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  return(data.frame(
    Model = model_name,
    Dataset = dataset_type,
    Accuracy = round(accuracy, 4),
    AUC = round(auc, 4),
    LogLoss = round(log_loss, 4),
    Precision = round(precision, 4),
    Recall = round(recall, 4),
    F1 = round(f1, 4)
  ))
}

# Evaluate all models
all_results <- bind_rows(
  # Validation set
  evaluate_model(y_val, model_results$logistic$val_prob, model_results$logistic$val_class, "Logistic", "Validation"),
  evaluate_model(y_val, model_results$random_forest$val_prob, model_results$random_forest$val_class, "Random Forest", "Validation"),
  evaluate_model(y_val, model_results$xgboost$val_prob, model_results$xgboost$val_class, "XGBoost", "Validation"),
  evaluate_model(y_val, model_results$ensemble$val_prob, model_results$ensemble$val_class, "Ensemble", "Validation"),
  
  # Test set
  evaluate_model(y_test, model_results$logistic$test_prob, model_results$logistic$test_class, "Logistic", "Test"),
  evaluate_model(y_test, model_results$random_forest$test_prob, model_results$random_forest$test_class, "Random Forest", "Test"),
  evaluate_model(y_test, model_results$xgboost$test_prob, model_results$xgboost$test_class, "XGBoost", "Test"),
  evaluate_model(y_test, model_results$ensemble$test_prob, model_results$ensemble$test_class, "Ensemble", "Test")
)

print("\n=== MODEL PERFORMANCE ===")
print(all_results)

# ==============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ==============================================================================

print("\n=== FEATURE IMPORTANCE ===")

# Random Forest importance
rf_importance <- importance(model_objects$random_forest) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Feature") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  mutate(Model = "Random Forest", Importance = MeanDecreaseGini / sum(MeanDecreaseGini)) %>%
  select(Model, Feature, Importance)

# XGBoost importance
xgb_importance <- xgb.importance(model = model_objects$xgboost) %>%
  mutate(Model = "XGBoost", Importance = Gain / sum(Gain)) %>%
  select(Model, Feature, Importance)

combined_importance <- bind_rows(
  rf_importance,
  xgb_importance
) %>%
  group_by(Feature) %>%
  summarise(Avg_Importance = mean(Importance)) %>%
  arrange(desc(Avg_Importance))

print("\nFeature Importance Rankings:")
print(combined_importance)

# ==============================================================================
# FEATURE SET COMPARISON
# ==============================================================================



# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n=== SAVING RESULTS ===")

# Save models
saveRDS(model_objects, here(models_path, "trained_models.rds"))

# Save performance metrics
write_csv(all_results, here(models_path, "model_performance.csv"))

# Save feature importance
write_csv(combined_importance, here(models_path, "feature_importance.csv"))

# Save feature set comparison
#write_csv(feature_set_comparison, here(models_path, "feature_set_comparison.csv"))

# Save predictions
test_predictions <- data.frame(
  game_id = test_data$game_id,
  season = test_data$season,
  week = test_data$week,
  home_team = test_data$home_team,
  away_team = test_data$away_team,
  actual_result = as.character(y_test),
  logistic_prob = model_results$logistic$test_prob,
  rf_prob = model_results$random_forest$test_prob,
  xgb_prob = model_results$xgboost$test_prob,
  ensemble_prob = model_results$ensemble$test_prob,
  ensemble_prediction = as.character(model_results$ensemble$test_class),
  correct = as.character(y_test) == as.character(model_results$ensemble$test_class)
)

write_csv(test_predictions, here(models_path, "test_predictions.csv"))

# Save configuration
config_summary <- data.frame(
  Setting = c("Feature Set", "Rolling Window", "Train Seasons", "Val Season", "Test Season", "Train Games", "Val Games", "Test Games"),
  Value = c(ACTIVE_FEATURE_SET, ROLLING_WINDOW, "2020-2022", "2023", "2024", nrow(train_data), nrow(val_data), nrow(test_data))
)

write_csv(config_summary, here(models_path, "training_configuration.csv"))

print("✓ All results saved")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n=== TRAINING SUMMARY ===")

best_test_model <- all_results %>% 
  filter(Dataset == "Test") %>% 
  arrange(desc(Accuracy)) %>% 
  head(1)

cat("\nBest Model:", best_test_model$Model, "\n")
cat("Test Accuracy:", best_test_model$Accuracy, "\n")
cat("Test AUC:", best_test_model$AUC, "\n")

cat("\nTop 3 Features:\n")
print(head(combined_importance, 3))

#cat("\nBest Feature Set:", feature_set_comparison$Feature_Set[which.max(feature_set_comparison$Test_Accuracy)], "\n")
#cat("Accuracy:", max(feature_set_comparison$Test_Accuracy), "\n")

print("\n=== READY FOR PREDICTIONS ===")
print("Next: Run 04_predictions.R")