# ==============================================================================
# NFL PREDICTION MODEL - MAKE PREDICTIONS (FIXED)
# Script: 04_predictions.R
# Purpose: Use trained models to predict upcoming games
# ==============================================================================

library(dplyr)
library(readr)
library(here)
library(nflreadr)
library(xgboost)
library(glmnet)
library(randomForest)

# Set up paths
data_processed_path <- here("data", "processed")
data_raw_path <- here("data", "raw")
models_path <- here("models")
output_path <- here("output", "predictions")
if (!dir.exists(output_path)) dir.create(output_path, recursive = TRUE)

CURRENT_SEASON <- 2025

print("=== LOADING MODELS AND DATA ===")

# Load trained models and data
trained_models <- readRDS(here(models_path, "trained_models.rds"))
team_stats_by_game <- read_rds(here(data_processed_path, "team_stats_by_game.rds"))
team_divisions <- read_rds(here(data_raw_path, "team_divisions.rds"))

# Load the features that were used in training
modeling_data <- read_rds(here(data_processed_path, "modeling_data.rds"))
FEATURES_USED <- c("epa_advantage",
                   "success_rate_advantage",
                   "rush_advantage",
                   "pass_advantage",
                   "red_zone_advantage",
                   "third_down_advantage",
                   "home_net_epa",
                   "away_net_epa")

print("✓ Models and data loaded")
print(paste("Using", length(FEATURES_USED), "features"))

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Function to get latest stats for a team
get_latest_team_stats <- function(team, season, current_week, stats_data) {
  
  # Try to get stats from current season before the current week
  stats <- stats_data %>%
    filter(posteam == team, season == !!season, week < current_week) %>%
    arrange(desc(week)) %>%
    slice(1)
  
  # If no current season data, use last season
  if (nrow(stats) == 0) {
    stats <- stats_data %>%
      filter(posteam == team, season == !!season - 1) %>%
      arrange(desc(week)) %>%
      slice(1)
    
    if (nrow(stats) > 0) {
      cat("  Note: Using", season - 1, "data for", team, "\n")
    }
  }
  
  return(stats)
}

# Function to check if game is divisional
is_divisional_game <- function(home_team, away_team) {
  home_div <- team_divisions$division[team_divisions$team_abbr == home_team]
  away_div <- team_divisions$division[team_divisions$team_abbr == away_team]
  
  if (length(home_div) == 0 || length(away_div) == 0) return(0)
  return(as.numeric(home_div == away_div))
}

# ==============================================================================
# CREATE GAME FEATURES FUNCTION
# ==============================================================================

load_pbp()

create_game_features <- function(home_team, away_team, week, season = CURRENT_SEASON) {
  
  # Get team stats
  home_stats <- get_latest_team_stats(home_team, season, week, team_stats_by_game)
  away_stats <- get_latest_team_stats(away_team, season, week, team_stats_by_game)
  
  if (nrow(home_stats) == 0 || nrow(away_stats) == 0) {
    stop(paste("No stats available for", home_team, "or", away_team))
  }
  
  # Helper function to safely get a value from a tibble
  safe_get <- function(df, col) {
    if (col %in% names(df)) {
      val <- df[[col]][1]  # Get first element for tibbles
      if (is.na(val) || is.infinite(val)) return(0)
      return(as.numeric(val))
    }
    return(0)
  }
  
  # Calculate matchup features (only the 8 we trained on)
  features <- data.frame(
    # EPA advantages
    epa_advantage = (safe_get(home_stats, "off_epa_per_play_rolling") - 
                       safe_get(away_stats, "def_epa_per_play_rolling")) -
      (safe_get(away_stats, "off_epa_per_play_rolling") - 
         safe_get(home_stats, "def_epa_per_play_rolling")),
    
    success_rate_advantage = (safe_get(home_stats, "off_success_rate_rolling") - 
                                safe_get(away_stats, "def_success_rate_rolling")) -
      (safe_get(away_stats, "off_success_rate_rolling") - 
         safe_get(home_stats, "def_success_rate_rolling")),
    
    # Specific matchups
    rush_advantage = (safe_get(home_stats, "off_rush_epa_per_play_rolling") - 
                        safe_get(away_stats, "def_rush_epa_per_play_rolling")) -
      (safe_get(away_stats, "off_rush_epa_per_play_rolling") - 
         safe_get(home_stats, "def_rush_epa_per_play_rolling")),
    
    pass_advantage = (safe_get(home_stats, "off_pass_epa_per_play_rolling") - 
                        safe_get(away_stats, "def_pass_epa_per_play_rolling")) -
      (safe_get(away_stats, "off_pass_epa_per_play_rolling") - 
         safe_get(home_stats, "def_pass_epa_per_play_rolling")),
    
    # Situational
    red_zone_advantage = (safe_get(home_stats, "off_red_zone_td_rate_rolling") - 
                            safe_get(away_stats, "def_red_zone_td_rate_rolling")) -
      (safe_get(away_stats, "off_red_zone_td_rate_rolling") - 
         safe_get(home_stats, "def_red_zone_td_rate_rolling")),
    
    third_down_advantage = (safe_get(home_stats, "off_third_down_conv_rate_rolling") - 
                              safe_get(away_stats, "def_third_down_conv_rate_rolling")) -
      (safe_get(away_stats, "off_third_down_conv_rate_rolling") - 
         safe_get(home_stats, "def_third_down_conv_rate_rolling")),
    
    # Team strengths (absolute values)
    home_net_epa = safe_get(home_stats, "net_epa_per_play_rolling"),
    away_net_epa = safe_get(away_stats, "net_epa_per_play_rolling")
  )
  
  return(features)
}

# ==============================================================================
# PREDICT GAME FUNCTION
# ==============================================================================

predict_game <- function(home_team, away_team, week = 1, season = CURRENT_SEASON, verbose = TRUE) {
  
  if (verbose) cat("Predicting:", away_team, "@", home_team, "Week", week, "\n")
  
  tryCatch({
    # Create features
    game_features <- create_game_features(home_team, away_team, week, season)
    
    # Make sure we have exactly the right features
    game_features <- game_features[, FEATURES_USED, drop = FALSE]
    
    # Convert to matrix for models
    X_matrix <- as.matrix(game_features)
    
    # Logistic regression prediction
    print("log:")
    logistic_prob <- as.numeric(predict(trained_models$logistic, newx = X_matrix, s = "lambda.min", type = "response"))    # Random Forest prediction
    print("rf:")
    rf_prob <- predict(trained_models$random_forest, game_features, type = "prob")[1, "Win"]
    
    # XGBoost prediction
    print("xgb:")
    xgb_matrix <- xgb.DMatrix(data = X_matrix)
    xgb_prob <- predict(trained_models$xgboost, xgb_matrix)
    
    # Ensemble prediction (average of all three)
    ensemble_prob <- (logistic_prob + rf_prob + xgb_prob) / 3
    
    # Determine winner
    predicted_winner <- ifelse(ensemble_prob > 0.5, home_team, away_team)
    confidence <- ifelse(ensemble_prob > 0.5, ensemble_prob, 1 - ensemble_prob) * 100
    
    # Create result
    result <- data.frame(
      home_team = home_team,
      away_team = away_team,
      week = week,
      season = season,
      logistic_prob = round(logistic_prob, 3),
      rf_prob = round(rf_prob, 3),
      xgb_prob = round(xgb_prob, 3),
      ensemble_prob = round(ensemble_prob, 3),
      predicted_winner = predicted_winner,
      confidence = round(confidence, 1)
    )
    
    if (verbose) {
      cat("  → Predicted winner:", predicted_winner, "with", round(confidence, 1), "% confidence\n")
    }
    
    return(result)
    
  }, error = function(e) {
    cat("  ✗ Error:", e$message, "\n")
    return(NULL)
  })
}

# ==============================================================================
# PREDICT WEEK FUNCTION
# ==============================================================================

predict_week <- function(week_num, season = CURRENT_SEASON) {
  
  cat("\n=== Predicting Week", week_num, "of", season, "season ===\n\n")
  
  # Load schedule
  schedule <- tryCatch({
    load_schedules(seasons = season)
  }, error = function(e) {
    cat("Error loading schedule:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(schedule)) {
    cat("Could not load schedule for", season, "\n")
    return(data.frame())
  }
  
  # Filter to the specific week
  week_games <- schedule %>%
    filter(week == week_num) %>%
    filter(!is.na(home_team), !is.na(away_team))
  
  if (nrow(week_games) == 0) {
    cat("No games found for Week", week_num, "\n")
    return(data.frame())
  }
  
  cat("Found", nrow(week_games), "games\n\n")
  
  # Make predictions
  predictions <- list()
  
  for (i in 1:nrow(week_games)) {
    game <- week_games[i, ]
    
    pred <- predict_game(game$home_team, game$away_team, week_num, season, verbose = TRUE)
    
    if (!is.null(pred)) {
      pred$game_id <- game$game_id
      pred$gameday <- game$gameday
      predictions[[i]] <- pred
    }
    
    cat("\n")
  }
  
  result <- bind_rows(predictions)
  
  if (nrow(result) > 0) {
    cat("✓ Predicted", nrow(result), "games successfully\n")
    result <- result %>% arrange(desc(confidence))
  }
  
  return(result)
}

# ==============================================================================
# GET TEAM RANKINGS
# ==============================================================================

get_team_rankings <- function(season = CURRENT_SEASON) {
  
  # Get most recent stats for each team
  current_stats <- team_stats_by_game %>%
    filter(season == !!season | season == !!season - 1) %>%
    group_by(posteam) %>%
    arrange(desc(season), desc(week)) %>%
    slice(1) %>%
    ungroup()
  
  if (nrow(current_stats) == 0) {
    cat("No team stats available\n")
    return(data.frame())
  }
  
  # Calculate ratings
  rankings <- current_stats %>%
    mutate(
      overall_rating = net_epa_per_play_rolling * 100,
      offensive_rating = off_epa_per_play_rolling * 100,
      defensive_rating = -def_epa_per_play_rolling * 100
    ) %>%
    arrange(desc(overall_rating)) %>%
    mutate(rank = row_number()) %>%
    select(rank, posteam, overall_rating, offensive_rating, defensive_rating,
           off_epa_per_play_rolling, def_epa_per_play_rolling, net_epa_per_play_rolling)
  
  return(rankings)
}

# ==============================================================================
# MANUAL PREDICTION FUNCTION (USER-FRIENDLY)
# ==============================================================================

manual_predict <- function(home_team, away_team, week = 1) {
  
  cat("\n")
  cat("=" %R% rep("=", 60), "\n")
  cat("               NFL GAME PREDICTION\n")
  cat("=" %R% rep("=", 60), "\n\n")
  
  result <- predict_game(home_team, away_team, week, verbose = FALSE)
  
  if (!is.null(result)) {
    cat(sprintf("Matchup:  %s @ %s\n", result$away_team, result$home_team))
    cat(sprintf("Week:     %d\n", result$week))
    cat(sprintf("Season:   %d\n\n", result$season))
    
    cat("MODEL PROBABILITIES (Home Team Win):\n")
    cat(sprintf("  Logistic Regression: %.1f%%\n", result$logistic_prob * 100))
    cat(sprintf("  Random Forest:       %.1f%%\n", result$rf_prob * 100))
    cat(sprintf("  XGBoost:             %.1f%%\n", result$xgb_prob * 100))
    cat(sprintf("  ENSEMBLE:            %.1f%%\n\n", result$ensemble_prob * 100))
    
    cat(sprintf("PREDICTION: %s wins with %.1f%% confidence\n\n", 
                result$predicted_winner, result$confidence))
    
    cat("=" %R% rep("=", 60), "\n\n")
  }
  
  invisible(result)
}

# ==============================================================================
# INITIALIZATION & EXAMPLES
# ==============================================================================

print("\n=== PREDICTION SYSTEM READY ===")

# Check available teams
available_teams <- unique(team_stats_by_game$posteam)
print(paste("Teams with stats available:", length(available_teams)))

# Show team rankings
rankings <- get_team_rankings()
if (nrow(rankings) > 0) {
  cat("\nTop 5 Teams (by EPA):\n")
  top_5 <- head(rankings, 5)
  for (i in 1:nrow(top_5)) {
    cat(sprintf("  %d. %s (%.2f)\n", 
                top_5$rank[i], top_5$posteam[i], top_5$overall_rating[i]))
  }
}

cat("\n=== USAGE EXAMPLES ===\n\n")
cat("# Predict a single game:\n")
cat("  manual_predict('KC', 'BUF', week = 10)\n\n")
cat("# Or use the detailed function:\n")
cat("  result <- predict_game('KC', 'BUF', week = 10)\n\n")
cat("# Predict all games in a week:\n")
cat("  week_predictions <- predict_week(10)\n\n")
cat("# View team rankings:\n")
cat("  rankings <- get_team_rankings()\n")
cat("  View(rankings)\n\n")

#cat("Available teams:\n")
#cat(paste(sort(available_teams), collapse = ", "), "\n\n")

cat("System ready for predictions!\n")