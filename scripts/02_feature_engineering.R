# ==============================================================================
# NFL PREDICTION MODEL - TIME-AWARE FEATURE ENGINEERING
# Script: 02_feature_engineering.R
# Purpose: Calculate rolling statistics that respect temporal order
# ==============================================================================

library(dplyr)
library(tidyr)
library(readr)
library(here)
library(zoo)

# Set up paths
data_raw_path <- here("data", "raw")
data_processed_path <- here("data", "processed")

print("=== LOADING RAW DATA ===")

pbp_raw <- read_rds(here(data_raw_path, "pbp_raw.rds"))
games_raw <- read_rds(here(data_raw_path, "games_raw.rds"))
team_divisions <- read_rds(here(data_raw_path, "team_divisions.rds"))

print("✓ Data loaded")

# ==============================================================================
# CONFIGURATION: DEFINE FEATURE SETS
# ==============================================================================

# This makes it easy to experiment with different features
FEATURE_CONFIG <- list(
  
  # Basic EPA features
  basic_epa = c(
    "epa_per_play",
    "success_rate"
  ),
  
  # Advanced EPA features
  advanced_epa = c(
    "explosive_play_rate",  # plays with EPA > 0.5
    "stuff_rate"  # plays with EPA < -0.5
  ),
  
  # Rushing stats
  rushing = c(
    "rush_epa_per_play",
    "rush_success_rate",
    "rush_yards_per_play"
  ),
  
  # Passing stats  
  passing = c(
    "pass_epa_per_play",
    "pass_success_rate",
    "completion_rate",
    "air_yards_per_att",
    "yac_per_completion"
  ),
  
  # Situational stats
  situational = c(
    "red_zone_td_rate",
    "third_down_conv_rate",
    "fourth_down_conv_rate"
  ),
  
  # Turnovers and penalties
  turnovers = c(
    "int_rate",
    "fumble_rate",
    "takeaway_rate"
  ),
  
  # Pressure metrics
  pressure = c(
    "sack_rate",
    "pressure_rate"
  )
)

# Define which feature sets to use (easy to toggle on/off)
ACTIVE_FEATURE_SETS <- c(
  "basic_epa",
  "rushing", 
  "passing",
  "situational",
  "turnovers"
)

# Rolling window size (how many games to look back)
ROLLING_WINDOW <- 7  # Last 4 games

print("=== FEATURE CONFIGURATION ===")
print(paste("Active feature sets:", paste(ACTIVE_FEATURE_SETS, collapse = ", ")))
print(paste("Rolling window:", ROLLING_WINDOW, "games"))

# ==============================================================================
# CALCULATE GAME-LEVEL TEAM STATISTICS
# ==============================================================================

print("=== CALCULATING GAME-LEVEL STATS ===")

# Function to calculate stats for a single game
calculate_game_stats <- function(pbp_data) {
  
  stats <- pbp_data %>%
    filter(!is.na(epa)) %>%
    summarise(
      plays = n(),
      
      # Basic EPA
      epa_per_play = mean(epa, na.rm = TRUE),
      success_rate = mean(success, na.rm = TRUE),
      
      # Advanced EPA
      explosive_play_rate = mean(epa > 0.5, na.rm = TRUE),
      stuff_rate = mean(epa < -0.5, na.rm = TRUE),
      
      # Rushing
      rush_plays = sum(rush == 1, na.rm = TRUE),
      rush_epa_per_play = mean(epa[rush == 1], na.rm = TRUE),
      rush_success_rate = mean(success[rush == 1], na.rm = TRUE),
      rush_yards_per_play = mean(yards_gained[rush == 1], na.rm = TRUE),
      
      # Passing
      pass_plays = sum(pass == 1, na.rm = TRUE),
      pass_epa_per_play = mean(epa[pass == 1], na.rm = TRUE),
      pass_success_rate = mean(success[pass == 1], na.rm = TRUE),
      completion_rate = mean(complete_pass[pass == 1], na.rm = TRUE),
      air_yards_per_att = mean(air_yards[pass == 1], na.rm = TRUE),
      yac_per_completion = mean(yards_after_catch[complete_pass == 1], na.rm = TRUE),
      
      # Situational
      red_zone_plays = sum(yardline_100 <= 20, na.rm = TRUE),
      red_zone_td_rate = mean(touchdown[yardline_100 <= 20], na.rm = TRUE),
      third_down_plays = sum(down == 3, na.rm = TRUE),
      third_down_conv_rate = mean(third_down_converted[down == 3], na.rm = TRUE),
      fourth_down_plays = sum(down == 4, na.rm = TRUE),
      fourth_down_conv_rate = mean(fourth_down_converted[down == 4], na.rm = TRUE),
      
      # Turnovers
      int_rate = sum(interception, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
      fumble_rate = sum(fumble_lost, na.rm = TRUE) / plays,
      takeaway_rate = (sum(interception, na.rm = TRUE) + sum(fumble_forced, na.rm = TRUE)) / plays,
      
      # Pressure
      sack_rate = sum(sack, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
      pressure_rate = sum(!is.na(qb_hit) & qb_hit == 1, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE)
    ) %>%
    # Replace NaN and Inf with 0
    mutate(across(where(is.numeric), ~ifelse(is.na(.x) | is.infinite(.x), 0, .x)))
  
  return(stats)
}

# Calculate offensive stats by game
print("Calculating offensive stats...")
offensive_game_stats <- pbp_raw %>%
  filter(!is.na(posteam)) %>%
  group_by(game_id, season, week, posteam) %>%
  group_modify(~calculate_game_stats(.x)) %>%
  ungroup() %>%
  rename_with(~paste0("off_", .x), .cols = -c(game_id, season, week, posteam))

# Calculate defensive stats by game  
print("Calculating defensive stats...")
defensive_game_stats <- pbp_raw %>%
  filter(!is.na(defteam)) %>%
  group_by(game_id, season, week, defteam) %>%
  group_modify(~calculate_game_stats(.x)) %>%
  ungroup() %>%
  rename(posteam = defteam) %>%
  rename_with(~paste0("def_", .x), .cols = -c(game_id, season, week, posteam))

print(paste("✓ Calculated stats for", n_distinct(offensive_game_stats$game_id), "games"))

# ==============================================================================
# CALCULATE ROLLING AVERAGES (TIME-AWARE)
# ==============================================================================

print("=== CALCULATING ROLLING STATISTICS ===")

# Function to calculate rolling stats for a team
calculate_rolling_stats <- function(team_data, window = ROLLING_WINDOW) {
  
  team_data <- team_data %>%
    arrange(season, week) %>%
    mutate(
      # Create a unique game index for proper ordering
      game_index = row_number()
    )
  
  # Get all numeric columns (the stats we calculated)
  stat_cols <- names(team_data)[sapply(team_data, is.numeric) & 
                                  !names(team_data) %in% c("season", "week", "game_index")]
  
  # Calculate rolling means for each stat
  rolling_data <- team_data
  
  for (col in stat_cols) {
    # Use lag to ensure we only use PAST games
    rolling_data[[paste0(col, "_rolling")]] <- rollapply(
      lag(team_data[[col]], 1),  # Lag by 1 to exclude current game
      width = window,
      FUN = mean,
      fill = NA,
      align = "right",
      na.rm = TRUE
    )
  }
  
  return(rolling_data)
}

# Calculate rolling offensive stats
print("Calculating rolling offensive stats...")
offensive_rolling <- offensive_game_stats %>%
  group_by(posteam) %>%
  group_modify(~calculate_rolling_stats(.x)) %>%
  ungroup()

# Calculate rolling defensive stats
print("Calculating rolling defensive stats...")
defensive_rolling <- defensive_game_stats %>%
  group_by(posteam) %>%
  group_modify(~calculate_rolling_stats(.x)) %>%
  ungroup()

print("✓ Rolling statistics calculated")

# ==============================================================================
# COMBINE OFFENSIVE AND DEFENSIVE STATS
# ==============================================================================

print("=== COMBINING TEAM STATISTICS ===")

team_stats_by_game <- offensive_rolling %>%
  left_join(
    defensive_rolling,
    by = c("game_id", "season", "week", "posteam")
  ) %>%
  # Calculate net metrics
  mutate(
    net_epa_per_play_rolling = off_epa_per_play_rolling - def_epa_per_play_rolling,
    net_success_rate_rolling = off_success_rate_rolling - def_success_rate_rolling,
    turnover_margin_rolling = def_takeaway_rate_rolling - (off_int_rate_rolling + off_fumble_rate_rolling)
  )

print(paste("✓ Combined stats for", nrow(team_stats_by_game), "team-game observations"))

# ==============================================================================
# CREATE MATCHUP FEATURES
# ==============================================================================

print("=== CREATING MATCHUP FEATURES ===")

# Start with games that have results
games_with_results <- games_raw %>%
  filter(!is.na(result), season >= min(team_stats_by_game$season))

# Function to create features for a game
create_matchup_features <- function(game_row, team_stats) {
  
  game_id <- game_row$game_id
  season <- game_row$season
  week <- game_row$week
  home_team <- game_row$home_team
  away_team <- game_row$away_team
  
  # Get home team stats UP TO but NOT INCLUDING this week
  home_stats <- team_stats %>%
    filter(posteam == home_team, season == !!season, week < !!week) %>%
    arrange(desc(week)) %>%
    slice(1)
  
  # Get away team stats UP TO but NOT INCLUDING this week
  away_stats <- team_stats %>%
    filter(posteam == away_team, season == !!season, week < !!week) %>%
    arrange(desc(week)) %>%
    slice(1)
  
  # If either team has no prior stats, return NULL
  if (nrow(home_stats) == 0 || nrow(away_stats) == 0) {
    return(NULL)
  }
  
  # Create matchup differentials
  features <- data.frame(
    game_id = game_id,
    season = season,
    week = week,
    home_team = home_team,
    away_team = away_team
  )
  
  # EPA advantages (offense vs opponent's defense)
  features$epa_advantage <- (home_stats$off_epa_per_play_rolling - away_stats$def_epa_per_play_rolling) -
    (away_stats$off_epa_per_play_rolling - home_stats$def_epa_per_play_rolling)
  
  features$success_rate_advantage <- (home_stats$off_success_rate_rolling - away_stats$def_success_rate_rolling) -
    (away_stats$off_success_rate_rolling - home_stats$def_success_rate_rolling)
  
  # Rush matchup
  features$rush_advantage <- (home_stats$off_rush_epa_per_play_rolling - away_stats$def_rush_epa_per_play_rolling) -
    (away_stats$off_rush_epa_per_play_rolling - home_stats$def_rush_epa_per_play_rolling)
  
  # Pass matchup  
  features$pass_advantage <- (home_stats$off_pass_epa_per_play_rolling - away_stats$def_pass_epa_per_play_rolling) -
    (away_stats$off_pass_epa_per_play_rolling - home_stats$def_pass_epa_per_play_rolling)
  
  # Situational advantages
  features$red_zone_advantage <- (home_stats$off_red_zone_td_rate_rolling - away_stats$def_red_zone_td_rate_rolling) -
    (away_stats$off_red_zone_td_rate_rolling - home_stats$def_red_zone_td_rate_rolling)
  
  features$third_down_advantage <- (home_stats$off_third_down_conv_rate_rolling - away_stats$def_third_down_conv_rate_rolling) -
    (away_stats$off_third_down_conv_rate_rolling - home_stats$def_third_down_conv_rate_rolling)
  
  # Turnover advantage
  features$turnover_advantage <- home_stats$turnover_margin_rolling - away_stats$turnover_margin_rolling
  
  # Team strength (absolute not relative)
  features$home_net_epa <- home_stats$net_epa_per_play_rolling
  features$away_net_epa <- away_stats$net_epa_per_play_rolling
  
  features$home_net_success <- home_stats$net_success_rate_rolling
  features$away_net_success <- away_stats$net_success_rate_rolling
  
  # Pressure differential
  features$pressure_advantage <- (home_stats$def_sack_rate_rolling - away_stats$off_sack_rate_rolling) -
    (away_stats$def_sack_rate_rolling - home_stats$off_sack_rate_rolling)
  
  return(features)
}

# Create features for all games
print("Creating matchup features for all games...")
all_matchup_features <- lapply(1:nrow(games_with_results), function(i) {
  if (i %% 500 == 0) cat("Processing game", i, "of", nrow(games_with_results), "\n")
  create_matchup_features(games_with_results[i, ], team_stats_by_game)
})

# Combine and remove NULLs
matchup_features <- bind_rows(all_matchup_features[!sapply(all_matchup_features, is.null)])

print(paste("✓ Created matchup features for", nrow(matchup_features), "games"))
print(paste("  (Excluded", nrow(games_with_results) - nrow(matchup_features), "games without sufficient history)"))

# ==============================================================================
# ADD GAME CONTEXT FEATURES
# ==============================================================================

print("=== ADDING GAME CONTEXT ===")

# Add divisional game indicator
matchup_features <- matchup_features %>%
  left_join(
    team_divisions %>% select(team_abbr, division) %>% rename(home_division = division),
    by = c("home_team" = "team_abbr")
  ) %>%
  left_join(
    team_divisions %>% select(team_abbr, division) %>% rename(away_division = division),
    by = c("away_team" = "team_abbr")
  ) %>%
  mutate(
    divisional_game = as.numeric(home_division == away_division),
    
    # Season phase
    season_phase = case_when(
      week <= 6 ~ "Early",
      week <= 12 ~ "Mid",
      week <= 18 ~ "Late",
      TRUE ~ "Playoffs"
    ),
    
    # Week number as feature
    week_num = week
  ) %>%
  select(-home_division, -away_division)

# Add game outcomes
matchup_features <- matchup_features %>%
  left_join(
    games_with_results %>% select(game_id, result, home_score, away_score),
    by = "game_id"
  ) %>%
  mutate(
    home_win = as.factor(ifelse(result > 0, "Win", "Loss")),
    home_win_binary = as.numeric(result > 0),
    point_differential = result
  )

# Remove any rows with missing features
modeling_data <- matchup_features %>%
  filter(complete.cases(select(., -game_id, -home_team, -away_team)))

print(paste("✓ Final modeling dataset:", nrow(modeling_data), "games"))

# ==============================================================================
# SAVE PROCESSED DATA
# ==============================================================================

print("=== SAVING PROCESSED DATA ===")

# Save team stats by game (for future predictions)
write_rds(team_stats_by_game, here(data_processed_path, "team_stats_by_game.rds"))
write_csv(team_stats_by_game, here(data_processed_path, "team_stats_by_game.csv"))

# Save modeling data
write_rds(modeling_data, here(data_processed_path, "modeling_data.rds"))
write_csv(modeling_data, here(data_processed_path, "modeling_data.csv"))

print("✓ All processed data saved")

# ==============================================================================
# FEATURE ENGINEERING SUMMARY
# ==============================================================================

print("=== FEATURE ENGINEERING SUMMARY ===")

# Check temporal integrity
temporal_check <- modeling_data %>%
  group_by(season) %>%
  summarise(
    games = n(),
    min_week = min(week),
    avg_week = mean(week),
    max_week = max(week)
  )

print("Games by season:")
print(temporal_check)

# Feature summary
cat("\nAvailable features:\n")
feature_cols <- names(modeling_data)[!names(modeling_data) %in% 
                                       c("game_id", "season", "week", "home_team", "away_team", "result", 
                                         "home_score", "away_score", "home_win", "home_win_binary", "point_differential")]
cat(paste("-", feature_cols), sep = "\n")

# Target variable distribution
print("\nTarget variable distribution:")
print(table(modeling_data$home_win))

print("\n=== READY FOR MODEL TRAINING ===")
print("Next: Run 03_model_training.R")