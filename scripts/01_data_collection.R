# ==============================================================================
# NFL PREDICTION MODEL - TIME-AWARE DATA COLLECTION
# Script: 01_data_collection.R
# Purpose: Download NFL data with proper temporal awareness
# ==============================================================================

library(nflreadr)
library(dplyr)
library(readr)
library(here)

# Configuration
SEASONS <- 2020:2024
CURRENT_SEASON <- 2025

# Set up paths
data_raw_path <- here("data", "raw")
data_processed_path <- here("data", "processed")

if (!dir.exists(data_raw_path)) dir.create(data_raw_path, recursive = TRUE)
if (!dir.exists(data_processed_path)) dir.create(data_processed_path, recursive = TRUE)

print("=== COLLECTING NFL DATA ===")
print(paste("Seasons:", min(SEASONS), "to", max(SEASONS)))

# ==============================================================================
# LOAD CORE DATA
# ==============================================================================

# Load play-by-play data (most granular level)
print("Loading play-by-play data...")
pbp_raw <- load_pbp(seasons = SEASONS)
print(paste("✓ Loaded", nrow(pbp_raw), "plays"))

# Load schedules
print("Loading schedules...")
games_raw <- load_schedules(seasons = SEASONS)
print(paste("✓ Loaded", nrow(games_raw), "games"))

# Load team info
teams <- load_teams()
print(paste("✓ Loaded", nrow(teams), "teams"))

# ==============================================================================
# CREATE TEAM DIVISION/CONFERENCE MAPPING
# ==============================================================================

team_divisions <- data.frame(
  team_abbr = c("ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", 
                "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
                "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
                "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"),
  division = c("NFC West", "NFC South", "AFC North", "AFC East", "NFC South", "NFC North", 
               "AFC North", "AFC North", "NFC East", "AFC West", "NFC North", "NFC North", 
               "AFC South", "AFC South", "AFC South", "AFC West", "AFC West", "AFC West", 
               "NFC West", "AFC East", "NFC North", "AFC East", "NFC South", "NFC East",
               "AFC East", "NFC East", "AFC North", "NFC West", "NFC West", "NFC South", 
               "AFC South", "NFC East"),
  conference = c("NFC", "NFC", "AFC", "AFC", "NFC", "NFC", "AFC", "AFC",
                 "NFC", "AFC", "NFC", "NFC", "AFC", "AFC", "AFC", "AFC",
                 "AFC", "AFC", "NFC", "AFC", "NFC", "AFC", "NFC", "NFC", 
                 "AFC", "NFC", "AFC", "NFC", "NFC", "NFC", "AFC", "NFC")
)

# ==============================================================================
# SAVE RAW DATA
# ==============================================================================

print("=== SAVING RAW DATA ===")

write_rds(pbp_raw, here(data_raw_path, "pbp_raw.rds"))
write_rds(games_raw, here(data_raw_path, "games_raw.rds"))
write_rds(teams, here(data_raw_path, "teams.rds"))
write_rds(team_divisions, here(data_raw_path, "team_divisions.rds"))

print("✓ All raw data saved")

# Summary
data_summary <- data.frame(
  Dataset = c("Play-by-Play", "Games", "Teams", "Divisions"),
  Rows = c(nrow(pbp_raw), nrow(games_raw), nrow(teams), nrow(team_divisions)),
  Seasons = c(paste(SEASONS, collapse = ", "), 
              paste(SEASONS, collapse = ", "), 
              "Current", 
              "Current")
)

print("=== DATA COLLECTION SUMMARY ===")
print(data_summary)

write_csv(data_summary, here(data_raw_path, "data_collection_summary.csv"))

print("=== DATA COLLECTION COMPLETE ===")