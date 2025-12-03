# ==============================================================================
# NFL PREDICTION MODEL - SIMPLIFIED DASHBOARD
# Script: dashboard.R
# Purpose: Interactive dashboard for predictions and analysis
# ==============================================================================

library(shiny)
library(shinydashboard)
library(DT)
library(plotly)
library(dplyr)
library(readr)
library(here)

# Source prediction functions
source(here("scripts", "04_predictions.R"))

# Load data
tryCatch({
  trained_models <- readRDS(here("models", "trained_models.rds"))
  performance_data <- read_csv(here("models", "model_performance.csv"))
  feature_importance <- read_csv(here("models", "feature_importance.csv"))
  #feature_set_comparison <- read_csv(here("models", "feature_set_comparison.csv"))
  team_stats_by_game <- read_rds(here("data", "processed", "team_stats_by_game.rds"))
  
  team_list <- sort(unique(team_stats_by_game$posteam))
  data_loaded <- TRUE
}, error = function(e) {
  data_loaded <- FALSE
  print(paste("Error loading data:", e$message))
})

# ==============================================================================
# UI
# ==============================================================================

ui <- dashboardPage(
  dashboardHeader(title = "NFL Prediction System"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Make Predictions", tabName = "predict", icon = icon("football")),
      menuItem("Model Performance", tabName = "performance", icon = icon("chart-line")),
      menuItem("Feature Analysis", tabName = "features", icon = icon("cogs")),
      menuItem("Team Rankings", tabName = "rankings", icon = icon("trophy"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .prediction-box {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-radius: 10px;
          padding: 20px;
          margin: 10px 0;
        }
        .confidence-high { background-color: #28a745; color: white; padding: 5px 10px; border-radius: 5px; }
        .confidence-medium { background-color: #ffc107; color: black; padding: 5px 10px; border-radius: 5px; }
        .confidence-low { background-color: #dc3545; color: white; padding: 5px 10px; border-radius: 5px; }
      "))
    ),
    
    tabItems(
      # ==============================================================================
      # PREDICTIONS TAB
      # ==============================================================================
      tabItem(tabName = "predict",
              h2("Make Game Predictions"),
              
              fluidRow(
                box(
                  title = "Single Game Prediction", status = "primary", solidHeader = TRUE, width = 6,
                  selectInput("home_team", "Home Team:", choices = if(data_loaded) team_list else NULL),
                  selectInput("away_team", "Away Team:", choices = if(data_loaded) team_list else NULL),
                  numericInput("game_week", "Week:", value = 1, min = 1, max = 18),
                  actionButton("predict_btn", "Predict Game", class = "btn-success"),
                  hr(),
                  uiOutput("single_prediction_result")
                ),
                
                box(
                  title = "Week Predictions", status = "info", solidHeader = TRUE, width = 6,
                  numericInput("week_number", "Week:", value = 1, min = 1, max = 18),
                  numericInput("season_year", "Season:", value = 2025, min = 2020, max = 2025),
                  actionButton("predict_week_btn", "Predict Week", class = "btn-primary"),
                  hr(),
                  textOutput("week_status")
                )
              ),
              
              fluidRow(
                box(
                  title = "Week Predictions", status = "success", solidHeader = TRUE, width = 12,
                  DT::dataTableOutput("week_predictions_table")
                )
              )
      ),
      
      # ==============================================================================
      # PERFORMANCE TAB
      # ==============================================================================
      tabItem(tabName = "performance",
              h2("Model Performance"),
              
              fluidRow(
                valueBoxOutput("best_accuracy"),
                valueBoxOutput("best_auc"),
                valueBoxOutput("best_model")
              ),
              
              fluidRow(
                box(
                  title = "Performance Metrics", status = "primary", solidHeader = TRUE, width = 12,
                  DT::dataTableOutput("performance_table")
                )
              ),
              
              fluidRow(
                box(
                  title = "Accuracy by Model", status = "info", solidHeader = TRUE, width = 6,
                  plotlyOutput("accuracy_plot")
                ),
                box(
                  title = "Feature Set Comparison", status = "warning", solidHeader = TRUE, width = 6,
                  plotlyOutput("feature_set_plot")
                )
              )
      ),
      
      # ==============================================================================
      # FEATURES TAB
      # ==============================================================================
      tabItem(tabName = "features",
              h2("Feature Analysis"),
              
              fluidRow(
                box(
                  title = "Feature Importance", status = "primary", solidHeader = TRUE, width = 12,
                  plotlyOutput("feature_importance_plot", height = "500px")
                )
              ),
              
              fluidRow(
                box(
                  title = "Top Features", status = "success", solidHeader = TRUE, width = 12,
                  DT::dataTableOutput("feature_importance_table")
                )
              )
      ),
      
      # ==============================================================================
      # RANKINGS TAB
      # ==============================================================================
      tabItem(tabName = "rankings",
              h2("Team Rankings"),
              
              fluidRow(
                box(
                  title = "Current Team Rankings", status = "primary", solidHeader = TRUE, width = 12,
                  DT::dataTableOutput("rankings_table")
                )
              )
      )
    )
  )
)

# ==============================================================================
# SERVER
# ==============================================================================

server <- function(input, output, session) {
  
  if (!data_loaded) {
    showNotification("Error loading data", type = "error", duration = NULL)
    return()
  }
  
  # Reactive values
  values <- reactiveValues(
    single_prediction = NULL,
    week_predictions = NULL
  )
  
  # ==============================================================================
  # PREDICTIONS
  # ==============================================================================
  
  # Single game prediction
  observeEvent(input$predict_btn, {
    withProgress(message = 'Making prediction...', value = 0.5, {
      values$single_prediction <- predict_game(
        input$home_team,
        input$away_team,
        input$game_week
      )
    })
  })
  
  output$single_prediction_result <- renderUI({
    req(values$single_prediction)
    pred <- values$single_prediction
    
    confidence_class <- if(pred$confidence >= 70) "confidence-high" else 
      if(pred$confidence >= 60) "confidence-medium" else "confidence-low"
    
    div(class = "prediction-box",
        h3(paste(pred$away_team, "@", pred$home_team)),
        h4(paste("Week", pred$week)),
        hr(style = "border-color: white;"),
        h3(paste("Winner:", pred$predicted_winner)),
        h4(span(class = confidence_class, paste0("Confidence: ", pred$confidence, "%"))),
        hr(style = "border-color: white;"),
        p(paste("Logistic:", pred$logistic_prob)),
        p(paste("Random Forest:", pred$rf_prob)),
        p(paste("XGBoost:", pred$xgb_prob)),
        p(paste("Ensemble:", pred$ensemble_prob))
    )
  })
  
  # Week predictions
  observeEvent(input$predict_week_btn, {
    withProgress(message = 'Predicting week...', value = 0.5, {
      values$week_predictions <- predict_week(input$week_number, input$season_year)
    })
  })
  
  output$week_status <- renderText({
    if (is.null(values$week_predictions)) {
      "Click 'Predict Week' to see predictions"
    } else if (nrow(values$week_predictions) == 0) {
      "No games found"
    } else {
      paste("Predicted", nrow(values$week_predictions), "games")
    }
  })
  
  output$week_predictions_table <- DT::renderDataTable({
    req(values$week_predictions)
    
    values$week_predictions %>%
      select(away_team, home_team, predicted_winner, confidence, ensemble_prob, gameday) %>%
      arrange(desc(confidence))
  }, options = list(pageLength = 20))
  
  # ==============================================================================
  # PERFORMANCE
  # ==============================================================================
  
  output$best_accuracy <- renderValueBox({
    best <- performance_data %>% 
      filter(Dataset == "Test") %>% 
      slice_max(Accuracy, n = 1)
    
    valueBox(
      value = paste0(round(best$Accuracy * 100, 1), "%"),
      subtitle = "Best Test Accuracy",
      icon = icon("bullseye"),
      color = "green"
    )
  })
  
  output$best_auc <- renderValueBox({
    best <- performance_data %>% 
      filter(Dataset == "Test") %>% 
      slice_max(AUC, n = 1)
    
    valueBox(
      value = round(best$AUC, 3),
      subtitle = "Best Test AUC",
      icon = icon("chart-area"),
      color = "blue"
    )
  })
  
  output$best_model <- renderValueBox({
    best <- performance_data %>% 
      filter(Dataset == "Test") %>% 
      slice_max(Accuracy, n = 1)
    
    valueBox(
      value = best$Model,
      subtitle = "Best Performing Model",
      icon = icon("trophy"),
      color = "yellow"
    )
  })
  
  output$performance_table <- DT::renderDataTable({
    performance_data %>%
      arrange(Dataset, desc(Accuracy))
  }, options = list(pageLength = 10))
  
  output$accuracy_plot <- renderPlotly({
    plot_ly(performance_data, x = ~Model, y = ~Accuracy, color = ~Dataset,
            type = "bar") %>%
      layout(barmode = "group", yaxis = list(title = "Accuracy"))
  })
  
  #output$feature_set_plot <- renderPlotly({
   # plot_ly(feature_set_comparison, x = ~Feature_Set, y = ~Test_Accuracy,
        #    type = "bar", marker = list(color = "steelblue")) %>%
     # layout(yaxis = list(title = "Test Accuracy"))
  #})
  
  # ==============================================================================
  # FEATURES
  # ==============================================================================
  
  output$feature_importance_plot <- renderPlotly({
    plot_ly(feature_importance, x = ~Avg_Importance, y = ~reorder(Feature, Avg_Importance),
            type = "bar", orientation = "h", marker = list(color = "forestgreen")) %>%
      layout(yaxis = list(title = ""), xaxis = list(title = "Importance"))
  })
  
  output$feature_importance_table <- DT::renderDataTable({
    feature_importance %>%
      mutate(Avg_Importance = round(Avg_Importance, 4))
  }, options = list(pageLength = 20))
  
  # ==============================================================================
  # RANKINGS
  # ==============================================================================
  
  output$rankings_table <- DT::renderDataTable({
    get_team_rankings()
  }, options = list(pageLength = 32))
}

# ==============================================================================
# RUN APP
# ==============================================================================

shinyApp(ui = ui, server = server)