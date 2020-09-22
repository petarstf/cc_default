library(janitor)
library(tidymodels)
library(h2o)
library(tidyverse)

# Load data ----

source('functions/load_data.R')


# Functions ----

source('functions/train_grid.R')

# Recipes ----
rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()


train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)


# MLP ----
learn_rate_opt <- c(0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1)
activation_opt <- c("RectifierWithDropout", "MaxoutWithDropout")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
hidden_dropout_ratios_opt <- seq(0, 0.5, 0.1)
hidden_opt <- c(54, 128, 256, 512, 1024)
epochs_opt <- c(10)

params <- list(activation = activation_opt,
               l1 = l1_opt,
               l2 = l2_opt,
               hidden_dropout_ratios = hidden_dropout_ratios_opt,
               hidden = hidden_opt,
               epochs = epochs_opt,
               rate = learn_rate_opt,
               rate_annealing = c(0.99, 1))

search_criteria = list(strategy = 'RandomDiscrete',
                       stopping_metric = 'logloss',
                       stopping_rounds = 5,
                       max_models = 10,
                       seed = 11)

train_grid(algorithm = 'deeplearning',
           data = train_featured_baked,
           grid_id = 'mlp_grid_featured',
           params = params,
           search_criteria = search_criteria)


mlp_grid <- h2o.loadGrid('grids/mlp_grid_featured/mlp_grid_featured')
top_mlp <- h2o.getModel(mlp_grid@model_ids[[1]])

h2o.performance(top_mlp, as.h2o(test_featured_baked))

h2o.varimp_plot(top_mlp)
