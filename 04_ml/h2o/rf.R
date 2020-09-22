library(janitor)
library(tidymodels)
library(h2o)
library(tidyverse)

# Load data ----

source('01_functions/load_data.R')



# Functions ----

source('01_functions/train_grid.R')

# Recipes ----
rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()


train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)


# RandomForest ----

h2o.init()

h2o.removeAll()

grid <- grid_random(parameters(finalize(mtry(), train_featured_baked),
                               tree_depth(),
                               min_n()),
                    size = 2000)

params <- list(mtries = unique(grid$mtry),
               max_depth = unique(grid$tree_depth),
               min_rows = unique(grid$min_n),
               ntrees = c(200, 500, 1000, 2000, 2500),
               sample_rate = seq(0.6, 1, 0.05),
               col_sample_rate_per_tree = seq(0.6, 1, 0.05),
               fold_assignment = c('Random', 'Stratified'),
               distribution = 'bernoulli')

search_criteria = list(strategy = 'RandomDiscrete',
                       stopping_metric = 'AUC',
                       stopping_rounds = 5,
                       max_models = 10,
                       seed = 11)

train_grid(algorithm = 'randomForest', 
           data = train_featured_baked, 
           grid_id = 'rf_grid_featured', 
           params = params, 
           search_criteria = search_criteria)

rf_grid <- h2o.loadGrid('grids/rf_grid_featured/rf_grid_featured')
top_rf <- h2o.getModel(rf_grid@model_ids[[1]])

h2o.performance(top_rf, as.h2o(test_featured_baked))

h2o.varimp_plot(top_rf)