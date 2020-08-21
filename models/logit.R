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



# Logit ----

h2o.init()
h2o.removeAll()


grid <- grid_random(parameters(mixture(),
                               penalty()),
                    size = 200)

params <- list(alpha = unique(grid$mixture),
               lambda = unique(grid$penalty))

search_criteria <- list(strategy = 'RandomDiscrete',
                        stopping_metric = 'AUC',
                        stopping_rounds = 5,
                        max_models = 1,
                        seed = 11)

train_grid(algorithm = 'glm',
           data = train_featured_baked,
           grid_id = 'glm_grid_featured',
           params = params,
           search_criteria = search_criteria)

log_grid <- h2o.loadGrid('grids/glm_grid_featured/glm_grid_featured')
log_best <- h2o.getModel(log_grid@model_ids[[1]])

h2o.performance(log_best, as.h2o(test_featured_baked))
