library(janitor)
library(tidymodels)
library(h2o)
library(tidyverse)

# Load data ----

source('functions/load_data.R')


# Functions ----

source('functions/train_grid.R')

# Recipes ----

# Recipes ----
rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()


train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)


# Stack ----

h2o.init()
h2o.removeAll()

gbm_grid <- h2o.loadGrid('grids/gbm_grid_featured/gbm_grid_featured')
glm_grid <- h2o.loadGrid('grids/glm_grid_featured/glm_grid_featured')
rf_grid <- h2o.loadGrid('grids/rf_grid_featured/rf_grid_featured')
nb_grid <- h2o.loadGrid('grids/nb_grid_featured/nb_grid_featured')
mlp_grid <- h2o.loadGrid('grids/mlp_grid_featured/mlp_grid_featured')


top_gbm <- h2o.getModel(gbm_grid@model_ids[[1]])
top_mlp <- h2o.getModel(mlp_grid@model_ids[[1]])
top_rf <- h2o.getModel(rf_grid@model_ids[[1]])

stacked_gbm <- h2o.stackedEnsemble(y = 'default',
                    training_frame = as.h2o(train_featured_baked),
                    model_id = 'stack_ensemble_gbm',
                    metalearner_algorithm = 'gbm',
                    metalearner_params = top_gbm@parameters,
                    base_models = c(gbm_grid,
                                    glm_grid,
                                    rf_grid,
                                    nb_grid,
                                    mlp_grid),
                    metalearner_fold_assignment = 'Random',
                    seed = 11)

h2o.saveModel(stacked_gbm, path = 'models/')



stacked_mlp <- h2o.stackedEnsemble(y = 'default',
                                   training_frame = as.h2o(train_featured_baked),
                                   model_id = 'stack_ensemble_gbm',
                                   metalearner_algorithm = 'deeplearning',
                                   metalearner_params = top_mlp@parameters,
                                   base_models = c(gbm_grid,
                                                   glm_grid,
                                                   rf_grid,
                                                   nb_grid,
                                                   mlp_grid),
                                   metalearner_fold_assignment = 'Random',
                                   seed = 11)

h2o.saveModel(stacked_mlp, path = 'models/')





stacked_mlp <- h2o.stackedEnsemble(y = 'default',
                                   training_frame = as.h2o(train_featured_baked),
                                   model_id = 'stack_ensemble_gbm',
                                   metalearner_algorithm = 'drf',
                                   metalearner_params = top_rf@parameters,
                                   base_models = c(gbm_grid,
                                                   glm_grid,
                                                   rf_grid,
                                                   nb_grid,
                                                   mlp_grid),
                                   metalearner_fold_assignment = 'Random',
                                   seed = 11)

h2o.saveModel(stacked_mlp, path = 'models/')
