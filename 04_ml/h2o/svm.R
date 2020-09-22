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

# SVM ----
h2o.init()
h2o.removeAll()


grid <- grid_random(parameters(rbf_sigma(),
                               cost()),
                    size = 200)

params <- list(gamma = unique(grid$rbf_sigma),
               hyper_param = unique(grid$cost),
               rank_ratio = c(0.01))

# params = list(gamma = -1,
#               hyper_param = 1)

search_criteria <- list(strategy = 'RandomDiscrete',
                        stopping_metric = 'AUC',
                        stopping_rounds = 5,
                        max_models = 1,
                        seed = 1)

train_grid(algorithm = 'psvm',
           data = train_featured_baked,
           grid_id = 'svm_grid_featured',
           params = params,
           search_criteria = search_criteria)

svm_grid <- h2o.loadGrid('grids/svm_grid_featured/svm_grid_featured')
top_svm <- h2o.getModel(svm_grid@model_ids[[1]])

h2o.performance(top_svm, as.h2o(top_baked))