library(janitor)
library(tidymodels)
library(tidyverse)
library(DMwR)
library(treesnip)
library(doParallel)
library(tictoc)


# Data prep ----

source('01_functions/load_data.R')
source('01_functions/get_predictions_parsnip.R')
source('01_functions/get_optimal_predictions.R')

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)


lgbm_mod <- boost_tree(mode = 'classification',
                       mtry = 38,
                       trees = 344,
                       min_n = 13,
                       tree_depth = 4,
                       sample_size = 0.152577040181495,
                       stop_iter = 5) %>% 
  set_engine('lightgbm', objective = 'binary', num_leaves = 255)

set.seed(11)
lgbm <- lgbm_mod %>% 
  fit(default ~ ., train_featured_baked)

lgbm_pred <- get_predictions(lgbm, test_featured_baked)

bind_rows(accuracy(lgbm_pred, default, predict),
          f_meas(lgbm_pred, default, predict),
          precision(lgbm_pred, default, predict),
          recall(lgbm_pred, default, predict),
          roc_auc(lgbm_pred, default, p1))
