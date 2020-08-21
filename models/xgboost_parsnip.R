library(janitor)
library(tidymodels)
library(tidyverse)

# Load data ----

source('functions/load_data.R')

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)

set.seed(11)
folds <- vfold_cv(train_featured_baked, v = 5, strata = default)

# XGBoost ----

xgb_mod <- boost_tree(mode = 'classification',
                      mtry = tune(),
                      trees = tune(),
                      min_n = tune(),
                      tree_depth = tune(),
                      learn_rate = tune(),
                      loss_reduction = tune(),
                      sample_size = tune()) %>% 
  set_engine('xgboost')

set.seed(11)
grid <- grid_random(parameters(finalize(mtry(), train_featured_baked),
                               trees(),
                               min_n(),
                               tree_depth(),
                               learn_rate(),
                               loss_reduction(),
                               sample_prop()),
                    size = 1000)

xgb_res <- xgb_mod %>% 
  tune_grid(default ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, roc_auc, pr_auc, precision, recall, specificity),
            control = control_grid(verbose = T, save_pred = T))
