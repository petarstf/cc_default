library(AmesHousing)
library(janitor)
library(tidymodels)
library(treesnip)
library(tidyverse)
library(doParallel)

sessionInfo()
all_cores <- parallel::detectCores(logical = FALSE) 
registerDoParallel(cores = all_cores)


set.seed(11)
data <- make_ames() %>% clean_names()

split <- initial_split(data, prop = 0.8, strata = sale_price)
train_data <- split %>% training()
test_data <- split %>% testing()

rec <- recipe(sale_price ~ ., train_data) %>% 
  step_other(all_nominal(), threshold = 0.01) %>% 
  step_nzv(all_nominal()) %>% 
  prep()

train_baked <- bake(rec, train_data)
test_baked <- bake(rec, test_data)

folds <- vfold_cv(train_baked, v = 5, strata = sale_price)

lgbm_mod <- boost_tree(mode = 'regression',
                       trees = 1000,
                       min_n = tune(),
                       tree_depth = tune()) %>% 
  set_engine('lightgbm', objective = 'binary', verbose = 2)

lgbm_mod %>% parameters

grid <- grid_max_entropy(parameters(min_n(),
                                    tree_depth()),
                         size = 30)

lgbm_res <- lgbm_mod %>% 
  tune_grid(sale_price ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(rmse, rsq, mae),
            control = control_grid(verbose = T, save_pred = T))

lgbm_res %>% 
  select(.notes) %>% 
  unnest(.notes)

lgbm_tuned %>%
  tune::show_best(metric = "rmse",n = 5)