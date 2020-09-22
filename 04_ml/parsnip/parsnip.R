library(janitor)
library(baguette)
library(tidymodels)
library(tidyverse)


# Load and prep the data ----

source('01_functions/load_data.R')
source('01_functions/train_grid.R')

doParallel::registerDoParallel()

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)

set.seed(11)
folds = vfold_cv(train_featured_baked, v = 5, strata = default)

# Bagged MARS ----

bag_mars_mod <- bag_mars(mode = 'classification',
                         num_terms = tune(),
                         prod_degree = tune(),
                         prune_method = tune()) %>% 
  set_engine('earth')

bag_mars_mod %>% 
  parameters()

grid <- grid_regular(parameters(finalize(num_terms(), train_featured_baked),
                                prod_degree(),
                                prune_method()),
                     levels = 200)

set.seed(11)
grid <- grid %>% 
  slice_sample(n = 20)

set.seed(11)
bag_mars_res <- bag_mars_mod %>% 
  tune_grid(default ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, roc_auc, pr_auc, f_meas),
            control = control_grid(verbose = T, save_pred = T))

# MARS ----

mars_mod <- mars(mode = 'classification',
                 num_terms = tune(),
                 prod_degree = tune(),
                 prune_method = tune())

set.seed(11)
mars_res <- mars_mod %>% 
  tune_grid(default ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, roc_auc, pr_auc, f_meas),
            control = control_grid(verbose = T, save_pred = T))

# KNN ----

knn_mod <- nearest_neighbor(mode = 'classification',
                            neighbors = tune(),
                            weight_func = tune(),
                            dist_power = tune()) %>% 
  set_engine('kknn')

knn_mod %>% 
  parameters()

grid <- grid_regular(parameters(neighbors(),
                                weight_func(),
                                dist_power()),
                     levels = 200)

set.seed(11)
grid <- grid %>% 
  slice_sample(n = 20)

set.seed(11)
knn_res <- knn_mod %>% 
  tune_grid(default ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, roc_auc, pr_auc, f_meas),
            control = control_grid(verbose = T, save_pred = T))

# Decision Tree ----

tree_mod <- decision_tree(mode = 'classification',
                          cost_complexity = tune(),
                          tree_depth = tune(),
                          min_n = tune()) %>% 
  set_engine('rpart')

tree_mod %>% parameters()

grid <- grid_regular(parameters(cost_complexity(),
                                tree_depth(),
                                min_n()),
                     levels = 200)

set.seed(11)
grid <- grid %>% 
  slice_sample(n = 20)

set.seed(11)
tree_res <- tree_mod %>% 
  tune_grid(default ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, roc_auc, pr_auc, f_meas),
            control = control_grid(verbose = T, save_pred = T))

# SVM ----

svm_mod <- svm_rbf(mode = 'classification',
                   cost = tune(),
                   rbf_sigma = tune()) %>% 
  set_engine('kernlab')

svm_mod %>% parameters()

grid <- grid_regular(parameters(cost(),
                                rbf_sigma()),
                     levels = 200)

set.seed(11)
grid <- grid %>% 
  slice_sample(n = 20)

svm_res <- svm_mod %>% 
  tune_grid(default ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, roc_auc, pr_auc, f_meas),
            control = control_grid(verbose = T, save_pred = T))

save.image(file='03_env/parsnip_env.RData')
