library(janitor)
library(tidymodels)
library(treesnip)
library(tidyverse)

# Load data ----

source('01_functions/load_data.R')
doParallel::registerDoParallel()

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

rec_dummy <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  prep()


train_featured_baked <- bake(rec, train_featured)
train_dummy_baked <- bake(rec, train_featured)

test_featured_baked <- bake(rec, test_featured)
test_dummy_baked <- bake(rec, test_featured)


set.seed(11)
folds = vfold_cv(train_featured_baked, v = 5, strata = default)
set.seed(11)
folds_dummy = vfold_cv(train_dummy_baked, v = 5, strata = default)

# LGBM ----

lgbm_mod <- boost_tree(mode = 'classification',
                       mtry = tune(),
                       trees = tune(),
                       min_n = tune(),
                       tree_depth = tune(),
                       sample_size = tune(),
                       stop_iter = 5) %>% 
  set_engine('lightgbm', objective = 'binary')


lgbm_mod %>% parameters()

set.seed(11)
grid <- grid_regular(parameters(finalize(mtry(), test_featured_baked %>% select(-default)),
                                min_n(),
                                tree_depth()),
                     levels = 200)

set.seed(11)
grid <- as_tibble(expand.grid(list(mtry = unique(grid$mtry), 
                                   trees = c(500, 1000, 1500),
                                   min_n = unique(grid$min_n),
                                   tree_depth = unique(grid$tree_depth),
                                   sample_size = seq(0.6, 1, 0.1))) %>% 
                    slice_sample(n = 200))


set.seed(11)
lgbm_res <- lgbm_mod %>% 
  tune_grid(default ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, precision, recall),
            control = control_grid(verbose = T, save_pred = T))


lgbm_res %>% 
  show_best('f_meas')

set.seed(11)
lgbm <- finalize_model(lgbm_mod, select_best(lgbm_res, 'f_meas')) %>% 
  fit(default ~ ., train_featured_baked)

lgbm_res %>% 
  select(.notes) %>% unlist

lgbm_pred <- bind_cols(predict(lgbm, test_featured_baked),
                       predict(lgbm, test_featured_baked, type = 'prob'),
                       test_featured_baked %>% select(default)) %>% 
  rename(predict = .pred_class, p1 = .pred_1, p0 = .pred_0)


save.image('03_env/lgbm_env.RData')

lgbm_pred <- lgbm_pred %>% 
  rename(predict = .pred_class, p1 = .pred_1, p0 = .pred_0)

r <- pROC::roc(lgbm_pred, default, p1)
optimal_ts <- r$thresholds[which.max(r$sensitivities + r$specificities)]

lgbm_pred <- lgbm_pred %>%
  mutate(optimal_ts = optimal_ts,
         p_optimal = factor(ifelse(p1 > optimal_ts, 1, 0), levels = c(1, 0)))

bind_rows(bind_cols(bind_rows(accuracy(lgbm_pred, predict, default),
                              f_meas(lgbm_pred, predict, default)),
                    model = 'LGBM',
                    threshold = 0.5),
          bind_cols(bind_rows(accuracy(lgbm_pred, p_optimal, default),
                              f_meas(lgbm_pred, p_optimal, default)),
                    model = 'LGBM',
                    threshold = optimal_ts))
