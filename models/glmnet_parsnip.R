library(janitor)
library(tidymodels)
library(tidyverse)
library(DMwR)
library(doParallel)
library(tictoc)

# Load data ----

source('functions/load_data.R')

source('functions/get_predictions_parsnip.R')
source('functions/get_optimal_predictions.R')


registerDoParallel(cores = 6)

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)

folds = vfold_cv(train_featured_baked, v = 5, strata = default)

# Modelling ----

glm_mod <- logistic_reg(mode = 'classification',
                        penalty = tune(),
                        mixture = tune()) %>% 
  set_engine('glmnet')

set.seed(11)
grid <- grid_max_entropy(parameters(penalty(),
                                    mixture()),
                         size = 500)

glm_res <- glm_mod %>% 
  tune_grid(default ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, roc_auc, precision, recall),
            control = control_grid(verbose = T, save_pred = T))

glm <- finalize_model(glm_mod, select_best(glm_res, 'f_meas')) %>% 
  fit(default ~ ., train_featured_baked)

glm_pred <- get_predictions(glm, test_featured_baked)

r <- pROC::roc(glm_pred, default, p1)
optimal_ts <- r$thresholds[which.max(r$sensitivities + r$specificities)]

metrics <- bind_rows(bind_cols(bind_rows(accuracy(glm_pred, default, predict),
                              f_meas(glm_pred, default, predict),
                              precision(glm_pred, default, predict),
                              recall(glm_pred, default, predict),
                              roc_auc(glm_pred, default, p1)),
                    model = 'GLMNET',
                    threshold = 0.5), 
            bind_cols(bind_rows(accuracy(glm_pred, default, p_optimal),
                                f_meas(glm_pred, default, p_optimal),
                                precision(glm_pred, default, p_optimal),
                                recall(glm_pred, default, p_optimal),
                                roc_auc(glm_pred, default, p1)),
                      model = 'GLMNET - Threshold',
                      threshold = optimal_ts))

# GLM x3 x7 x9 ----

set.seed(11)
grid <- grid_max_entropy(parameters(glm_mod),
                         size = 500)
  
glm3_res <- glm_mod %>% 
  tune_grid(default ~ sex + pay_1 + pay_3,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, precision, recall, roc_auc),
            control = control_grid(verbose = T, save_pred = T))

glm3 <- finalize_model(glm_mod, select_best(glm3_res, 'f_meas')) %>% 
  fit(default ~ sex + pay_1 + pay_3, train_featured_baked)

glm3_pred <- get_predictions(glm3, test_featured_baked)

metrics <- bind_rows(metrics,
                     bind_rows(bind_cols(bind_rows(accuracy(glm3_pred, default, predict),
                                                  f_meas(glm3_pred, default, predict),
                                                  precision(glm3_pred, default, predict),
                                                  recall(glm3_pred, default, predict),
                                                  roc_auc(glm3_pred, default, p1)),
                              model = 'GLMNET3',
                              threshold = 0.5), 
                    bind_cols(bind_rows(accuracy(glm3_pred, default, p_optimal),
                                        f_meas(glm3_pred, default, p_optimal),
                                        precision(glm3_pred, default, p_optimal),
                                        recall(glm3_pred, default, p_optimal),
                                        roc_auc(glm3_pred, default, p1)),
                              model = 'GLMNET3 - Threshold',
                              threshold = optimal_ts))) %>% 
  pivot_wider(names_from = .metric, values_from = .estimate) %>% 
  select(-.estimator)

save.image('env/glmnet_metrics.RData')


vip::vip(glm)

vip::vip(glm3)
