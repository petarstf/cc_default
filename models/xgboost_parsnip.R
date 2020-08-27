library(janitor)
library(tidymodels)
library(tidyverse)
library(DMwR)

# Load data ----

source('functions/load_data.R')

doParallel::registerDoParallel()

xgb_cv <- function(folds) {
  xgb_mod %>% 
    tune_grid(default ~ .,
              resamples = folds,
              grid = grid,
              metrics = metric_set(accuracy, f_meas, roc_auc, pr_auc, precision, recall, specificity),
              control = control_grid(verbose = T, save_pred = T))
}

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()


rec_down <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  themis::step_downsample(default, skip = F) %>% 
  prep()

rec_up <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  themis::step_upsample(default, skip = F) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
downsample_featured_baked <- bake(rec_down, train_featured)
upsample_featured_baked <- bake(rec_up, train_featured)

test_featured_baked <- bake(rec, test_featured)

smote_train_featured <- SMOTE(default ~ ., as.data.frame(train_featured))
smote_featured_baked <- bake(rec, smote_train_featured)

set.seed(11)
folds <- vfold_cv(train_featured_baked, v = 5, strata = default)
downfolds <- vfold_cv(downsample_featured_baked, v = 5, strata = default)
upfolds <- vfold_cv(upsample_featured_baked, v = 5, strata = default)
smotefolds <- vfold_cv(smote_featured_baked, v = 5, strata = default)

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
set.seed(11)
xgb_res <- xgb_cv(folds)

set.seed(11)
xgb_downsample_res <- xgb_cv(downfolds)

set.seed(11)
xgb_upsample_res <- xgb_cv(upfolds)

set.seed(11)
xgb_smote_res <- xgb_cv(smotefolds)


xgb_final <- finalize_model(xgb_mod, xgb_res %>% select_best('f_meas')) %>%
  fit(default ~ ., train_featured_baked)

xgb_pred <- bind_cols(predict(xgb_final, test_featured_baked),
          predict(xgb_final, test_featured_baked, type = 'prob'),
          default = test_featured_baked$default) %>%
  rename(predict = .pred_class, p1 = .pred_1, p0 = .pred_0)

xgb_pred <- get_optimal_predictions(xgb_pred)

bind_rows(bind_cols(bind_rows(accuracy(xgb_pred, default, p_optimal),
                              f_meas(xgb_pred, default, p_optimal),
                              recall(xgb_pred, default, p_optimal),
                              precision(xgb_pred, default, p_optimal)),
                    model = 'XGB',
                    threshold = unique(xgb_pred$optimal_ts)),
          bind_cols(bind_rows(accuracy(xgb_pred, default, predict),
                              f_meas(xgb_pred, default, predict),
                              recall(xgb_pred, default, predict),
                              precision(xgb_pred, default, predict)),
                    model = 'XGB optimal',
                    threshold = 0.5))
