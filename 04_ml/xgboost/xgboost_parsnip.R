library(janitor)
library(tidymodels)
library(tidyverse)
library(DMwR)
library(doParallel)
library(tictoc)

# Load data ----

source('01_functions/load_data.R')
source('01_functions/get_predictions_parsnip.R')
source('01_functions/get_optimal_predictions.R')

registerDoParallel(cores = 6)

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
  step_normalize(all_numeric(), -contains('months')) %>% 
  # step_dummy(all_nominal(), -all_outcomes()) %>% 
  # step_nzv(all_predictors()) %>% 
  # step_zv(all_predictors()) %>% 
  prep()


rec_down <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>%
  step_normalize(all_numeric(), -contains('months')) %>% 
  themis::step_downsample(default, skip = F) %>% 
  prep()

rec_up <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>%
  step_normalize(all_numeric(), -contains('months')) %>% 
  themis::step_upsample(default, skip = F) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
downsample_featured_baked <- bake(rec_down, train_featured)
upsample_featured_baked <- bake(rec_up, train_featured)

test_featured_baked <- bake(rec, test_featured)

smote_train_featured <- as_tibble(SMOTE(default ~ ., as.data.frame(train_featured)))
smote_featured_baked <- bake(rec, smote_train_featured)

set.seed(11)
folds <- vfold_cv(train_featured_baked, v = 5, strata = default)
set.seed(11)
downfolds <- vfold_cv(downsample_featured_baked, v = 5, strata = default)
set.seed(11)
upfolds <- vfold_cv(upsample_featured_baked, v = 5, strata = default)
set.seed(11)
smotefolds <- vfold_cv(smote_featured_baked, v = 5, strata = default)

# XGBoost ----

xgb_mod <- boost_tree(mode = 'classification',
                      mtry = tune(),
                      trees = tune(),
                      min_n = tune(),
                      tree_depth = tune(),
                      learn_rate = tune(),
                      loss_reduction = tune(),
                      sample_size = tune(),
                      stop_iter = 5) %>% 
  set_engine('xgboost', tree_method = 'hist')

xgb_mod %>% parameters()

set.seed(11)
grid <- grid_regular(parameters(finalize(mtry(), train_featured_baked),
                               trees(),
                               min_n(),
                               tree_depth(),
                               learn_rate(),
                               loss_reduction(),
                               sample_prop()),
                    levels = 5) %>%
  slice_sample(n = 200)

set.seed(11)
xgb_res_hist <- xgb_cv(folds)

set.seed(11)
xgb_res_hist_norm <- xgb_cv(folds)

set.seed(11);
xgb_res_reg <- xgb_cv(folds)

set.seed(11)
tic()
xgb_downsample_res <- xgb_cv(downfolds)
toc()
set.seed(11)
tic()
xgb_upsample_res <- xgb_cv(upfolds)
toc()
set.seed(11)
tic()
xgb_smote_res <- xgb_cv(smotefolds)
toc()
save.image('env/xgboost_parsnip_env1.RData')

xgb_res %>% 
  show_best('f_meas')

xgb_res_hist %>% 
  show_best('f_meas')

xgb_res_hist_norm %>% 
  show_best('f_meas')

xgb_res_reg %>% 
  show_best('f_meas')

xgb_downsample_res %>% 
  show_best('f_meas')

xgb_upsample_res %>% 
  show_best('f_meas')

xgb_smote_res %>% 
  show_best('f_meas')

xgb_final <- finalize_model(xgb_mod, xgb_res_reg %>% select_best('f_meas')) %>%
  fit(default ~ ., train_featured_baked)
xgb_final_down <- finalize_model(xgb_mod, select_best(xgb_downsample_res, 'f_meas')) %>% 
  fit(default ~ ., downsample_featured_baked)
xgb_final_up <- finalize_model(xgb_mod, select_best(xgb_upsample_res, 'f_meas')) %>% 
  fit(default ~ ., upsample_featured_baked)
xgb_final_smote <- finalize_model(xgb_mod, select_best(xgb_smote_res, 'f_meas')) %>% 
  fit(default ~ ., smote_featured_baked)

xgb_pred <- get_predictions(xgb_final, test_featured_baked)
xgb_pred_down <- get_predictions(xgb_final_down, test_featured_baked)
xgb_pred_up <- get_predictions(xgb_final_up, test_featured_baked)
xgb_pred_smote <- get_predictions(xgb_final_smote, test_featured_baked)

xgb_metrics <- bind_rows(bind_cols(bind_rows(accuracy(xgb_pred, default, p_optimal),
                              f_meas(xgb_pred, default, p_optimal),
                              recall(xgb_pred, default, p_optimal),
                              precision(xgb_pred, default, p_optimal),
                              roc_auc(xgb_pred, default, p1)),
                    model = 'XGB - Threshold',
                    threshold = unique(xgb_pred$optimal_ts)),
          bind_cols(bind_rows(accuracy(xgb_pred, default, predict),
                              f_meas(xgb_pred, default, predict),
                              recall(xgb_pred, default, predict),
                              precision(xgb_pred, default, predict),
                              roc_auc(xgb_pred, default, p1)),
                    model = 'XGB',
                    threshold = 0.5),
          bind_cols(bind_rows(accuracy(xgb_pred_down, default, p_optimal),
                              f_meas(xgb_pred_down, default, p_optimal),
                              recall(xgb_pred_down, default, p_optimal),
                              precision(xgb_pred_down, default, p_optimal),
                              roc_auc(xgb_pred_down, default, p1)),
                    model = 'XGB - Downsample - Threshold',
                    threshold = unique(xgb_pred_down$optimal_ts)),
          bind_cols(bind_rows(accuracy(xgb_pred_down, default, predict),
                              f_meas(xgb_pred_down, default, predict),
                              recall(xgb_pred_down, default, predict),
                              precision(xgb_pred_down, default, predict),
                              roc_auc(xgb_pred_down, default, p1)),
                    model = 'XGB - Downsample',
                    threshold = 0.5),
          bind_cols(bind_rows(accuracy(xgb_pred_up, default, p_optimal),
                              f_meas(xgb_pred_up, default, p_optimal),
                              recall(xgb_pred_up, default, p_optimal),
                              precision(xgb_pred_up, default, p_optimal),
                              roc_auc(xgb_pred_up, default, p1)),
                    model = 'XGB - Upsample - Threshold',
                    threshold = unique(xgb_pred_up$optimal_ts)),
          bind_cols(bind_rows(accuracy(xgb_pred_up, default, predict),
                              f_meas(xgb_pred_up, default, predict),
                              recall(xgb_pred_up, default, predict),
                              precision(xgb_pred_up, default, predict),
                              roc_auc(xgb_pred_up, default, p1)),
                    model = 'XGB - Upsample',
                    threshold = 0.5),
          bind_cols(bind_rows(accuracy(xgb_pred_smote, default, p_optimal),
                              f_meas(xgb_pred_smote, default, p_optimal),
                              recall(xgb_pred_smote, default, p_optimal),
                              precision(xgb_pred_smote, default, p_optimal),
                              roc_auc(xgb_pred_smote, default, p1)),
                    model = 'XGB - SMOTE - Threshold',
                    threshold = unique(xgb_pred_smote$optimal_ts)),
          bind_cols(bind_rows(accuracy(xgb_pred_smote, default, predict),
                              f_meas(xgb_pred_smote, default, predict),
                              recall(xgb_pred_smote, default, predict),
                              precision(xgb_pred_smote, default, predict),
                              roc_auc(xgb_pred_smote, default, p1)),
                    model = 'XGB - SMOTE',
                    threshold = 0.5))

xgb_metrics <- xgb_metrics %>% 
  pivot_wider(names_from = .metric, values_from = .estimate)

xgb_metrics %>% 
  arrange(desc(roc_auc))

conf_xgb_reg <- plot_conf_mat(xgb_pred$predict,
                              xgb_pred$default,
                              'XGBoost - Regular')
conf_xgb_reg_ts <- plot_conf_mat(xgb_pred$p_optimal,
                                 xgb_pred$default,
                                 'XGBoost - Regular - Threshold')

conf_xgb_down <- plot_conf_mat(xgb_pred_down$predict,
                              xgb_pred_down$default,
                              'XGBoost - Downsample')
conf_xgb_down_ts <- plot_conf_mat(xgb_pred_down$p_optimal,
                                 xgb_pred_down$default,
                                 'XGBoost - Downsample - Threshold')

conf_xgb_up <- plot_conf_mat(xgb_pred_up$predict,
                              xgb_pred_up$default,
                              'XGBoost - Upsample')
conf_xgb_up_ts <- plot_conf_mat(xgb_pred_up$p_optimal,
                                 xgb_pred_up$default,
                                 'XGBoost - Upsample - Threshold')

conf_xgb_smote <- plot_conf_mat(xgb_pred_smote$predict,
                              xgb_pred_smote$default,
                              'XGBoost - SMOTE')
conf_xgb_smote_ts <- plot_conf_mat(xgb_pred_smote$p_optimal,
                                 xgb_pred_smote$default,
                                 'XGBoost - SMOTE - Threshold')


# save.image('03_env/xgboost_metrics_1.RData')

xgb_metrics %>% 
  select(-.estimator)

rm(list = setdiff(ls(), 'xgb_metrics'))
# save.image('03_env/xgb_met.RData')