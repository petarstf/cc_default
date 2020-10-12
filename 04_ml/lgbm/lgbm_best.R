library(janitor)
library(tictoc)
library(DataExplorer)
library(lightgbm)
library(tidyverse)
library(tidymodels)
library(doParallel)
library(vip)

# Load data ----

source('01_functions/load_data.R')
source('01_functions/get_optimal_predictions.R')
source('01_functions/evalf1.R')
source('01_functions/get_vip_lgbm.R')
source('01_functions/plot_conf_mat.R')

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

data_baked <- bake(rec, data_featured)
train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)

# val_split <- initial_split(train_featured_baked, prop = 0.8, strata = default) 
# val_train_baked <- val_split %>% training()
# val_test_baked <- val_split %>% testing()

# Train - Test split ----

x <- data.matrix(data_baked %>% select(-default))
y <- data_baked %>% mutate(default = as.numeric(as.character(default))) %>% pull(default)

x_train <- data.matrix(train_featured_baked %>% select(-default))
y_train <- train_featured_baked %>% mutate(default = as.numeric(as.character(default))) %>% pull(default)
x_test <- data.matrix(test_featured_baked %>% select(-default))
y_test <- test_featured_baked %>% mutate(default = as.numeric(as.character(default))) %>% pull(default)

dtrain <- lgb.Dataset(data = x, label = y)
dtest <- lgb.Dataset(data = x_test, label = y_test)

# dtest <- lgb.Dataset.create.valid(dtrain, data = x_test, label = y_test)

# Model ----

params <- list(objective = 'binary',
               metric = c('f1_score', 'auc'),
               learning_rate = 0.05,
               feature_fraction = 1,
               feature_fraction_seed = 11,
               num_iterations = 344L,
               min_data_in_leaf = 13L,
               max_depth = 4L,
               nthreads = 6,
               bagging_fraction = 0.152577040181495,
               bagging_seed = 11,
               lambda_l1 = 0.9,
               num_leaves = 255L)
# 
# set.seed(11)
# lgbm_cv <- lgb.cv(params = params,
#                   data = dtrain,
#                   nfold = 5,
#                   eval = evalf1,
#                   early_stopping_rounds = 50L,
#                   verbose = 2)

# cat(paste('Max f1_score:', max(as_vector(lgbm_cv$record_evals$valid$f1_score$eval))),
#     paste('Max f1_score iteration: ', which.max(as_vector(lgbm_cv$record_evals$valid$f1_score$eval))),
#     sep = '\n')
# 

# best_params <- list(objective = 'binary',
#                     learning_rate = 0.05,
#                     feature_fraction = 1,
#                     feature_fraction_seed = 11,
#                     num_iterations = 344L,
#                     min_data_in_leaf = 13L,
#                     max_depth = 4L,
#                     bagging_fraction = 0.152577040181495,
#                     bagging_seed = 11,
#                     num_leaves = 255L)
# 

dtrain <- lgb.Dataset(data = x, label = y)

set.seed(11)
lgbm <- lgb.train(params = params,
                  data = dtrain,
                  nrounds = 3000,
                  eval = evalf1,
                  force_row_wise = T,
                  reset_data = T,
                  verbose = 2)

saveRDS.lgb.Booster(object = lgbm,
                    file = '05_saved_models/lightgbm.rds',
                    raw = T)

dtrain <- lgb.Dataset(data = x_train, label = y_train)

params_down <- list(objective = 'binary',
                    metric = c('auc'),
                    feature_fraction = 0.1052632,
                    feature_fraction_seed = 11,
                    num_iterations = 218L,
                    min_data_in_leaf = 15L,
                    max_depth = 2L,
                    nthreads = 6,
                    bagging_fraction = 0.345,
                    bagging_seed = 11,
                    num_leaves = 255L)


set.seed(11)
lgbm_down <- lgb.train(params = params_down,
                       data = dtrain,
                       nrounds = 156,
                       eval = evalf1,
                       force_row_wise = T,
                       verbose = 2)

dtrain <- lgb.Dataset(data = x_train, label = y_train)

params_up <- list(objective = 'binary',
                  metric = c('auc'),
                  feature_fraction = 0.5789474,
                  feature_fraction_seed = 11,
                  num_iterations = 913L,
                  nthreads = 6,
                  min_data_in_leaf = 2L,
                  max_depth = 13L,
                  bagging_fraction = 0.729,
                  bagging_seed = 11,
                  num_leaves = 255L)

set.seed(11)
lgbm_up <- lgb.train(params = params_up,
                     data = dtrain,
                     nrounds = 156,
                     eval = evalf1,
                     force_row_wise = T,
                     verbose = 2)

dtrain <- lgb.Dataset(data = x_train, label = y_train)

params_smote <- list(objective = 'binary',
                    metric = c('auc'),
                    feature_fraction = 0.4736842,
                    feature_fraction_seed = 11,
                    num_iterations = 253L,
                    min_data_in_leaf = 4L,
                    nthreads = 6,
                    max_depth = 13L,
                    bagging_fraction = 0.688,
                    bagging_seed = 11,
                    num_leaves = 255L)

set.seed(11)
lgbm_smote <- lgb.train(params = params_smote,
                        data = dtrain,
                        nrounds = 156,
                        eval = evalf1,
                        force_row_wise = T,
                        verbose = 2)

# Variable Importance Plots ----

vip_lgbm_reg <- get_vip_lgbm(lgbm, title = 'LGBM - Regular')
vip_lgbm_down <- get_vip_lgbm(lgbm_down, title = 'LGBM - Downsample')
vip_lgbm_up <- get_vip_lgbm(lgbm_up, title = 'LGBM - Upsample')
vip_lgbm_smote <- get_vip_lgbm(lgbm_smote, title = 'LGBM - SMOTE')

get_lgb_pred <- function(model, test) {
  pred <- tibble(p1 = predict(model, test),
         predict = factor(ifelse(p1 > 0.5, 1, 0), levels = c(1, 0)),
         default = test_featured_baked$default)
  get_optimal_predictions(pred)
}

lgbm_pred <- get_lgb_pred(lgbm, x_test)
lgbm_down_pred <- get_lgb_pred(lgbm_down, x_test)
lgbm_up_pred <- get_lgb_pred(lgbm_up, x_test)
lgbm_smote_pred <- get_lgb_pred(lgbm_smote, x_test)

lgbm_pred <- tibble(p1 = predict(lgbm, x_test),
                    predict = factor(ifelse(p1 > 0.5, 1, 0), levels = c(1, 0)),
                    default = test_featured_baked$default)
lgbm_pred <- get_optimal_predictions(lgbm_pred)

get_results <- function(predictions, model_name) {
  bind_rows(bind_cols(bind_rows(accuracy(predictions, default, predict),
                                f_meas(predictions, default, predict),
                                precision(predictions, default, predict),
                                recall(predictions, default, predict),
                                roc_auc(predictions, default, p1)),
                      model = model_name,
                      threshold = 0.5),
            bind_cols(bind_rows(accuracy(predictions, default, p_optimal),
                                f_meas(predictions, default, p_optimal),
                                precision(predictions, default, p_optimal),
                                recall(predictions, default, p_optimal),
                                roc_auc(predictions, default, p1)),
                      model = paste(model_name, '- Threshold'),
                      threshold = unique(predictions$optimal_ts)))
}

lgbm_metrics <- bind_rows(get_results(lgbm_pred, 'Lightgbm'),
                          get_results(lgbm_down_pred, 'Lightgbm Downsample'),
                          get_results(lgbm_up_pred, 'Lightgbm Upsample'),
                          get_results(lgbm_smote_pred, 'Lightgbm SMOTE')) %>% 
  select(-.estimator) %>% 
  pivot_wider(names_from = .metric, values_from = .estimate)

lgbm_metrics %>% view


rm(list = (setdiff(ls(), 'lgbm_metrics')))

save.image('03_env/lgbm_met.RData')

conf_lgbm_reg <- plot_conf_mat(lgbm_pred$predict, lgbm_pred$default, 'LightGBM')
conf_lgbm_down <- plot_conf_mat(lgbm_down_pred$predict, lgbm_down_pred$default, 'LightGBM Downsample')
conf_lgbm_up <- plot_conf_mat(lgbm_up_pred$predict, lgbm_up_pred$default, 'LightGBM Upsample')
conf_lgbm_smote <- plot_conf_mat(lgbm_pred$predict, lgbm_pred$default, 'LightGBM SMOTE')

conf_lgbm_reg_ts <- plot_conf_mat(lgbm_pred$p_optimal, lgbm_pred$default, 'LightGBM - Threshold')
conf_lgbm_down_ts <- plot_conf_mat(lgbm_down_pred$p_optimal, lgbm_down_pred$default, 'LightGBM Downsample - Threshold')
conf_lgbm_up_ts <- plot_conf_mat(lgbm_up_pred$p_optimal, lgbm_up_pred$default, 'LightGBM Upsample - Threshold')
conf_lgbm_smote_ts <- plot_conf_mat(lgbm_smote_pred$p_optimal, lgbm_smote_pred$default, 'LightGBM SMOTE - Threshold')

save.image('03_env/lgbm_best.RData')

rm(list = (setdiff(ls(), ls(pattern = 'conf_'))))

save.image('03_env/lgbm_conf.RData')
# 
# bind_rows(bind_cols(bind_rows(accuracy(lgbm_preds, default, predict),
#                               f_meas(lgbm_preds, default, predict),
#                               precision(lgbm_preds, default, predict),
#                               recall(lgbm_preds, default, predict),
#                               roc_auc(lgbm_preds, default, p1)),
#                     model = 'LightGBM - Regular'),
#           bind_cols(bind_rows(accuracy(lgbm_preds, default, p_optimal),
#                               f_meas(lgbm_preds, default, p_optimal),
#                               precision(lgbm_preds, default, p_optimal),
#                               recall(lgbm_preds, default, p_optimal),
#                               roc_auc(lgbm_preds, default, p1)),
#                     model = 'LightGBM - Regular - Threshold'))

# lgbm <- lgbm$save_model_to_string()
# sink('05_saved_models/lightgbm.txt')
# cat(lgbm)
# sink()

lgb.save(lgbm, '05_saved_models/lightgbm.txt', num_iteration = 156)
