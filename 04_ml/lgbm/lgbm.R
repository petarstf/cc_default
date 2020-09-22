library(janitor)
library(lightgbm)
library(tidymodels)
library(tidyverse)

# Load data ----

source('functions/load_data.R')

# Functions ----

source('functions/train_automl.R')
source('functions/train_grid.R')
source('functions/get_predictions.R')
source('functions/get_optimal_predictions.R')


# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
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

# LGBM ----

params <- list(objective = 'binary',
               learning_rate = 0.01,
               num_iterations = 500)

x <- data.matrix(train_featured_baked %>% select(-default))
y <- train_featured_baked %>% select(default) %>% mutate(default = as.numeric(as.character(default))) %>% pull

x_test <- data.matrix(test_featured_baked %>% select(-default))
y_test <- test_featured_baked %>% select(default) %>% mutate(default = as.numeric(as.character(default))) %>% pull


dtrain <- lgb.Dataset(data = x, label = y)
dtest <- lgb.Dataset(data = x_test, label = y_test)

temp <- lgb.cv(params = params,
               data = dtrain,
               nfold = 5L,
               eval = c('binary_error', 'binary_logloss', 'auc'))

# CV results ---- 

imap_dfr(temp$record_evals$valid, ~tibble(metric = .y, value = .x)) %>% 
  unnest_wider(value) %>% 
  slice(1, 3, 5) %>% 
  pivot_longer(cols = -metric) %>% 
  select(-name) %>% 
  pivot_wider(names_from = metric, values_from = value) %>% 
  unnest(cols = everything()) %>% 
  arrange(binary_logloss) %>%
  colMeans


lgbm_best <- lgb.train(params = params,
                       data = dtrain,
                       valids = list(test = dtest),
                       eval = c('binary_error', 'binary_logloss'))

lgbm_pred <- predict(lgbm_best, x_test)
lgbm_pred <- tibble(p1 = lgbm_pred) %>% 
  mutate(predict = factor(ifelse(p1 > 0.5, 1, 0), levels = c(1, 0)),
         default = factor(y_test, levels = c(1, 0)))



caret::confusionMatrix(lgbm_pred$predict, lgbm_pred$default, mode = 'everything')

lgbm_pred <- get_optimal_predictions(lgbm_pred)

lgbm_metrics <- bind_rows(bind_cols(bind_rows(accuracy(lgbm_pred, default, predict),
                                              f_meas(lgbm_pred, predict, default),
                                              recall(lgbm_pred, predict, default),
                                              precision(lgbm_pred, predict, default)),
                                    model = 'Lgbm base',
                                    threshold = 0.5),
                          bind_cols(bind_rows(accuracy(lgbm_pred, p_optimal, default),
                                              f_meas(lgbm_pred, p_optimal, default),
                                              recall(lgbm_pred, p_optimal, default),
                                              precision(lgbm_pred, p_optimal, default)),
                                    model = 'Lgbm base',
                                    threshold = unique(lgbm_pred$optimal_ts)))


lgbm_pred %>% 
  accuracy(predict, default)

lgbm_pred %>% 
  f_meas(predict, default)

lgbm_pred %>% 
  recall(default, predict)

lgbm_pred %>% 
  accuracy(default, p_optimal)

lgbm_pred %>% 
  f_meas(default, p_optimal)

lgbm_pred %>% 
  recall(default, p_optimal)

# Tuning ----
set.seed(11)
grid <- grid_random(parameters(finalize(mtry(), train_featured_baked),
                               trees(),
                               min_n(),
                               tree_depth(),
                               learn_rate(),
                               loss_reduction(),
                               sample_prop()),
                    size = 1000)

params <- as_tibble(expand.grid(learning_rate = c(0.1, 0.3, 0.5, 0.01, 0.03, 0.05, 0.001, 0.003, 0.005),
                                num_iterations = seq(500, 1000, 100),
                                num_leaves = unique(grid$min_n),
                                max_depth = unique(grid$tree_depth),
                                bagging_fraction = seq(0.6, 1, 0.1),
                                feature_fraction = seq(0.6, 1, 0.1),
                                lamda_l1 = c(0, 1),
                                lambda_l2 = c(0, 1),
                                early_stopping_round = 5))

set.seed(11)
params <- params %>% slice_sample(n = 3)

set.seed(11)
temp <- pmap_dfr(params, ~tibble(models = list(lgb.cv(params = list(learning_rate = ..1,
                                                                    num_iterations = ..2,
                                                                    num_leaves = ..3,
                                                                    max_depth = ..4,
                                                                    bagging_fraction = ..5,
                                                                    feature_fraction = ..6,
                                                                    lamda_l1 = ..7,
                                                                    lambda_l2 = ..8,
                                                                    early_stopping_round = ..9),
                                                      obj = 'binary',
                                                      data = dtrain,
                                                      nfold = 5L,
                                                      valids = list(test = dtest),
                                                      eval = c('binary_logloss', 'auc'))),
                                 learning_rate = ..1,
                                 num_iterations = ..2,
                                 num_leaves = ..3,
                                 max_depth = ..4,
                                 bagging_fraction = ..5,
                                 feature_fraction = ..6,
                                 lamda_l1 = ..7,
                                 lambda_l2 = ..8,
                                 early_stopping_round = ..9))


temp %>% view
temp$models[[1]]$record_evals$valid %>% as_data_frame() %>% 
  slice(1) %>% 
  unnest(cols = everything()) %>% 
  unnest(cols = everything())

temp$models %>% 
  map_dfr(~print(.$best_score))

imap(temp, ~map_dfr(., ~tibble(binary = ..1$record_evals$valid$binary_logloss,
                               auc = ..1$record_evals$valid$auc)))
# s%>% 
#       unnest(cols = everything()) %>% 
#       unnest(cols = everything()))

map(temp, ~.$best_score)
map_dfr(temp, ~ tibble(best_score = .$best_score))