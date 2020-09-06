library(janitor)
library(tictoc)
library(DataExplorer)
library(tidyverse)
library(tidymodels)
library(doParallel)
library(lightgbm)
library(vip)

# Load data ----

source('functions/load_data.R')
source('functions/get_optimal_predictions.R')
source('functions/evalf1.R')

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)

# val_split <- initial_split(train_featured_baked, prop = 0.8, strata = default) 
# val_train_baked <- val_split %>% training()
# val_test_baked <- val_split %>% testing()

# Train - Test split ----

x_train <- data.matrix(train_featured_baked %>% select(-default))
y_train <- train_featured_baked %>% mutate(default = as.numeric(as.character(default))) %>% pull(default)
x_test <- data.matrix(test_featured_baked %>% select(-default))
y_test <- test_featured_baked %>% mutate(default = as.numeric(as.character(default))) %>% pull(default)

dtrain <- lgb.Dataset(data = x_train, label = y_train)
dtest <- lgb.Dataset(data = x_test, label = y_test)

dtest <- lgb.Dataset.create.valid(dtrain, data = x_test, label = y_test)

# Model ----

params <- list(objective = 'binary',
               learning_rate = 0.05,
               feature_fraction = 1,
               feature_fraction_seed = 11,
               num_iterations = 344L,
               min_data_in_leaf = 13L,
               max_depth = 4L,
               bagging_fraction = 0.152577040181495,
               bagging_seed = 11,
               lambda_l1 = 0.9,
               num_leaves = 255L)

set.seed(11)
lgbm_cv <- lgb.cv(params = params,
                  data = dtrain,
                  nfold = 5,
                  eval = evalf1,
                  early_stopping_rounds = 50L,
                  verbose = 1)

cat(paste('Max F1_score:', max(as_vector(lgbm_cv$record_evals$valid$F1_score$eval))),
    paste('Max F1_score iteration: ', which.max(as_vector(lgbm_cv$record_evals$valid$F1_score$eval))),
    sep = '\n')



best_params <- list(objective = 'binary',
                    learning_rate = 0.05,
                    feature_fraction = 1,
                    feature_fraction_seed = 11,
                    num_iterations = 344L,
                    min_data_in_leaf = 13L,
                    max_depth = 4L,
                    bagging_fraction = 0.152577040181495,
                    bagging_seed = 11,
                    num_leaves = 255L)

valids <- list(test = dtest, train = dtrain)

set.seed(11)
lgbm <- lgb.train(params = params,
                  data = dtrain,
                  nrounds = 555L,
                  valids = list(test = dtrain),
                  eval = evalf1,
                  early_stopping_rounds = 150L,
                  force_row_wise = T,
                  verbose = 2)

lgb.plot.importance(lgb.importance(lgbm))

lgbm_preds <- tibble(p1 = predict(lgbm, x_test),
                     predict = factor(ifelse(p1 > 0.5, 1, 0), levels = c(1, 0)),
                     default = test_featured_baked$default)

lgbm_preds <- get_optimal_predictions(lgbm_preds)

bind_rows(bind_cols(bind_rows(accuracy(lgbm_preds, default, predict),
                              f_meas(lgbm_preds, default, predict),
                              precision(lgbm_preds, default, predict),
                              recall(lgbm_preds, default, predict),
                              roc_auc(lgbm_preds, default, p1)),
                    model = 'LightGBM - Regular'),
          bind_cols(bind_rows(accuracy(lgbm_preds, default, p_optimal),
                              f_meas(lgbm_preds, default, p_optimal),
                              precision(lgbm_preds, default, p_optimal),
                              recall(lgbm_preds, default, p_optimal),
                              roc_auc(lgbm_preds, default, p1)),
                    model = 'LightGBM - Regular - Threshold'))

