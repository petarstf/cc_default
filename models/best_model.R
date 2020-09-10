library(janitor)
library(lightgbm)
library(tidyverse)
library(tidymodels)

# load('env/lgbm_metrics.RData')
source('functions/load_data.R')
source('functions/evalf1.R')

get_lgb_pred <- function(model, test) {
  pred <- tibble(p1 = predict(model, test),
                 predict = factor(ifelse(p1 > 0.5, 1, 0), levels = c(1, 0)),
                 default = test_featured_baked$default)
  get_optimal_predictions(pred)
}


# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

data_featured_baked <- bake(rec, data_featured)
train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)

# val_split <- initial_split(train_featured_baked, prop = 0.8, strata = default) 
# val_train_baked <- val_split %>% training()
# val_test_baked <- val_split %>% testing()

# Train - Test split ----

x_train <- data.matrix(data_featured_baked %>% select(-default))
y_train <- data_featured_baked %>% mutate(default = as.numeric(as.character(default))) %>% pull(default)
x_test <- data.matrix(test_featured_baked %>% select(-default))
y_test <- test_featured_baked %>% mutate(default = as.numeric(as.character(default))) %>% pull(default)

dtrain <- lgb.Dataset(data = x_train, label = y_train)
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

set.seed(11)
lgbm <- lgb.train(params = params,
                  data = dtrain,
                  nrounds = 156,
                  eval = evalf1,
                  force_row_wise = T,
                  verbose = 2)

rm(list = (setdiff(ls(), 'lgbm')))

save.image('env/best_model.RData')

# lgbm_pred <- get_lgb_pred(lgbm, x_test)
# 
# bind_rows(accuracy(lgbm_pred, default, p_optimal),
#           f_meas(lgbm_pred, default, p_optimal),
#           precision(lgbm_pred, default, p_optimal),
#           recall(lgbm_pred, default, p_optimal),
#           roc_auc(lgbm_pred, default, p1))
