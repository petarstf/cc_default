library(xgboost)
library(tidymodels)
library(tidyverse)

# Load data ----

source('01_functions/load_data.R')

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)

set.seed(11)
val_split <- initial_split(train_featured_baked, prop = 0.8, strata = default)
train <- val_split %>% training()
val <- val_split %>% testing()

# XGBoost ----

dtrain <- xgb.DMatrix(data = data.matrix(train %>% select(-default)),
                      label = train %>% select(default) %>% mutate(default = as.numeric(default)-1) %>% pull)
dtest <- xgb.DMatrix(data = data.matrix(val %>% select(-default)),
                     label = val %>% select(default) %>% mutate(default = as.numeric(default)-1) %>% pull)

watchlist <- list(train = dtrain, test = dtest)

params <- list(objective = 'binary:logistic',
               eta = c(0.001),
               max_depth = 9)

xgb.cv(params = params,
       data = dtrain,
       nrounds = 20,
       nfold = 5,
       prediction = T,
       metrics = list('auc', 'logloss', 'error'),
       stratified = T,
       early_stopping_rounds = 5)

xgboost <- xgb.train(booster = 'gbtree',
                     data = dtrain, 
                     max_depth=9, 
                     eta=1, 
                     nrounds=200, 
                     watchlist=watchlist,
                     eval_metric = 'error',
                     eval_metric = 'logloss',
                     eval_metric = 'auc',
                     objective = "binary:logistic")














