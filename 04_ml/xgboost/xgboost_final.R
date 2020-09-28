library(janitor)
library(tidymodels)
library(tidyverse)
library(DMwR)
library(treesnip)
library(doParallel)
library(tictoc)
library(xgboost)
library(furrr)

registerDoParallel(cores = parallel::detectCores(logical = F))

# Load functions ----

source('01_functions/load_data.R')
source('01_functions/get_predictions_parsnip.R')
source('01_functions/get_optimal_predictions.R')
source('01_functions/evalerr_xgb.R')

# Data prep ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)


x_train <- data.matrix(train_featured_baked %>% select(-default))
y_train <- train_featured_baked %>% mutate(default = as.numeric(as.character(default))) %>% pull(default)
x_test <- data.matrix(test_featured_baked %>% select(-default))
y_test <- test_featured_baked %>% mutate(default = as.numeric(as.character(default))) %>% pull(default)

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

set.seed(11)
folds = vfold_cv(data = train_featured_baked, v = 5, strata = default)

# Model ----

best_params <- list(colsample_bytree = 1,
               min_child_weight = 8L,
               max_depth = 3L,
               subsample = 0.885235332860728,
               tree_method = 'hist',
               # eval_metric = 'logloss',
               nthread = 12)

set.seed(11)

unique(grid_regular(min_n(), levels = 100)$min_n)
grid <- grid_regular(parameters(finalize(mtry(), train_featured_baked %>% select(-default)),
                                        trees(),
                                        min_n(),
                                        tree_depth(),
                                        learn_rate(),
                                        loss_reduction(),
                                        sample_prop()), levels = 10)
set.seed(11)
params <- list(colsample_bytree = seq(0, 1, 0.05),
                                nrounds = c(1500, 2000, 2500, 3000),
                                max_depth = seq(3, 14),
                                max_leaves = c(15, 31, 63, 127, 255, 511, 1023, 2047, 4095),
                                subsample = seq(0, 1, 0.05),
                                tree_method = 'hist',
                                min_child_weight = seq(0, 40)) 

set.seed(11)
params_tbl <- as_tibble(expand.grid(colsample_bytree = seq(0, 1, 0.05),
                                    nrounds = c(1500, 2000, 2500, 3000),
                                    max_depth = seq(3, 14),
                                    max_leaves = c(15, 31, 63, 127, 255, 511, 1023, 2047, 4095),
                                    subsample = seq(0, 1, 0.05),
                                    tree_method = 'hist',
                                    min_child_weight = seq(0, 40))) %>% 
  slice_sample(n = 200)

future_pmap_dfr(params_tbl, ~tibble(x = list(xgb.cv(params = list(colsample_bytree = ..1,
                                                            max_depth = ..3,
                                                            max_leaves = ..4,
                                                            subsample = ..5,
                                                            tree_method = ..6,
                                                            min_child_weight = ..7),
                                                    nthread = 1,
                                                    nrounds = ..2,
                                                    data = dtrain,
                                                    feval = evalf1,
                                                    maximize = T,
                                                    early_stopping_rounds = 50L,
                                                    nfold = 5,
                                                    stratified = T,
                                                    verbose = T))))

xgb_cv <- xgb.cv(params = list(colsample_bytree = sample(params$colsample_bytree, 1),
                     nrounds = sample(params$nrounds, 1),
                     max_depth = sample(params$max_depth, 1),
                     max_leaves = sample(params$max_leaves, 1),
                     subsample = sample(params$subsample, 1),
                     tree_method = 'hist',
                     min_child_weight = sample(params$min_child_weight, 1)),
       data = dtrain,
       nrounds = sample(params$nrounds),
       feval = evalf1,
       early_stopping_rounds = 50L,
       maximize = T,
       nfold = 5,
       stratified = T,
       verbose = 2)

xgb_res <- pmap_df(params, ~tibble(cv = xgb.cv(params = list(colsample_bytree = ..1,
                                                  max_depth = ..3,
                                                  max_leaves = ..4,
                                                  subsample = ..5,
                                                  tree_method = ..6,
                                                  min_child_weight = ..7),
                                               nrounds = ..2,
                                               data = dtrain,
                                               early_stopping_rounds = 50L,
                                               feval = evalf1,
                                               maximize = T,
                                               nfold = 5,
                                               stratified = T)))

