library(janitor)
library(DMwR)
library(tidymodels)
library(h2o)
library(tidyverse)

# Load data ----

source('functions/load_data.R')

# Functions ----

source('functions/train_automl.R')
source('functions/train_grid.R')


# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

rec2 <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_normalize(all_numeric()) %>% 
  prep()

rec3 <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_nzv(all_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  prep()

rec4 <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_nzv(all_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  prep()


rec5 <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  prep()

rec6 <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  prep()



train_baked <- bake(rec, train_data)
test_baked <- bake(rec, test_data)
train_baked2 <- bake(rec2, train_data)
test_baked2 <- bake(rec2, test_data)
train_baked3 <- bake(rec3, train_data)
test_baked3 <- bake(rec3, test_data)
train_baked4 <- bake(rec4, train_data)
test_baked4 <- bake(rec4, test_data)
train_baked5 <- bake(rec5, train_data)
test_baked5 <- bake(rec5, test_data)
train_baked6 <- bake(rec6, train_data)
test_baked6 <- bake(rec6, test_data)

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)
train_featured_baked2 <- bake(rec2, train_featured)
test_featured_baked2 <- bake(rec2, test_featured)
train_featured_baked3 <- bake(rec3, train_featured)
test_featured_baked3 <- bake(rec3, test_featured)
train_featured_baked4 <- bake(rec4, train_featured)
test_featured_baked4 <- bake(rec4, test_featured)
train_featured_baked5 <- bake(rec5, train_featured)
test_featured_baked5 <- bake(rec5, test_featured)
train_featured_baked6 <- bake(rec6, train_featured)
test_featured_baked6 <- bake(rec6, test_featured)

# Automl ----

h2o.init()
h2o.removeAll()

# 
# a1 <- train_automl('a1', train_baked)
# a2 <- train_automl('a2', train_baked2)
# a3 <- train_automl('a3', train_baked3)
# a4 <- train_automl('a4', train_baked4)
# a5 <- train_automl('a5', train_baked5)
# a6 <- train_automl('a6', train_baked6)
# 
# 
# a1@leader
# a2@leader
# a3@leader
# a4@leader
# a5@leader
# a6@leader
# 
# 
# f1 <- train_automl('f1', train_featured_baked)
# f2 <- train_automl('f2', train_featured_baked2)
# f3 <- train_automl('f3', train_featured_baked3)
# f4 <- train_automl('f4', train_featured_baked4)
# f5 <- train_automl('f5', train_featured_baked5)
# f6 <- train_automl('f6', train_featured_baked6)
# 
# 
# f1@leader
# f2@leader
# f3@leader
# f4@leader
# f5@leader
# f6@leader
# 
# 
# h2o.performance(a1@leader, as.h2o(test_baked))
# h2o.performance(f1@leader, as.h2o(test_featured_baked))


# GBM ----

h2o.init()
h2o.removeAll()

gbm_params <- grid_regular(parameters(finalize(mtry(), train_featured_baked),
                                      min_n(),
                                      tree_depth(),
                                      trees()),
                           levels = 30)

params <- list(learn_rate = c(1e-3, 3e-3, 5e-3, 1e-4, 3e-4, 5e-4, 1e-5, 3e-5, 5e-5),
               learn_rate_annealing = c(0.95, 0.99, 1),
               min_rows = unique(gbm_params$min_n),
               max_depth = unique(gbm_params$tree_depth),
               ntrees = c(200, 500, 1000),
               sample_rate = seq(0.6, 1, 0.05),
               col_sample_rate = seq(0.6, 1, 0.05),
               fold_assignment = c('Random', 'Stratified'))

search_criteria = list(strategy = 'RandomDiscrete',
                       stopping_metric = 'logloss',
                       stopping_rounds = 5,
                       max_models = 10,
                       seed = 11)


train_grid(algorithm = 'gbm', 
           data = train_featured_baked,
           grid_id = 'gbm_grid_featured',
           params = params,
           search_criteria = search_criteria)

gbm_grid <- h2o.loadGrid('grids/gbm_grid_featured/gbm_grid_featured')
top_gbm <- h2o.getModel(gbm_grid@model_ids[[1]])
h2o.performance(top_gbm, as.h2o(test_featured_baked))

h2o.varimp_plot(top_gbm)


# Sampling ----

rec_down <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  themis::step_downsample(default, skip = F) %>% 
  prep()

rec_up <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  themis::step_upsample(default, skip = F) %>% 
  prep()

downsample_featured_baked <- bake(rec_down, train_featured)
upsample_featured_baked <- bake(rec_up, train_featured)

# H2o ----

h2o.init()
h2o.remove()

train_grid(algorithm = 'gbm', 
           data = downsample_featured_baked,
           grid_id = 'gbm_grid_downsample',
           params = params,
           search_criteria = search_criteria)

gbm_grid_downsample <- h2o.loadGrid('grids/gbm_grid_downsample/gbm_grid_downsample')
top_gbm_down <- h2o.getModel(gbm_grid_downsample@model_ids[[1]])

train_grid(algorithm = 'gbm', 
           data = upsample_featured_baked,
           grid_id = 'gbm_grid_upsample',
           params = params,
           search_criteria = search_criteria)

gbm_grid_upsample <- h2o.loadGrid('grids/gbm_grid_upsample/gbm_grid_upsample')
top_gbm_up <- h2o.getModel(gbm_grid_upsample@model_ids[[1]])

# SMOTE ----

smote_featured <- SMOTE(default ~ ., as.data.frame(train_featured_baked))

train_grid(algorithm = 'gbm', 
           data = smote_featured,
           grid_id = 'gbm_grid_smote',
           params = params,
           search_criteria = search_criteria)

gbm_grid_smote <- h2o.loadGrid('grids/gbm_grid_smote/gbm_grid_smote')
top_gbm_smote <- h2o.getModel(gbm_grid_smote@model_ids[[1]])
h2o.performance(top_gbm_smote, as.h2o(test_featured_baked))

gbm_pred <- as_tibble(h2o.predict(top_gbm_smote, as.h2o(test_featured_baked))) %>% 
  mutate(default = test_featured_baked$default,
         predict = factor(predict, levels = c(1, 0))) 

f_meas(gbm_pred, default, predict) 
