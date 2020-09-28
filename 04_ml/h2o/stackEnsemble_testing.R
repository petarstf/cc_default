library(janitor)
library(DMwR)
library(tidymodels)
library(tidyverse)
library(furrr)
library(h2o)

# Load data ----

source('01_functions/load_data.R')
source('01_functions/train_grid.R')

temp <- data %>% 
  select(contains('pay'), -contains('amt'))

data %>% 
  select(contains('pay'), -contains('amt'))

as_tibble((temp*(temp > 0)) %>% rowSums)

data_mod <- data_cleaned %>% 
  mutate(months_not_paid = (temp > 0) %>% rowSums,
         months_not_paid_sum = (temp*(temp > 0)) %>% rowSums,
         months_not_paid_sum_cat = cut(months_not_paid_sum, 
                                       breaks = c(-Inf, 1, 10, +Inf), 
                                       labels = c(0, 1, 2))) %>% 
  select(months_not_paid, months_not_paid_sum, months_not_paid_sum_cat, everything())

data_mod <- data_mod %>% 
  bind_cols(future_map2_dfr(data %>% select(contains('bill')),
                            data %>% select(contains('pay_amt')),
                            ~.x - (.x - .y)) %>%
              setNames(c('use_amt1', 'use_amt2', 'use_amt3', 'use_amt4', 'use_amt5', 'use_amt6')))

data_mod <- data_mod %>% 
  bind_cols(future_map2_dfr(data_mod %>% select(contains('use_amt')),
                            data %>% select(limit_bal),
                            ~.x/.y) %>% 
              setNames(c('use_rate1', 'use_rate2', 'use_rate3', 'use_rate4', 'use_rate5', 'use_rate6')))

# data_mod %>%
#   mutate(rem_amt1 = bill_amt1 - (bill_amt1 - pay_amt1),
#          rem_amt2 = bill_amt2 - (bill_amt2 - pay_amt2),
#          rem_amt3 = bill_amt3 - (bill_amt3 - pay_amt3),
#          rem_amt4 = bill_amt4 - (bill_amt4 - pay_amt4),
#          rem_amt5 = bill_amt5 - (bill_amt5 - pay_amt5),
#          rem_amt6 = bill_amt6 - (bill_amt6 - pay_amt6),
#          rem_rate1 = rem_amt1 / limit_bal,
#          rem_rate2 = rem_amt2 / limit_bal,
#          rem_rate3 = rem_amt3 / limit_bal,
#          rem_rate4 = rem_amt4 / limit_bal,
#          rem_rate5 = rem_amt5 / limit_bal,
#          rem_rate6 = rem_amt6 / limit_bal)

# data_mod <- data_mod %>%
#   mutate_if(is.factor, function(x) {
#     as.numeric(as.character(x))
#   }) %>%
#   mutate(default = factor(default, levels = c(1, 0)))




# Train/test split ----

split <- initial_split(data_mod, prop = 0.8, strata = default)
train_data <- split %>% training()
test_data <- split %>% testing()

smote_train_featured <- as_tibble(SMOTE(default ~ ., as.data.frame(train_data)))

rec <- recipe(default ~ ., train_data) %>% 
  step_rm(id) %>% 
  prep()

rec2 <- recipe(default ~ ., train_data) %>% 
  step_rm(id) %>% 
  step_pca(all_predictors(), threshold = 0.99) %>% 
  prep()

rec3 <- recipe(default ~ ., train_data) %>% 
  step_rm(id) %>% 
  step_pca(all_predictors(), threshold = 0.95) %>% 
  prep()

rec4 <- recipe(default ~ ., train_data) %>% 
  step_rm(id) %>% 
  step_pca(all_predictors(), threshold = 0.9) %>% 
  prep() 

rec_chinese <- recipe(default ~ ., train_data) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_pca(all_predictors(), threshold = 0.9) %>% 
  prep()
  


train_baked <- bake(rec, train_data)
train_baked2 <- bake(rec2, train_data)
train_baked3 <- bake(rec3, train_data)
train_baked4 <- bake(rec4, train_data)
train_chinese <- bake(rec_chinese, train_data)

smote_baked <- bake(rec, smote_train_featured)
smote_baked2 <- bake(rec2, smote_train_featured)
smote_baked3 <- bake(rec3, smote_train_featured)
smote_baked4 <- bake(rec4, smote_train_featured)

test_baked <- bake(rec, test_data)
test_baked2 <- bake(rec2, test_data)
test_baked3 <- bake(rec3, test_data)
test_baked4 <- bake(rec4, test_data)
test_chinese <- bake(rec_chinese, test_data)

# Automl ----

h2o.init()
h2o.removeAll()

# am <- h2o.automl(y = 'default',
#                  training_frame = as.h2o(train_baked),
#                  nfolds = 5,
#                  stopping_metric = 'logloss',
#                  stopping_rounds = 5,
#                  max_runtime_secs = 30,
#                  seed = 11)
# 
# am@leaderboard
# 
# am@leader
# 
# 
# top_gbm <- h2o.getModel('GBM_grid__1_AutoML_20200821_200004_model_1')
# 
# top_gbm@allparameters
# h2o.varimp_plot(top_gbm)
# 
# h2o.performance(am@leader, as.h2o(test_baked))


# gbm ----

h2o.init()
h2o.removeAll()

gbm_params <- grid_regular(parameters(finalize(mtry(), train_baked),
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
               fold_assignment = c('Random'))

search_criteria = list(strategy = 'RandomDiscrete',
                       stopping_metric = 'AUC',
                       stopping_rounds = 5,
                       max_models = 10,
                       seed = 11)

# GBM grids ----

gbm_grid <- train_grid(algorithm = 'gbm',
                       data = train_baked,
                       grid_id = 'gbm_grid',
                       params = params,
                       search_criteria = search_criteria)

gbm_99_grid <- train_grid(algorithm = 'gbm', 
                          data = train_baked2, 
                          grid_id = 'gbm_99_grid',
                          params = params,
                          search_criteria = search_criteria)

gbm_95_grid <- train_grid(algorithm = 'gbm',
                          data = train_baked3,
                          grid_id = 'gbm_95_grid',
                          params = params,
                          search_criteria = search_criteria)

gbm_90_grid <- train_grid(algorithm = 'gbm',
                          data = train_baked4,
                          grid_id = 'gbm_90_grid',
                          params = params,
                          search_criteria = search_criteria)

gbm_90_chinese_grid <- train_grid(algorithm = 'gbm',
                          data = train_chinese,
                          grid_id = 'gbm_90_chinese_grid',
                          params = params,
                          search_criteria = search_criteria)
# SMOTE grids ----

gbm_smote_grid <- train_grid(algorithm = 'gbm',
                             data = smote_baked,
                             grid_id = 'gbm_smote_grid',
                             params = params,
                             search_criteria = search_criteria)

gbm_99_smote_grid <- train_grid(algorithm = 'gbm', 
                                data = smote_baked2, 
                                grid_id = 'gbm_99_smote_grid',
                                params = params,
                                search_criteria = search_criteria)

gbm_95_smote_grid <- train_grid(algorithm = 'gbm', 
                                data = smote_baked3, 
                                grid_id = 'gbm_95_smote_grid',
                                params = params,
                                search_criteria = search_criteria)

gbm_90_smote_grid <- train_grid(algorithm = 'gbm', 
                                data = smote_baked4, 
                                grid_id = 'gbm_90_smote_grid',
                                params = params,
                                search_criteria = search_criteria)


# gbm_99_grid <- h2o.loadGrid('grids/gbm_99_grid/gbm_99_grid')
# gbm_95_grid <- h2o.loadGrid('grids/gbm_95_grid/gbm_95_grid')
# gbm_90_grid <- h2o.loadGrid('grids/gbm_90_grid/gbm_90_grid')


# Stacking ----

stack <- h2o.stackedEnsemble(y = 'default',
                             model_id = 'stack_reg',
                             training_frame = as.h2o(train_baked),
                             base_models = c(gbm_grid, gbm_99_grid, gbm_95_grid, gbm_90_grid),
                             seed = 11)

stack_smote <- h2o.stackedEnsemble(y = 'default',
                                   model_id = 'stack_smote',
                                   training_frame = as.h2o(smote_baked),
                                   base_models = c(gbm_smote_grid,
                                                   gbm_99_smote_grid, 
                                                   gbm_95_smote_grid, 
                                                   gbm_90_smote_grid),
                                   seed = 11)


final_stack <- h2o.stackedEnsemble(y = 'default',
                                   model_id = 'final_stack',
                                   training_frame = as.h2o(train_baked),
                                   base_models = c(stack, stack_smote),
                                   seed = 11)

# Saving custom StackEnsembles
h2o.saveModel(stack, '05_saved_models/')
h2o.saveModel(stack_smote, '05_saved_models/')
h2o.saveModel(final_stack, '05_saved_models/')

h2o.performance(h2o.getModel(gbm_90_chinese_grid@model_ids[[1]]), as.h2o(test_chinese))

h2o.performance(stack, as.h2o(test_baked))
h2o.performance(stack_smote, as.h2o(test_baked))