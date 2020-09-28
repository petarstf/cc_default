library(janitor)
library(tidymodels)
library(h2o)
library(tidyverse)
library(doParallel)

registerDoParallel(cores = parallel::detectCores(logical = F))

# Load data ----

source('01_functions/load_data.R')


# Functions ----

source('01_functions/train_grid.R')
source('01_functions/get_optimal_predictions.R')

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)



# Logit ----

h2o.init()
h2o.removeAll()


grid <- grid_random(parameters(mixture(),
                               penalty()),
                    size = 2000)

params <- list(alpha = unique(grid$mixture),
               lambda = unique(grid$penalty))

search_criteria <- list(strategy = 'RandomDiscrete',
                        stopping_metric = 'AUC',
                        stopping_rounds = 5,
                        max_models = 10,
                        seed = 11)

train_grid(algorithm = 'glm',
           data = train_featured_baked,
           grid_id = 'glm_grid_featured',
           params = params,
           search_criteria = search_criteria)

log_grid <- h2o.loadGrid('grids/glm_grid_featured/glm_grid_featured')
log_best <- h2o.getModel(log_grid@model_ids[[1]])

h2o.performance(log_best, as.h2o(test_featured_baked))


h2o.varimp_plot(top_glm)
top_glm@allparameters
# Tidymodels ----

log_mod <- logistic_reg(mode = 'classification',
                        mixture = 0.3435143) %>% 
  set_engine('glmnet')

log_mod %>% translate

log_res <- log_mod %>% fit(default ~ ., train_featured_baked)

log_pred <- bind_cols(predict(log_res, test_featured_baked),
          predict(log_res, test_featured_baked, type = 'prob'),
          default = test_featured_baked$default) %>% 
  rename(predict = .pred_class, p1 = .pred_1, p0 = .pred_0)

log_pred <- get_optimal_predictions(log_pred)

log_res

accuracy(log_pred, default, predict)
f_meas(log_pred, default, predict)
recall(log_pred, default, predict)
precision(log_pred, default, predict)
roc_auc(log_pred, default, p1)

accuracy(log_pred, default, p_optimal)
f_meas(log_pred, default, p_optimal)
recall(log_pred, default, p_optimal)
precision(log_pred, default, p_optimal)

vip::vip(log_res)

