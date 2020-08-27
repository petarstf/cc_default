library(janitor)
library(h2o)
library(tidymodels)
library(tidyverse)

# Load and prep data ----

source('functions/load_data.R')
source('functions/get_predictions.R')
source('functions/get_optimal_predictions.R')

# Recipes ----

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

rec_up <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  themis::step_upsample(default, skip = F) %>% 
  prep()

train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)

# Load grids ----
h2o.init()
h2o.removeAll()

gbm_grid <- h2o.loadGrid('grids/gbm_grid_featured/gbm_grid_featured')
gbm_grid_downsample <- h2o.loadGrid('grids/gbm_grid_downsample/gbm_grid_downsample')
gbm_grid_upsample <- h2o.loadGrid('grids/gbm_grid_upsample/gbm_grid_upsample')
gbm_grid_smote <- h2o.loadGrid('grids/gbm_grid_smote/gbm_grid_smote')

glm_grid <- h2o.loadGrid('grids/glm_grid_featured/glm_grid_featured')
rf_grid <- h2o.loadGrid('grids/rf_grid_featured/rf_grid_featured')
nb_grid <- h2o.loadGrid('grids/nb_grid_featured/nb_grid_featured')
mlp_grid <- h2o.loadGrid('grids/mlp_grid_featured/mlp_grid_featured')

top_gbm <- h2o.getModel(gbm_grid@model_ids[[1]])
top_gbm_down <- h2o.getModel(gbm_grid_downsample@model_ids[[1]])
top_gbm_up <- h2o.getModel(gbm_grid_upsample@model_ids[[1]])
top_gbm_smote <- h2o.getModel(gbm_grid_smote@model_ids[[1]])

top_glm <- h2o.getModel(glm_grid@model_ids[[1]])
top_rf <- h2o.getModel(rf_grid@model_ids[[1]])
top_nb <- h2o.getModel(nb_grid@model_ids[[1]])
top_mlp <- h2o.getModel(mlp_grid@model_ids[[1]])

top_gbm_pred <- get_predictions(top_gbm, test_featured_baked)
top_gbm_down_pred <- get_predictions(top_gbm_down, test_featured_baked)
top_gbm_up_pred <- get_predictions(top_gbm_up, test_featured_baked)
top_gbm_smote_pred <- get_predictions(top_gbm_smote, test_featured_baked)

top_glm_pred <- get_predictions(top_glm, test_featured_baked)
top_rf_pred <- get_predictions(top_rf, test_featured_baked)
top_nb_pred <- get_predictions(top_nb, test_featured_baked)
top_mlp_pred <- get_predictions(top_mlp, test_featured_baked)

top_gbm_pred <- get_optimal_predictions(top_gbm_pred)
top_gbm_down_pred <- get_optimal_predictions(top_gbm_down_pred)
top_gbm_up_pred <- get_optimal_predictions(top_gbm_up_pred)
top_gbm_smote_pred <- get_optimal_predictions(top_gbm_smote_pred)

top_glm_pred <- get_optimal_predictions(top_glm_pred)
top_rf_pred <- get_optimal_predictions(top_rf_pred)
top_nb_pred <- get_optimal_predictions(top_nb_pred)
top_mlp_pred <- get_optimal_predictions(top_mlp_pred)

# 
# bind_rows(accuracy(top_gbm_pred, default, p_optimal),
#           f_meas(top_gbm_pred, default, p_optimal),
#           recall(top_gbm_pred, default, p_optimal))
# 
# temp <- top_gbm_smote_pred %>% 
#   mutate(ts = 0.262954,
#          ts_pred = factor(ifelse(p1 > ts, 1, 0), levels = c(1, 0)))
# 
# caret::confusionMatrix(top_gbm_smote_pred$p_optimal, 
#                        top_gbm_smote_pred$default,
#                        mode = 'everything')
# 
# h2o.performance(top_gbm, as.h2o(test_featured_baked))
# h2o.performance(top_gbm_smote, as.h2o(test_featured_baked))

metrics <- bind_rows(bind_cols(bind_rows(accuracy(top_gbm_pred, default, p_optimal),
                              f_meas(top_gbm_pred, default, p_optimal),
                              recall(top_gbm_pred, default, p_optimal),
                              precision(top_gbm_pred, default, p_optimal)),
                    model = 'GBM',
                    threshold = unique(top_gbm_pred$optimal_ts)),
          bind_cols(bind_rows(accuracy(top_gbm_pred, default, predict),
                              f_meas(top_gbm_pred, default, predict),
                              recall(top_gbm_pred, default, predict),
                              precision(top_gbm_pred, default, predict)),
                    model = 'GBM',
                    threshold = 0.5),
          bind_cols(bind_rows(accuracy(top_gbm_smote_pred, default, p_optimal),
                              f_meas(top_gbm_smote_pred, default, p_optimal),
                              recall(top_gbm_smote_pred, default, p_optimal),
                              precision(top_gbm_smote_pred, default, p_optimal)),
                    model = 'GBM Smote',
                    threshold = unique(top_gbm_smote_pred$optimal_ts)),
          bind_cols(bind_rows(accuracy(top_gbm_smote_pred, default, predict),
                              f_meas(top_gbm_smote_pred, default, predict),
                              recall(top_gbm_smote_pred, default, predict),
                              precision(top_gbm_smote_pred, default, predict)),
                    model = 'GBM Smote',
                    threshold = 0.5),
          bind_cols(bind_rows(accuracy(top_glm_pred, default, p_optimal),
                              f_meas(top_glm_pred, default, p_optimal),
                              recall(top_glm_pred, default, p_optimal),
                              precision(top_glm_pred, default, p_optimal)),
                    model = 'Glmnet',
                    threshold = unique(top_glm_pred$optimal_ts)),
          bind_cols(bind_rows(accuracy(top_glm_pred, default, predict),
                              f_meas(top_glm_pred, default, predict),
                              recall(top_glm_pred, default, predict),
                              precision(top_glm_pred, default, predict)),
                    model = 'Glmnet',
                    threshold = 0.5),
          bind_cols(bind_rows(accuracy(top_rf_pred, default, p_optimal),
                              f_meas(top_rf_pred, default, p_optimal),
                              recall(top_rf_pred, default, p_optimal),
                              precision(top_rf_pred, default, p_optimal)),
                    model = 'Random Forest',
                    threshold = unique(top_rf_pred$optimal_ts)),
          bind_cols(bind_rows(accuracy(top_rf_pred, default, predict),
                              f_meas(top_rf_pred, default, predict),
                              recall(top_rf_pred, default, predict),
                              precision(top_rf_pred, default, predict)),
                    model = 'Random Forest',
                    threshold = 0.5),
          bind_cols(bind_rows(accuracy(top_nb_pred, default, p_optimal),
                              f_meas(top_nb_pred, default, p_optimal),
                              recall(top_nb_pred, default, p_optimal),
                              precision(top_nb_pred, default, p_optimal)),
                    model = 'Naive Bayes',
                    threshold = unique(top_nb_pred$optimal_ts)),
          bind_cols(bind_rows(accuracy(top_nb_pred, default, predict),
                              f_meas(top_nb_pred, default, predict),
                              recall(top_nb_pred, default, predict),
                              precision(top_nb_pred, default, predict)),
                    model = 'Naive Bayes',
                    threshold = 0.5),
          bind_cols(bind_rows(accuracy(top_mlp_pred, default, p_optimal),
                              f_meas(top_mlp_pred, default, p_optimal),
                              recall(top_mlp_pred, default, p_optimal),
                              precision(top_mlp_pred, default, p_optimal)),
                    model = 'Multi-layer perceptron',
                    threshold = unique(top_mlp_pred$optimal_ts)),
          bind_cols(bind_rows(accuracy(top_mlp_pred, default, predict),
                              f_meas(top_mlp_pred, default, predict),
                              recall(top_mlp_pred, default, predict),
                              precision(top_mlp_pred, default, predict)),
                    model = 'Multi-layer perceptron',
                    threshold = 0.5)) %>% 
  pivot_wider(names_from = .metric, values_from = .estimate) %>% 
  select(-.estimator)

metrics %>% 
  arrange(desc(f_meas))

caret::confusionMatrix(top_rf_pred$p_optimal, top_rf_pred$default, mode='everything')
