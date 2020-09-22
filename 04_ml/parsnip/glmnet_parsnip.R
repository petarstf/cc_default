library(janitor)
library(tidymodels)
library(tidyverse)
library(DMwR)
library(doParallel)
library(tictoc)
library(vip)

# Load data ----

source('01_functions/load_data.R')
source('01_functions/get_predictions_parsnip.R')
source('01_functions/get_optimal_predictions.R')


registerDoParallel(cores = parallel::detectCores(logical = F))

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

rec_dummy <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = T) %>% 
  step_nzv(all_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  prep()

rec_dummy_down <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = T) %>% 
  step_nzv(all_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  themis::step_downsample(default, skip = F) %>% 
  prep()

rec_dummy_up <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = T) %>% 
  step_nzv(all_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  themis::step_upsample(default, skip = T) %>% 
  prep()


train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)

train_featured_dummy <- bake(rec_dummy, train_featured)
downsample_featured_dummy <- bake(rec_dummy_down, train_featured)
upsample_featured_dummy <- bake(rec_dummy_up, train_featured)
smote_featured_dummy <- as_tibble(SMOTE(default ~ ., as.data.frame(train_featured_dummy)))

test_featured_dummy <- bake(rec_dummy, test_featured)

folds <- vfold_cv(train_featured_baked, v = 5, strata = default)
folds_dummy <- vfold_cv(train_featured_dummy, v = 5, strata = default)
folds_dummy_down <- vfold_cv(downsample_featured_dummy, v = 5, strata = default)
folds_dummy_up <- vfold_cv(upsample_featured_dummy, v = 5, strata = default)
folds_dummy_smote <- vfold_cv(smote_featured_dummy, v = 5, strata = default)

# Modelling ----

glm_mod <- logistic_reg(mode = 'classification',
                        penalty = tune(),
                        mixture = tune()) %>% 
  set_engine('glmnet')

set.seed(11)
grid <- grid_max_entropy(parameters(penalty(),
                                    mixture()),
                         size = 500)

glm_res <- glm_mod %>% 
  tune_grid(default ~ .,
            resamples = folds,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, roc_auc, precision, recall),
            control = control_grid(verbose = T, save_pred = T))

glm_down_res <- glm_mod %>% 
  tune_grid(default ~ .,
            resamples = folds_dummy_down,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, roc_auc, precision, recall),
            control = control_grid(verbose = T, save_pred = T))

glm_up_res <- glm_mod %>% 
  tune_grid(default ~ .,
            resamples = folds_dummy_up,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, roc_auc, precision, recall),
            control = control_grid(verbose = T, save_pred = T))

glm_smote_res <- glm_mod %>% 
  tune_grid(default ~ .,
            resamples = folds_dummy_smote,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, roc_auc, precision, recall),
            control = control_grid(verbose = T, save_pred = T))

glm <- finalize_model(glm_mod, select_best(glm_res, 'f_meas')) %>% 
  fit(default ~ ., train_featured_dummy)

glm_down <- finalize_model(glm_mod, select_best(glm_down_res, 'f_meas')) %>% 
  fit(default ~ ., downsample_featured_dummy)

glm_up <- finalize_model(glm_mod, select_best(glm_up_res, 'f_meas')) %>% 
  fit(default ~ ., upsample_featured_dummy)

glm_smote <- finalize_model(glm_mod, select_best(glm_smote_res, 'f_meas')) %>% 
  fit(default ~ ., smote_featured_dummy)

glm_pred <- get_predictions(glm, test_featured_dummy)
glm_down_pred <- get_predictions(glm_down, test_featured_dummy)
glm_up_pred <- get_predictions(glm_up, test_featured_dummy)
glm_smote_pred <- get_predictions(glm_smote, test_featured_dummy)

r <- pROC::roc(glm_pred, default, p1)
optimal_ts <- r$thresholds[which.max(r$sensitivities + r$specificities)]

glm_pred <- get_optimal_predictions(glm_pred)
glm_down_pred <- get_optimal_predictions(glm_down_pred)
glm_up_pred <- get_optimal_predictions(glm_up_pred)
glm_smote_pred <- get_optimal_predictions(glm_smote_pred)

metrics <- bind_rows(bind_cols(bind_rows(accuracy(glm_pred, default, predict),
                                         f_meas(glm_pred, default, predict),
                                         precision(glm_pred, default, predict),
                                         recall(glm_pred, default, predict),
                                         roc_auc(glm_pred, default, p1)),
                               model = 'GLMNET',
                               threshold = 0.5), 
                     bind_cols(bind_rows(accuracy(glm_down_pred, default, predict),
                                         f_meas(glm_down_pred, default, predict),
                                         precision(glm_down_pred, default, predict),
                                         recall(glm_down_pred, default, predict),
                                         roc_auc(glm_down_pred, default, p1)),
                               model = 'GLMNET Downsample',
                               threshold = 0.5),
                     bind_cols(bind_rows(accuracy(glm_up_pred, default, predict),
                                         f_meas(glm_up_pred, default, predict),
                                         precision(glm_up_pred, default, predict),
                                         recall(glm_up_pred, default, predict),
                                         roc_auc(glm_up_pred, default, p1)),
                               model = 'GLMNET Upsample',
                               threshold = 0.5),
                     bind_cols(bind_rows(accuracy(glm_smote_pred, default, predict),
                                         f_meas(glm_smote_pred, default, predict),
                                         precision(glm_smote_pred, default, predict),
                                         recall(glm_smote_pred, default, predict),
                                         roc_auc(glm_smote_pred, default, p1)),
                               model = 'GLMNET SMOTE',
                               threshold = 0.5),
                     bind_cols(bind_rows(accuracy(glm_pred, default, p_optimal),
                                         f_meas(glm_pred, default, p_optimal),
                                         precision(glm_pred, default, p_optimal),
                                         recall(glm_pred, default, p_optimal),
                                         roc_auc(glm_pred, default, p1)),
                               model = 'GLMNET - Threshold',
                               threshold = unique(glm_pred$optimal_ts)),
                     bind_cols(bind_rows(accuracy(glm_down_pred, default, p_optimal),
                                         f_meas(glm_down_pred, default, p_optimal),
                                         precision(glm_down_pred, default, p_optimal),
                                         recall(glm_down_pred, default, p_optimal),
                                         roc_auc(glm_down_pred, default, p1)),
                               model = 'GLMNET Downsample - Threshold',
                               threshold = unique(glm_down_pred$optimal_ts)),
                     bind_cols(bind_rows(accuracy(glm_up_pred, default, p_optimal),
                                         f_meas(glm_up_pred, default, p_optimal),
                                         precision(glm_up_pred, default, p_optimal),
                                         recall(glm_up_pred, default, p_optimal),
                                         roc_auc(glm_up_pred, default, p1)),
                               model = 'GLMNET Upsample - Threshold',
                               threshold = unique(glm_up_pred$optimal_ts)),
                     bind_cols(bind_rows(accuracy(glm_smote_pred, default, p_optimal),
                                         f_meas(glm_smote_pred, default, p_optimal),
                                         precision(glm_smote_pred, default, p_optimal),
                                         recall(glm_smote_pred, default, p_optimal),
                                         roc_auc(glm_smote_pred, default, p1)),
                               model = 'GLMNET SMOTE - Threshold',
                               threshold = unique(glm_smote_pred$optimal_ts)))

metrics %>% 
  pivot_wider(names_from = .metric, values_from = .estimate) %>% 
  arrange(desc(f_meas))

# GLM x3 x7 x9 ----

set.seed(11)
grid <- grid_max_entropy(parameters(glm_mod),
                         size = 500)

set.seed(11)  
glm3_res <- glm_mod %>% 
  tune_grid(default ~ sex_X1 + sex_X2 + pay_1_X0 + pay_1_X1 + pay_1_X2 + pay_3_X0 + pay_3_X2,
            resamples = folds_dummy,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, precision, recall, roc_auc),
            control = control_grid(verbose = T, save_pred = T))

set.seed(11)
glm3_down_res <- glm_mod %>% 
  tune_grid(default ~ sex_X1 + sex_X2 + pay_1_X0 + pay_1_X1 + pay_1_X2 + pay_3_X0 + pay_3_X2,
            resamples = folds_dummy_down,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, roc_auc, precision, recall),
            control = control_grid(verbose = T, save_pred = T))
glm3_down_res %>% 
  show_best('f_meas')

set.seed(11)
glm3_up_res <- glm_mod %>% 
  tune_grid(default ~ sex_X1 + sex_X2 + pay_1_X0 + pay_1_X1 + pay_1_X2 + pay_3_X0 + pay_3_X2,
            resamples = folds_dummy_up,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, roc_auc, precision, recall),
            control = control_grid(verbose = T, save_pred = T))

set.seed(11)
glm3_smote_res <- glm_mod %>% 
  tune_grid(default ~ sex_X1 + sex_X2 + pay_1_X0 + pay_1_X1 + pay_1_X2 + pay_3_X0 + pay_3_X2,
            resamples = folds_dummy_smote,
            grid = grid,
            metrics = metric_set(accuracy, f_meas, roc_auc, precision, recall),
            control = control_grid(verbose = T, save_pred = T))

set.seed(11)
glm3 <- finalize_model(glm_mod, select_best(glm3_res, 'f_meas')) %>% 
  fit(default ~ sex_X1 + sex_X2 + pay_1_X0 + pay_1_X1 + pay_1_X2 + pay_3_X0 + pay_3_X2, train_featured_dummy)

set.seed(11)
glm3_down <- finalize_model(glm_mod, select_best(glm3_down_res, 'recall')) %>% 
  fit(default ~ sex_X1 + sex_X2 + pay_1_X0 + pay_1_X1 + pay_1_X2 + pay_3_X0 + pay_3_X2, downsample_featured_dummy)

set.seed(11)
glm3_up <- finalize_model(glm_mod, select_best(glm3_up_res, 'f_meas')) %>% 
  fit(default ~ sex_X1 + sex_X2 + pay_1_X0 + pay_1_X1 + pay_1_X2 + pay_3_X0 + pay_3_X2, upsample_featured_dummy)

set.seed(11)
glm3_smote <- finalize_model(glm_mod, select_best(glm3_smote_res, 'f_meas')) %>% 
  fit(default ~ sex_X1 + sex_X2 + pay_1_X0 + pay_1_X1 + pay_1_X2 + pay_3_X0 + pay_3_X2, smote_featured_dummy)

glm3_pred <- get_predictions(glm3, test_featured_dummy)
glm3_down_pred <- get_predictions(glm3_down, test_featured_dummy)
glm3_up_pred <- get_predictions(glm3_up, test_featured_dummy)
glm3_smote_pred <- get_predictions(glm3_smote, test_featured_dummy)

bind_cols(predict(glm3_down, test_featured_dummy),
          predict(glm3_down, test_featured_dummy, type = 'prob'),
          default = test_featured_dummy$default) %>% 
  rename(predict = .pred_class, p1 = .pred_1, p0 = .pred_0)

r <- pROC::roc(glm3_down_pred, default, p1)
r$thresholds[which.max(r$sensitivities + r$specificities)]

# glm3_pred <- get_optimal_predictions(glm3_pred)
# glm3_down_pred <- get_optimal_predictions(glm3_down_pred)
# glm3_up_pred <- get_optimal_predictions(glm3_up_pred)
# glm3_smote_pred <- get_optimal_predictions(glm3_smote_pred)

# glm3 <- finalize_model(glm_mod, select_best(glm3_res, 'f_meas')) %>% 
#   fit_xy(x = train_featured_dummy %>% select(contains('sex'), contains('pay_1'), contains('pay_3')),
#          y = train_featured_dummy %>% pull(default))
#   # fit(default ~ sex_X1 + sex_X2 + pay_1_X0 + pay_1_X1 + pay_1_X2 + pay_3_X0 + pay_3_X2, train_featured_dummy)

metrics <- bind_rows(metrics,
                     bind_rows(bind_cols(bind_rows(accuracy(glm3_pred, default, predict),
                                                  f_meas(glm3_pred, default, predict),
                                                  precision(glm3_pred, default, predict),
                                                  recall(glm3_pred, default, predict),
                                                  roc_auc(glm3_pred, default, p1)),
                                         model = 'GLMNET3',
                                         threshold = 0.5),
                              bind_cols(bind_rows(accuracy(glm3_pred, default, p_optimal),
                                                  f_meas(glm3_pred, default, p_optimal),
                                                  precision(glm3_pred, default, p_optimal),
                                                  recall(glm3_pred, default, p_optimal),
                                                  roc_auc(glm3_pred, default, p1)),
                                        model = 'GLMNET3 - Threshold',
                                        threshold = unique(glm3_pred$optimal_ts)),
                              bind_cols(bind_rows(accuracy(glm3_down_pred, default, predict),
                                                  f_meas(glm3_down_pred, default, predict),
                                                  precision(glm3_down_pred, default, predict),
                                                  recall(glm3_down_pred, default, predict),
                                                  roc_auc(glm3_down_pred, default, p1)),
                                        model = 'GLMNET3 Downsample',
                                        threshold = 0.5),
                              bind_cols(bind_rows(accuracy(glm3_down_pred, default, p_optimal),
                                                  f_meas(glm3_down_pred, default, p_optimal),
                                                  precision(glm3_down_pred, default, p_optimal),
                                                  recall(glm3_down_pred, default, p_optimal),
                                                  roc_auc(glm3_down_pred, default, p1)),
                                        model = 'GLMNET3 Downsample - Threshold',
                                        threshold = unique(glm3_down_pred$optimal_ts)),
                              bind_cols(bind_rows(accuracy(glm3_up_pred, default, predict),
                                                  f_meas(glm3_up_pred, default, predict),
                                                  precision(glm3_up_pred, default, predict),
                                                  recall(glm3_up_pred, default, predict),
                                                  roc_auc(glm3_up_pred, default, p1)),
                                        model = 'GLMNET3 Upsample',
                                        threshold = 0.5),
                              bind_cols(bind_rows(accuracy(glm3_up_pred, default, p_optimal),
                                                  f_meas(glm3_up_pred, default, p_optimal),
                                                  precision(glm3_up_pred, default, p_optimal),
                                                  recall(glm3_up_pred, default, p_optimal),
                                                  roc_auc(glm3_up_pred, default, p1)),
                                        model = 'GLMNET3 Upsample - Threshold',
                                        threshold = unique(glm3_up_pred$optimal_ts)),
                              bind_cols(bind_rows(accuracy(glm3_smote_pred, default, predict),
                                                  f_meas(glm3_smote_pred, default, predict),
                                                  precision(glm3_smote_pred, default, predict),
                                                  recall(glm3_smote_pred, default, predict),
                                                  roc_auc(glm3_smote_pred, default, p1)),
                                        model = 'GLMNET3 SMOTE',
                                        threshold = 0.5),
                              bind_cols(bind_rows(accuracy(glm3_smote_pred, default, p_optimal),
                                                  f_meas(glm3_smote_pred, default, p_optimal),
                                                  precision(glm3_smote_pred, default, p_optimal),
                                                  recall(glm3_smote_pred, default, p_optimal),
                                                  roc_auc(glm3_smote_pred, default, p1)),
                                        model = 'GLMNET3 SMOTE - Threshold',
                                        threshold = unique(glm3_smote_pred$optimal_ts)))) %>% 
  pivot_wider(names_from = .metric, values_from = .estimate) %>% 
  select(-.estimator)


glmnet_metrics <- glmnet_metrics %>% 
  mutate_at(vars(threshold), ~ifelse(. == -Inf, 0.5, .))

save.image('03_env/glmnet_metrics.RData')

glmnet_metrics <- metrics
rm(list = (setdiff(ls(), ls(pattern = 'glmnet_metrics'))))

save.image('03_env/glm_met.RData')


# Variable Importance Plots ----

glm_vip <- vi(glm) %>% 
  mutate(color = factor(seq(1:53)))

glm_down_vip <- vi(glm_down) %>% 
  left_join(y = glm_vip %>% select(Variable, color), by = 'Variable')
  
glm_up_vip <- vi(glm_up) %>% 
  left_join(y = glm_vip %>% select(Variable, color), by = 'Variable')

glm_smote_vip <- vi(glm_smote) %>% 
  left_join(y = glm_vip %>% select(Variable, color), by = 'Variable')


vip_glm_reg <- vip(glm_vip, mapping = aes(fill = color), aes = list(color = 'black')) +
  labs(title = 'Logistic Regression Regular') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'None',
        axis.title = element_text(size = 13))


vip_glm_down <- vip(glm_down_vip, mapping = aes(fill = color), aes = list(color = 'black')) +
  labs(title = 'Logistic Regression Downsample') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'None',
        axis.title = element_text(size = 13))


vip_glm_up <- vip(glm_up_vip, mapping = aes(fill = color), aes = list(color = 'black')) +
  labs(title = 'Logistic Regression Upsample') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'None',
        axis.title = element_text(size = 13))


vip_glm_smote <- vip(glm_smote_vip, mapping = aes(fill = color), aes = list(color = 'black')) +
  labs(title = 'Logistic Regression SMOTE') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'None',
        axis.title = element_text(size = 13))



glm3_vip <- vi(glm3) %>% 
  left_join(glm_vip %>% select(Variable, color), by = 'Variable')

glm3_down_vip <- vi(glm3_down) %>% 
  left_join(y = glm_vip %>% select(Variable, color), by = 'Variable')

glm3_up_vip <- vi(glm3_up) %>% 
  left_join(y = glm_vip %>% select(Variable, color), by = 'Variable')

glm3_smote_vip <- vi(glm3_smote) %>% 
  left_join(y = glm_vip %>% select(Variable, color), by = 'Variable')

vip_glm3_reg <- vip(glm3_vip, mapping = aes(fill = Variable), aes = list(color = 'black')) +
  labs(title = 'Logistic Regression Regular - X3, X7, X9') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'None',
        axis.title = element_text(size = 13))

vip_glm3_up <- vip(glm3_up_vip, mapping = aes(fill = Variable), aes = list(color = 'black')) +
  labs(title = 'Logistic Regression Downsample - X3, X7, X9') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'None',
        axis.title = element_text(size = 13))

vip_glm3_down <- vip(glm3_down, mapping = aes(fill = Variable), aes = list(color = 'black')) +
  labs(title = 'Logistic Regression Upsample - X3, X7, X9') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'None',
        axis.title = element_text(size = 13))

vip_glm3_smote <- vip(glm3_smote_vip, mapping = aes(fill = Variable), aes = list(color = 'black')) +
  labs(title = 'Logistic Regression SMOTE - X3, X7, X9') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'None',
        axis.title = element_text(size = 13))


rm(list = (setdiff(ls(), c(ls(pattern = 'vip_'), ls(pattern = 'conf_'), 'glmnet_metrics'))))

save.image('03_env/glm_data.RData')
