library(janitor)

library(lightgbm)
library(tidymodels)
library(tidyverse)
library(plotly)
library(furrr)

# Load data ----

source('01_functions/load_data.R')
source('01_functions/get_optimal_predictions.R')
source('01_functions/get_predictions_parsnip.R')
source('01_functions/plot_conf_mat.R')
source('01_functions/plot_rate_mat.R')

# Load models ----

# lgbm <- lgb.load('saved_models/lightgbm')
load('03_env/lightgbm_pred.RData')
load('03_env/glm_naked_predictions.RData')
load('03_env/glm_smote_predictions.RData')

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_baked <- bake(rec, train_featured)
test_baked <- bake(rec, test_featured)

comparison_tbl <- bind_rows(bind_cols(bind_rows(accuracy(lgbm_pred, default, p_optimal),
                              f_meas(lgbm_pred, default, p_optimal),
                              recall(lgbm_pred, default, p_optimal),
                              precision(lgbm_pred, default, p_optimal),
                              roc_auc(lgbm_pred, default, p1)),
                    model = 'LightGBM - Optimal Threshold',
                    threshold = unique(lgbm_pred$optimal_ts)) %>% 
            select(-.estimator) %>% 
            pivot_wider(names_from = .metric, values_from = .estimate),
          bind_cols(bind_rows(accuracy(glm_smote_pred, default, predict),
                              f_meas(glm_smote_pred, default, predict),
                              recall(glm_smote_pred, default, predict),
                              precision(glm_smote_pred, default, predict),
                              roc_auc(glm_smote_pred, default, p1)),
                    model = 'Logistic Regression SMOTE',
                    threshold = 0.5) %>% 
            select(-.estimator) %>% 
            pivot_wider(names_from = .metric, values_from = .estimate),
          bind_cols(bind_rows(accuracy(log_pred, default, predict),
                              f_meas(log_pred, default, predict),
                              recall(log_pred, default, predict),
                              precision(log_pred, default, predict),
                              roc_auc(log_pred, default, p1)),
                    model = 'Logistic Regression Basic',
                    threshold = 0.5) %>% 
            select(-.estimator) %>% 
            pivot_wider(names_from = .metric, values_from = .estimate)) %>% 
  mutate(across(c(accuracy, f_meas, recall, precision, roc_auc), .fns = ~round(. * 100, 3)))

# LightGBM ----
# 
# 
# X_train <- train_baked %>% select(-default) %>% data.matrix
# y_train <- train_baked %>% pull(default) %>% data.matrix %>% as.numeric
# 
# X_test <- test_baked %>% select(-default) %>% data.matrix
# y_test <- test_baked %>% pull(default) %>% data.matrix %>% as.numeric
# 
# 
# dtrain <- lgb.Dataset(data = X_train, label = y_train)
# 
# lgbm_pred <- tibble(p1 = predict(lgbm, X_test),
#                     predict = factor(ifelse(p1 > 0.5, 1, 0), levels = c(1, 0)),
#                     default = test_baked$default,
#                     optimal_ts = 0.232,
#                     p_optimal = factor(ifelse(p1 > optimal_ts, 1, 0), levels = c(1, 0)))

# bind_rows(accuracy(lgbm_pred, default, predict),
#           f_meas(lgbm_pred, default, predict),
#           precision(lgbm_pred, default, predict),
#           recall(lgbm_pred, default, predict),
#           roc_auc(lgbm_pred, default, p1))

# bind_rows(accuracy(lgbm_pred, default, p_optimal),
#           f_meas(lgbm_pred, default, p_optimal),
#           precision(lgbm_pred, default, p_optimal),
#           recall(lgbm_pred, default, p_optimal),
#           roc_auc(lgbm_pred, default, p1))



# Yardstick gains & lifts ----

# gain_capture(lgbm_pred, default, p1)
# 
# gain_curve(lgbm_pred %>% arrange(desc(p1)), truth = default, p1) %>% 
#   autoplot +
#   labs(title = 'Gain Chart',
#        x = '% of samples',
#        y = 'Gain') +
#   scale_y_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) +
#   scale_x_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) +
#   theme(axis.text = element_text(size = 13),
#         axis.title = element_text(size = 14))



# 
# lift_curve(lgbm_pred, default, p1) %>%
#   autoplot +
#   labs(title = 'Lift Chart',
#        x = '% of samples',
#        y = 'Lift') +
#   scale_y_continuous(limits = c(0, 5)) +
#   scale_x_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) +
#   theme(axis.text = element_text(size = 13),
#         axis.title = element_text(size = 14))

            

# Custom way to gains chart ----

get_gains_lift <- function(predictions, buckets = 10, model_name = NA) {
  data <- predictions %>% 
    arrange(desc(p1)) %>% 
    mutate(default = as.numeric(as.character(default)),
           predict = as.numeric(as.character(predict)),
           p_optimal = as.numeric(as.character(p_optimal)),
           bucket = ntile(x = p1, n = buckets)) %>% 
    group_by(bucket) %>%
    summarise(number_of_cases = n(), across(default, sum)) %>%
    arrange(desc(bucket)) %>%
    mutate(cases_pct = round(cumsum(number_of_cases) / sum(number_of_cases) * 100, 3),
           cum_responses = cumsum(default),
           prop_events = round(default / sum(default) * 100, 3),
           gain = cumsum(prop_events),
           bucket = rev(bucket),
           lift = round(gain / (cumsum(number_of_cases / sum(number_of_cases)) * 100), 3)) %>% 
    ungroup() %>% 
    arrange(bucket)
  
  if(!is.na(model_name)) {
    data <- data %>% 
      mutate(model_name = model_name)
  }
  data
}

gains_lift <- bind_rows(get_gains_lift(lgbm_pred, 
                                       buckets = 16, 
                                       model_name = 'Lightgbm'),
                        get_gains_lift(log_pred, 
                                       buckets = 16, 
                                       model_name = 'Logistic Regression - Basic'),
                        get_gains_lift(glm_smote_pred, 
                                       buckets = 16, 
                                       model_name = 'Logistic Regression - SMOTE')
                        )
bind_rows(
  log_pred %>% f_meas(default, predict),
  glm_smote_pred %>% f_meas(default, predict),
  lgbm_pred %>% f_meas(default, p_optimal)
)

gains_chart <- gains_lift %>% 
  ggplot() +
  geom_line(aes(cases_pct, gain, color = model_name)) +
  geom_point(aes(cases_pct, gain, color = model_name, text = paste('Model:', model_name,
                                                                   '\nSamples:', glue::glue('{cases_pct}%'),
                                                                   '\nGain:', gain,
                                                                   '\nClients:', number_of_cases,
                                                                   '\nDefaulters:', default))) +
  geom_line(aes(cases_pct, cases_pct, color = 'Baseline'), linetype = 'dotted') +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
  scale_color_discrete(name = 'Model') +
  labs(title = 'Gain Chart',
       x = '% of samples',
       y = 'Gain') +
  theme_bw() +
  theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 14),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'bottom')

gains_plot <- plotly::ggplotly(gains_chart, tooltip = 'text')

lift_chart <- gains_lift %>% 
  ggplot() +
  geom_line(aes(x = cases_pct, y = lift, color = model_name)) +
  geom_point(aes(x = cases_pct, y = lift, color = model_name, 
                 text = paste('Model:', model_name,
                              '\nSamples:', paste0(cases_pct, '%'),
                              '\nLift:', lift))) +
  geom_line(aes(x = cases_pct, y = 1, color = 'Baseline'), linetype = 'dotted') +
  scale_y_continuous(limits = c(0, 4.5)) +
  scale_x_continuous(breaks = seq(0, 100, 10)) +
  scale_color_discrete(name = 'Model') +
  labs(title = 'Lift Chart',
       x = '% of samples',
       y = 'Lift') +
  theme_bw() +
  theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 14),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'bottom')

lift_plot <- plotly::ggplotly(lift_chart, tooltip = 'text')



roc_data <- bind_rows(bind_cols(roc_curve(lgbm_pred, default, p1, na_rm = T),
                    model_name = 'Lightgbm'),
          bind_cols(roc_curve(log_pred, default, p1, na_rm = T),
                    model_name = 'Logistic Regression - Basic'),
          bind_cols(roc_curve(glm_smote_pred, default, p1, na_rm = T),
                    model_name = 'Logistic Regression - SMOTE')
          ) %>% 
  mutate(sensitivity = round(sensitivity * 100, 3),
         specificity = round(specificity * 100, 3))

roc_chart <- roc_data %>% 
  ggplot() +
  geom_line(aes(100 - specificity, sensitivity, color = model_name, 
                text = paste('Model:', model_name,
                             '\nSensitivity:', sensitivity,
                             '\nFalse Positive Rate:', 100 - specificity), 
                group = model_name)) +
  scale_color_discrete(name = 'Model') +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
  labs(title = 'ROC',
       x = '1 - Specificity (FPR) (%)',
       y = 'Sensitivity (Recall/TPR) (%)') +
  theme_bw() +
  theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 14),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'bottom')
  

roc_plot <- ggplotly(roc_chart, tooltip = 'text')



# PR_AUC ----

pr_data <- bind_rows(bind_cols(pr_curve(lgbm_pred, default, p1),
                                model_name = 'Lightgbm'),
                      bind_cols(pr_curve(log_pred, default, p1),
                                model_name = 'Logistic Regression - Basic'),
                      bind_cols(pr_curve(glm_smote_pred, default, p1),
                                model_name = 'Logistic Regression - SMOTE')) %>% 
  map_at(c('recall', 'precision'), ~ round(. * 100, 3)) %>% 
  as_tibble()

pr_chart <- pr_data %>% 
  ggplot() +
  geom_line(aes(recall, precision, color = model_name,
                text = paste('Model:', model_name,
                             '\nRecall:', recall,
                             '\nPrecision:', precision), 
                group = model_name)) +
  scale_color_discrete(name = 'Model') +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
  labs(title = 'Precision vs Recall',
       x = 'Recall (%)',
       y = 'Precision (%)') +
  theme_bw() +
  theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 14),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'bottom')

pr_plot <- ggplotly(pr_chart, tooltip = 'text')


# Confusion and Rate matrix plots ----

# lgbm_comp_conf_mat_plot <- plot_conf_mat(lgbm_pred$p_optimal, lgbm_pred$default, 'Lightgbm - Optimal Threshold')
# log_comp_conf_mat_plot <- plot_conf_mat(glm_smote_pred$predict, glm_smote_pred$default, 'Logistic Regression - SMOTE')
# glm_smote_comp_conf_mat_plot <- plot_conf_mat(log_pred$predict, log_pred$default, 'Logistic Regression - Basic')


lgbm_comp_rate_mat_plot <- plot_rate_mat(lgbm_pred$p_optimal, lgbm_pred$default, 'Lightgbm - Optimal Threshold')
glm_smote_comp_rate_mat_plot <- plot_rate_mat(glm_smote_pred$predict, glm_smote_pred$default, 'Logistic Regression - SMOTE')
log_comp_rate_mat_plot <- plot_rate_mat(log_pred$predict, log_pred$default, 'Logistic Regression - Basic')


desc_tbl <- tibble(name = c('Sensitivity / Recall', 
                            'Specificity',
                            'False Positive Rate (FPR)',
                            'Precision',
                            'Gain',
                            'Lift'),
                   desc = c('Tačno prediktovani default-eri / Ukupan broj default-era',
                            'Tačno prediktovani nedefault-eri / Ukupan broj nedefault-era',
                            'Netačno prediktovani default-eri / Ukupan broj default-era',
                            'Tačno prediktovani default-eri / (Tačno + Netačno) Prediktovani default-eri',
                            'Procenat default-era prediktovan u određenoj tački',
                            'Efektivnost modela u odnosu na nasumično pogađanje, proporcija između rezultata dobijenih uz pomoć modela i bez njega'))


# Check balance ----

# TP - Cost: 0, Benefit: 0
# FP - Cost: prevent_cost (1000$), Benefit: limit_bal * 0.1
# TN - Cost: 0, Benefit: limit_bal * 0.1
# FN - Cost: limit_bal, Benefit: 0


cb_tbl <- tibble(type = c('True Positive Rate',
                          'False Positive Rate',
                          'True Negative Rate',
                          'False Negative Rate'),
                 cost = c('Nothing = 0', 
                          'Prevention action cost',
                          'Nothing = 0',
                          'Limit balance'),
                 benefit = c('Nothing = 0',
                             '10% of Limit balance',
                             '10% of Limit balance',
                             'Nothing = 0'))

# EV = p * [tpr * b_tp + fnr * c_fn] + (1 - p) * (tnr * b_tn + fpr * c_fp)

# p1 * (tpr * cb_tp + fnr * cb_fn) + (1 - p) * (tnr * cb_tn + fpr + cb_fp)

get_expected_rates <- function(data, truth, estimate) {
  truth <- enquo(truth)
  estimate <- enquo(estimate)
  roc_curve(data, !!truth, !!estimate) %>% 
    rename(tpr = sensitivity, tnr = specificity) %>% 
    mutate(fpr = 1 - tnr,
           fnr = 1 - tpr) %>% 
    select(.threshold, tpr, fnr, tnr, fpr) %>% 
    rename(threshold = .threshold) %>% 
    filter(!(threshold %in% c(-Inf, Inf)))
}

get_expected_costs <- function(data, predictions, column) {
  col <- enquo(column)
  data %>% 
    mutate(cb_tp = 0,
           cb_fn = -1 * !!col,
           cb_fp = !!col * 0.1 - 1000,
           cb_tn = !!col * 0.1) %>% 
    rownames_to_column() %>% 
    select(rowname, !!col, starts_with('cb')) %>% 
    left_join(predictions %>% rownames_to_column()) %>% 
    select(p1, starts_with('cb'))
}

get_profit <- function(data, predictions, column, truth, estimate, model) {
  column <- enquo(column)
  truth <- enquo(truth)
  estimate <- enquo(estimate)
  
  expected_costs <- get_expected_costs(data, predictions, !!column) %>% 
    mutate(model = model)
  expected_rates <- get_expected_rates(predictions, !!truth, !!estimate)
  
  
  future_pmap_dfr(expected_rates, ~ tibble(profit = expected_costs$p1 *
                                             (..2 * expected_costs$cb_tp + ..3 * expected_costs$cb_fn) +
                                             (1 - expected_costs$p1) *
                                             (..4 * expected_costs$cb_tn + ..5 * expected_costs$cb_fp),
                                           threshold = ..1,
                                           model = expected_costs$model)) %>%
    group_by(threshold, model) %>%
    summarize(total_profit = sum(profit)) %>% 
    ungroup()
}

total_profits_tbl <- bind_rows(get_profit(data = test_baked,
                     predictions = lgbm_pred,
                     column = limit_bal,
                     truth = default,
                     estimate = p1,
                     model = 'LightGBM'),
          get_profit(data = test_baked,
                     predictions = glm_smote_pred,
                     column = limit_bal,
                     truth = default,
                     estimate = p1,
                     model = 'Logistic Regression SMOTE'),
          get_profit(data = test_baked,
                     predictions = log_pred,
                     column = limit_bal,
                     truth = default,
                     estimate = p1,
                     model = 'Logistic Regression Basic'))

# format(object.size(total_profits_tbl), units = 'MB')

max_profit_ts <- total_profits_tbl %>% 
  filter(total_profit == max(total_profit))

unique(lgbm_pred$optimal_ts)

lgbm_profit_ts <- total_profits_tbl %>% 
  filter(model == 'LightGBM', threshold > unique(lgbm_pred$optimal_ts)) %>% 
  slice(1)

glm_smote_profit_ts <- total_profits_tbl %>% 
  filter(model == 'Logistic Regression SMOTE', threshold > unique(glm_smote_pred$optimal_ts)) %>% 
  slice(1)

log_profit_ts <- total_profits_tbl %>% 
  filter(model == 'Logistic Regression Basic', threshold > unique(log_pred$optimal_ts)) %>% 
  slice(1)


ts_profit_chart <- total_profits_tbl %>%
  ggplot() +
  geom_line(aes(threshold,
                total_profit,
                color = model,
                text = paste('Threshold:', threshold,
                             '\nProfit:', round(total_profit, 2),
                             '\nModel:', model)),
            group = 3) +
  geom_vline(aes(xintercept = max_profit_ts$threshold,
                 color = 'Max Profit TS',
                 text = paste('Threshold:', max_profit_ts$threshold,
                              '\nMax Profit:', round(max_profit_ts$total_profit, 2)))) +
  geom_vline(aes(xintercept = lgbm_profit_ts$threshold,
                 color = 'LightGBM',
                 text = paste('Threshold:', lgbm_profit_ts$threshold,
                              '\nProfit:', round(lgbm_profit_ts$total_profit, 2),
                              '\nModel:', model))) +
  geom_vline(aes(xintercept = glm_smote_profit_ts$threshold,
                 color = 'Logistic Regression SMOTE',
                 text = paste('Threshold:', glm_smote_profit_ts$threshold,
                              '\nProfit:', round(glm_smote_profit_ts$total_profit, 2),
                              '\nModel:', model))) +
  geom_vline(aes(xintercept = log_profit_ts$threshold,
                 color = 'Logistic Regression Basic',
                 text = paste('Threshold:', log_profit_ts$threshold,
                              '\nProfit:', round(log_profit_ts$total_profit, 2),
                              '\nModel:', model))) +
  labs(title = 'Expected Profit Curve - Total Expected Profit',
       x = 'Threshold',
       y = 'Expected total profit') +
  theme_bw() +
  theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 14),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'bottom')


threshold_profit_plot <- ggplotly(ts_profit_chart, tooltip = 'text')

rm(list = (setdiff(ls(), c('comparison_tbl',
                           'gains_lift',
                           'desc_tbl',
                           'cb_tbl',
                           'max_profit_ts',
                           'lgbm_profit_ts',
                           'glm_smote_profit_ts',
                           'log_profit_ts',
                           ls(pattern = 'plot')))))
