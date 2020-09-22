library(janitor)
library(tictoc)
library(DataExplorer)
library(tidyverse)
library(tidymodels)
library(doParallel)
library(lightgbm)
library(vip)
library(correlationfunnel)

# Load Data ----

source('01_functions/load_data.R')

round_up <- function(metrics) {
  metrics %>% 
    map_at(c('accuracy', 'f_meas', 'recall', 'precision', 'roc_auc'),
           ~ round(., 5) * 100) %>% 
    as_tibble
}

registerDoParallel(cores = parallel::detectCores(logical = F))

load('env/h2o_met.RData')
load('env/xgb_met.RData')
load('env/lgbm_met.RData')
load('env/lgbm_conf.RData')
load('env/xgb_conf.RData')
load('env/xgb_vip.RData')
load('env/lgbm_vip.RData')
load('env/glm_data.RData')

xgb_metrics <- round_up(xgb_metrics) %>% mutate(threshold = round(threshold, 3))
lgbm_metrics <- round_up(lgbm_metrics) %>% mutate(threshold = round(threshold, 3))
glm_metrics <- round_up(glmnet_metrics) %>% mutate(threshold = round(threshold, 3))
h2o_metrics <- round_up(h2o_metrics) %>% mutate(threshold = round(threshold, 3))

total_metrics <- 
  bind_rows(bind_cols(bind_rows(xgb_metrics,
                                lgbm_metrics,
                                glm_metrics),
                      library = 'Parsnip'),
            bind_cols(h2o_metrics,
                      library = 'H2O')) %>% 
  select(model, library, everything())

# Correlation Plot ----

corr_funnel_plot <- data_featured %>% 
  binarize() %>%
  correlate(default__1) %>% 
  plot_correlation_funnel(interactive = F) +
  theme(axis.text = element_text(size = 12))
  
corr_funnel_plot <- plotly::ggplotly(corr_funnel_plot)

data_featured %>% names

variables_tbl <- tibble(Name = names(data),
                        Description = c('ID of each client',
                                  'Amount of given credit in NT dollars (includes individual and family/supplementary credit',
                                  'Gender (1 = Male, 0 = Female)',
                                  'Education status (1 = Graduate school, 2 = University, 3 = High school, 4 = Others)',
                                  'Marital status (1 = Married, 2 = Single, 3 = Others)',
                                  'Age in years - grouped',
                                  'Repayment status - Septembar 2005 (-0 = pay duly, 1 = payment delay for one month, 2 = payment delay for two months, ..., 8 = payment delay for eight months, 9 = payment delay for nine months and above)',
                                  'Repayment status - Avgust 2005',
                                  'Repayment status - Jul 2005',
                                  'Repayment status - Jun 2005',
                                  'Repayment status - May 2005',
                                  'Repayment status - April 2005',
                                  'Amount of bill statement - Septembar 2005',
                                  'Amount of bill statement - Avgust 2005',
                                  'Amount of bill statement - Jul 2005',
                                  'Amount of bill statement - Jun 2005',
                                  'Amount of bill statement - May 2005',
                                  'Amount of bill statement - April 2005',
                                  'Amount of previous payment - Septembar 2005',
                                  'Amount of previous payment - Avgust 2005',
                                  'Amount of previous payment - Jul 2005',
                                  'Amount of previous payment - Jun 2005',
                                  'Amount of previous payment - May 2005',
                                  'Amount of previous payment - April 2005',
                                  'Default payment (1 = Yes, 0 = No)'))

variables_custom_tbl <- tibble(Name = c(setdiff(names(data_featured), names(data))),
                               Description = c('Months client\'s repayment status was delayed',
                                               'Total of months client\'s repayment status was delayed',
                                               'Total of months client\'s repayment status was delayed - (0 = (-Inf, 1], 1 = (1, 10], 2 = (10, Inf)',
                                               'bill_amt1 - pay_amt1',
                                               'bill_amt2 - pay_amt2',
                                               'bill_amt3 - pay_amt3',
                                               'bill_amt4 - pay_amt4',
                                               'bill_amt5 - pay_amt5',
                                               'bill_amt6 - pay_amt6',
                                               'pay_amt1 / limit_bal',
                                               'pay_amt2 / limit_bal',
                                               'pay_amt3 / limit_bal',
                                               'pay_amt4 / limit_bal',
                                               'pay_amt5 / limit_bal',
                                               'pay_amt6 / limit_bal'))

metrics_tbl <- tibble(Name = c('accuracy', 'recall', 'precision', 'f_meas', 'roc_auc', 'optimal_threshold'),
                      Description = c('Number of correct predictions / Total number of predictions',
                                      'Proportion of the positive predictions made out of all the true positive cases',
                                      'Proportion of the positive predictions that are actually are actually correct',
                                      'Weighted harmonic mean of precision and recall',
                                      'Performance measurement for classification problem at various thresholds settings',
                                      'Threshold cutoff with highest Sensitivity + Specificity (Recall/TPR + TNR)'),
                      Formula = c('Number of correct predictions / Total number of predictions',
                                  'True Positive / (True Positive + False Negative)',
                                  'True Positive / (True Positive + False Positive)',
                                  '2 * Precision * Recall / (Precision + Recall)',
                                  '/',
                                  'max(Recall + True Negative / (True Negative + False Positive)'))

# load('03_env/total_metrics.RData')
# save.image('03_env/total_metrics.RData')

# rm(list = (setdiff(ls(), ls(pattern = 'conf_glm'))))
