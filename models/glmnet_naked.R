library(janitor)
library(tidymodels)
library(tidyverse)

# Load Data ----

source('functions/plot_conf_mat.R')
source('functions/get_predictions_parsnip.R')
source('functions/get_optimal_predictions.R')

data <- read_csv('data/UCI_Credit_Card.csv') %>% clean_names %>% 
  rename(default = default_payment_next_month, pay_1 = pay_0) %>% 
  mutate(default = factor(default, levels = c(1, 0)))

split <- initial_split(data, prop = 0.8, strata = default)
train_data <- split %>% training()
test_data <- split %>% testing()

log_mod <- logistic_reg(mode = 'classification') %>% 
  set_engine('glm')

log_fit <- log_mod %>% 
  fit(default ~ ., train_data %>% select(-id))

log_pred <- get_predictions(log_fit, test_data %>% select(-id))

conf_log <- plot_conf_mat(log_pred$predict, log_pred$default, 'Logistic Regression')

log_metrics <- bind_cols(bind_rows(accuracy(log_pred, default, predict),
                                   f_meas(log_pred, default, predict),
                                   precision(log_pred, default, predict),
                                   recall(log_pred, default, predict),
                                   roc_auc(log_pred, default, p1)),
                         model = 'Logistic Regression',
                         threshold = 0.5) %>% 
  select(-.estimator) %>% 
  pivot_wider(names_from = .metric, values_from = .estimate)

log_metrics <- log_metrics %>% 
  map_at(c('accuracy', 'f_meas', 'precision', 'recall', 'roc_auc'), ~round(. * 100, 3)) %>% 
  as_tibble %>% 
  mutate(library = 'Parsnip')

rm(list = (setdiff(ls(), c('conf_log', 'log_metrics'))))

save.image('env/log_data.RData')