library(janitor)

library(lightgbm)
library(tidymodels)
library(tidyverse)

# Load data ----

source('functions/load_data.R')

lgbm <- lgb.load('saved_models/lightgbm')

rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()

train_baked <- bake(rec, train_featured)
test_baked <- bake(rec, test_featured)

X_train <- train_baked %>% select(-default) %>% data.matrix
y_train <- train_baked %>% pull(default) %>% data.matrix %>% as.numeric

X_test <- test_baked %>% select(-default) %>% data.matrix
y_test <- test_baked %>% pull(default) %>% data.matrix %>% as.numeric


dtrain <- lgb.Dataset(data = X_train, label = y_train)

lgbm_pred <- tibble(p1 = predict(lgbm, X_test),
                    predict = factor(ifelse(p1 > 0.5, 1, 0), levels = c(1, 0)),
                    default = test_baked$default,
                    optimal_ts = 0.232,
                    p_optimal = factor(ifelse(p1 > optimal_ts, 1, 0), levels = c(1, 0)))

# bind_rows(accuracy(lgbm_pred, default, predict),
#           f_meas(lgbm_pred, default, predict),
#           precision(lgbm_pred, default, predict),
#           recall(lgbm_pred, default, predict),
#           roc_auc(lgbm_pred, default, p1))

bind_rows(accuracy(lgbm_pred, default, p_optimal),
          f_meas(lgbm_pred, default, p_optimal),
          precision(lgbm_pred, default, p_optimal),
          recall(lgbm_pred, default, p_optimal),
          roc_auc(lgbm_pred, default, p1))


lgbm_pred %>% 
  arrange(desc(p1)) %>% 
  slice_head(n = 5999/2) %>% 
  recall(default, p_optimal)

gain_capture(lgbm_pred, default, p1)

gain_curve(lgbm_pred %>% arrange(desc(p1)), truth = default, p1) %>% 
  autoplot +
  labs(title = 'Gain Chart',
       x = '% of samples',
       y = 'Gain') +
  scale_y_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) +
  scale_x_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 14))

lift_curve(lgbm_pred, default, p1) %>% 
  autoplot +
  labs(title = 'Lift Chart',
       x = '% of samples',
       y = 'Lift') +
  scale_y_continuous(limits = c(0, 5)) +
  scale_x_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 14))

# Custom way to gains 1 ----

gains_custom <- lgbm_pred %>% 
  mutate(default = as.numeric(as.character(default)),
         predict = as.numeric(as.character(predict)),
         p_optimal = as.numeric(as.character(p_optimal))) %>% 
  arrange(desc(p1)) %>% 
  mutate(bucket = ntile(-p1, 10)) %>% 
  group_by(bucket) %>% 
  summarise_at(vars(predict), funs(total = n(),
                                   totalresp = sum(., na.rm = T))) %>% 
  mutate(cumresp = cumsum(totalresp),
         gain = cumresp / sum(totalresp) * 100,
         samples = cumsum(total))

gains_plot <- gains_custom %>% 
  ggplot() +
  geom_line(aes(bucket * 10, gain)) +
  labs(title = 'Gain Chart',
       x = '% of samples',
       y = 'Gain') +
  scale_y_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) +
  scale_x_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 14)) +
  geom_point(aes(bucket * 10, gain)) +
  geom_label(aes(bucket * 10, gain, label = round(gain, 2)), nudge_x = 7.5, nudge_y = -2)

plotly::ggplotly(gains_plot)


# Custom way to gains chart 2 ----


lgbm_pred %>% 
  arrange(desc(p1)) %>% 
  mutate(default = as.numeric(as.character(default)),
         predict = as.numeric(as.character(predict)),
         p_optimal = as.numeric(as.character(p_optimal))) %>% 
  mutate(bucket = ntile(x = p1, n = 10)) %>% 
  group_by(bucket) %>% 
  summarise(number_of_cases = n(), across(default, sum)) %>% 
  arrange(desc(bucket)) %>%
  mutate(cum_responses = cumsum(default),
         prop_events = round(default / sum(default) * 100, 3),
         gain = cumsum(prop_events),
         bucket = rev(bucket))
            

lgbm_pred %>% 
  summarise(across(p1, ))