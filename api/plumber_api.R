library(janitor)
library(lightgbm)
library(tidymodels)
library(tidyverse)
library(plumber)

# Load Data ----

load('env/best_model.RData')
source('functions/get_predictions_parsnip.R')
source('functions/get_optimal_predictions.R')

# API ----

data <- read_csv('data/UCI_Credit_Card.csv') %>% clean_names

input <- data %>% dplyr::slice(1)

#* Return prediction
#* @param input
#* @post /predict
# function(input) {
#   input <- input %>% 
#     clean_names() %>% 
#     rename(default = default_payment_next_month, pay_1 = pay_0)
#   
#   input <- input %>%
#     mutate_at(vars(contains('pay'), -contains('amt')), ~factor(ifelse(. %in% c(-2, -1, 0), 0, .))) %>% 
#     mutate(sex = factor(sex),
#            education = factor(ifelse(education %in% c(0, 4, 5, 6), 4, education)),
#            marriage = factor(ifelse(marriage %in% c(0, 3), 3, marriage)),
#            age = factor(ifelse(age <= 30, 0,
#                                ifelse(age <= 40, 1,
#                                       ifelse(age <= 50, 2,
#                                              ifelse(age <= 60, 3, 4))))),
#            default = factor(default, levels = c(1, 0)))
#   input <- input %>% 
#     mutate(months_not_paid = (temp > 0) %>% rowSums,
#            months_not_paid_sum = (temp*(temp > 0)) %>% rowSums,
#            months_not_paid_sum_cat = cut(months_not_paid_sum, 
#                                          breaks = c(-Inf, 1, 10, +Inf), 
#                                          labels = c(0, 1, 2)))
#   
#   get_predictions(lgbm, input)
# }


get_pred <- function(input) {
  input <- input %>% 
    clean_names() %>% 
    rename(default = default_payment_next_month, pay_1 = pay_0)
  
  input <- input %>%
    mutate_at(vars(contains('pay'), -contains('amt')), ~factor(ifelse(. %in% c(-2, -1, 0), 0, .))) %>% 
    mutate(sex = factor(sex),
           education = factor(ifelse(education %in% c(0, 4, 5, 6), 4, education)),
           marriage = factor(ifelse(marriage %in% c(0, 3), 3, marriage)),
           age = factor(ifelse(age <= 30, 0,
                               ifelse(age <= 40, 1,
                                      ifelse(age <= 50, 2,
                                             ifelse(age <= 60, 3, 4))))),
           default = factor(default, levels = c(1, 0)))
  
  temp <- input %>% select(contains('pay'), -contains('amt'))
  
  print(temp)
  
  input <- input %>% 
    mutate(months_not_paid = (temp > 0) %>% rowSums,
           months_not_paid_sum = (temp*(temp > 0)) %>% rowSums,
           months_not_paid_sum_cat = cut(months_not_paid_sum, 
                                         breaks = c(-Inf, 1, 10, +Inf), 
                                         labels = c(0, 1, 2))) %>% 
    select(starts_with('months'), everything())
  
  input %>% view
  
  input <- input %>%
    mutate(rem_amt1 = bill_amt1 - pay_amt1,
           rem_amt2 = bill_amt2 - pay_amt2,
           rem_amt3 = bill_amt3 - pay_amt3,
           rem_amt4 = bill_amt4 - pay_amt4,
           rem_amt5 = bill_amt5 - pay_amt5,
           rem_amt6 = bill_amt6 - pay_amt6,
           pay_rate1 = pay_amt1 / limit_bal,
           pay_rate2 = pay_amt2 / limit_bal,
           pay_rate3 = pay_amt3 / limit_bal,
           pay_rate4 = pay_amt4 / limit_bal,
           pay_rate5 = pay_amt5 / limit_bal,
           pay_rate6 = pay_amt6 / limit_bal)
  # get_predictions(lgbm, input)
}

input <- get_pred(input = input)
data.matrix(input)
predict(lgbm, data.matrix(input))
get_predictions(lgbm, input)