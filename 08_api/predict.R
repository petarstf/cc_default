library(janitor)
library(lightgbm)
library(tidyverse)
library(tidymodels)
library(plumber)

column_names <- c("ID",
                  "LIMIT_BAL",
                  "SEX",
                  "EDUCATION",
                  "MARRIAGE",
                  "AGE",
                  "PAY_0",
                  "PAY_2",
                  "PAY_3",
                  "PAY_4",
                  "PAY_5",
                  "PAY_6",
                  "BILL_AMT1",
                  "BILL_AMT2",
                  "BILL_AMT3",
                  "BILL_AMT4",
                  "BILL_AMT5",
                  "BILL_AMT6",
                  "PAY_AMT1",
                  "PAY_AMT2",
                  "PAY_AMT3",
                  "PAY_AMT4",
                  "PAY_AMT5",
                  "PAY_AMT6")

#* Return prediction
#* @post /predict
#* @param data Dataframe containing an observation for prediction
#* @serializer unboxedJSON
#* @response 201 Class prediction
#* @response 400 Error message - string
function(req, res, data = NA) {
  if(all(is.na(data))) {
    res$status <- 400
    msg <- 'Your request did not include a required parameter.'
    res$body <- msg
    list(error = jsonlite::unbox(msg))

  } else if(!setequal(names(data), column_names)) {
      res$status <- 400
      msg <- 'Your request parameter does not have a required format.'
      res$body <- msg
      list(error = jsonlite::unbox(msg),
           names = (names(data)))
      
  } else {
    
    data <- data %>%
      clean_names() %>%
      rename(pay_1 = pay_0) %>% 
      select(-id)
  
    temp <- data %>% select(contains('pay'), -contains('amt'))
  
    data <- data %>%
      mutate_at(vars(contains('pay'), -contains('amt')), ~factor(ifelse(. %in% c(-2, -1, 0), 0, .))) %>%
      mutate(sex = factor(sex),
             education = factor(ifelse(education %in% c(0, 4, 5, 6), 4, education)),
             marriage = factor(ifelse(marriage %in% c(0, 3), 3, marriage)),
             age = factor(ifelse(age <= 30, 0,
                                 ifelse(age <= 40, 1,
                                        ifelse(age <= 50, 2,
                                               ifelse(age <= 60, 3, 4))))),
             months_not_paid = (temp > 0) %>% rowSums,
             months_not_paid_sum = (temp*(temp > 0)) %>% rowSums,
             months_not_paid_sum_cat = cut(months_not_paid_sum,
                                           breaks = c(-Inf, 1, 10, +Inf),
                                           labels = c(0, 1, 2)),
             rem_amt1 = bill_amt1 - pay_amt1,
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
  
    res$status <- 201
    
    tibble(predict_probability = stats::predict(lgbm, data.matrix(data)),
           predict_class = factor(ifelse(predict_probability > 0.5, 1, 0),
                                  levels = c(1, 0)),
           optimal_threshold = 0.232,
           predict_class_optimal = factor(ifelse(predict_probability > optimal_threshold, 1, 0),
                                          levels = c(1, 0)))

  }
    
}
