library(janitor)
library(lightgbm)
library(tidyverse)
library(tidymodels)
library(plumber)

# DB conn ----

dbname <- 'ccdefault'
username <- 'postgres'
host <- 'localhost'
password <- 'root'


# API ----


#* Return prediction
#* @get /predict
#* @serializer unboxedJSON
#* @response 201 Class prediction
#* @response 400 Error message - string
function(req, res) {
  conn <- DBI::dbConnect(RPostgres::Postgres(),
                         dbname = dbname,
                         user = username,
                         host = host,
                         password = password)
  
  data <- tbl(conn, 'clients')
  
  data <- data %>%
    as_tibble %>% 
    clean_names() %>%
    rename(pay_1 = pay_0) %>% 
    filter(is.na(probability))
  
  if(dim(data)[1] > 0) {
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
    
    
    
    
    
    data <- data %>%
      mutate(probability = stats::predict(lgbm, data.matrix(data %>% select(-id, -probability, -default) %>% 
                                                              mutate_if(is.factor, as.character) %>% 
                                                              mutate_if(is.character, as.numeric))),
             default = ifelse(probability > 0.231887956925153, 1L, 0L)) %>% 
      select(id, probability, default)
    
    
    pwalk(data, ~ RPostgres::dbClearResult(RPostgres::dbSendQuery(conn = conn,
                                                                statement = paste('UPDATE clients',
                                                                                  'SET probability =', ..2, 
                                                                                  ', "default" =', ..3,
                                                                                  'WHERE "ID" =', ..1))))
    
    res$status <- 201
    success <- T
    status <- 'success'
    msg <- 'Successfully updated db.'
    res$body <- msg
    
  } else {
    
    res$status <- 200
    success <- F
    status <- 'failed'
    msg <- 'No new clients to predict'
    res$body <- msg
    
  }
  # End connection
  DBI::dbDisconnect(conn = conn)
  
  # Return response
  list(success = jsonlite::unbox(success),
       msg = jsonlite::unbox(msg))
}
