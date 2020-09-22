library(janitor)
library(tidymodels)
library(tidyverse)

# Preload ----

source('functions/load_data.R')

bill_amt_ltz <- data_featured %>% 
  select(limit_bal, contains('bill'), contains('pay'), default, -contains('rate'), contains('rem')) %>% 
  filter(default == 1) %>% 
  rownames_to_column() %>% 
  filter_at(vars(contains('bill')), all_vars(. < 0))

duly_paid_default <- data_featured %>% 
  select(default, contains('pay'), everything()) %>% 
  filter(default == 1) %>% 
  filter_at(vars(contains('pay'), -contains('amt'), -contains('rate')), all_vars(. == 0))

rem_ltz <- data_featured %>% 
  select(default, contains('rem')) %>% 
  filter(default == 1) %>% 
  rownames_to_column() %>% 
  filter_at(vars(contains('rem')), all_vars(. < 0))


data_problems_tbl <- tibble(Name = c('pay',
                                     'pay_amt',
                                     'bill_amt',
                                     'rem_amt'),
                            Description = c('Repayment status ~ \'Month\' 2005 (-0 = pay duly, 1 = payment delay for one month, 2 = payment delay for two months, …, 8 = payment delay for eight months, 9 = payment delay for nine months and above)',
                                            'Amount of previous payment ~ \'Month\' 2005',
                                            'Amount of bill statement ~ \'Month\' 2005',
                                            'bill_amt - pay_amt ~ \'Month\' 2005'))

rm(list = (setdiff(ls(), c('bill_amt_ltz', 'duly_paid_default', 'rem_ltz', 'data_problems_tbl'))))


