library(janitor)
library(lightgbm)
library(tidyverse)
library(tidymodels)
library(plumber)

# Turn on parallel computations
future::plan('multicore')

lgb.load('05_saved_models/lightgbm.txt')

# Start router ----

root <- pr('08_api/predict_db.R') %>% 
  pr_run(host = '0.0.0.0', port = 8000)

