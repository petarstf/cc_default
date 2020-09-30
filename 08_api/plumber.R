library(janitor)
library(lightgbm)
library(tidyverse)
library(tidymodels)
library(plumber)

# Load prerequisites ----

lgbm <- lgb.load('05_saved_models/lightgbm_model')

# Start router ----

root <- pr('08_api/predict_db.R') %>% 
  pr_run(host = '0.0.0.0', port = 8000)
