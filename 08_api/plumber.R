library(janitor)
library(lightgbm)
library(tidyverse)
library(tidymodels)
library(plumber)

# Load prerequisites ----

load('03_env/columns.RData')
lgbm <- lgb.load('05_saved_models/lightgbm')

# Start router ----

root <- pr('08_api/predict.R') %>% 
  pr_run(host = '0.0.0.0')
