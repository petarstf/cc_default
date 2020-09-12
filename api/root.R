library(janitor)
library(lightgbm)
library(tidyverse)
library(tidymodels)
library(plumber)

# Load prerequisites ----

source('functions/get_predictions_parsnip.R')
source('functions/get_optimal_predictions.R')
load('env/columns.RData')
lgbm <- lgb.load('saved_models/lightgbm')

# Start router ----

root <- pr('api/plumber_api.R') %>% 
  pr_run(port = 8000)

