library(janitor)
library(lightgbm)
library(tidyverse)
library(tidymodels)
library(plumber)

# Load prerequisites ----

load('env/columns.RData')
lgbm <- lgb.load('saved_models/lightgbm')

# Start router ----

root <- pr('api/plumber_api.R') %>% 
  pr_run(port = 8000)
