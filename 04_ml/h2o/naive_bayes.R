library(janitor)
library(tidymodels)
library(h2o)
library(tidyverse)

# Load data ----

source('01_functions/load_data.R')


# Functions ----

source('01_functions/train_grid.R')

# Recipes ----
rec <- recipe(default ~ ., train_featured) %>% 
  step_rm(id) %>% 
  prep()


train_featured_baked <- bake(rec, train_featured)
test_featured_baked <- bake(rec, test_featured)


# NaiveBayes ----

h2o.init()

h2o.removeAll()

grid <- grid_random(parameters(Laplace()),
                    size = 2000)

params <- list(laplace = unique(grid$Laplace),
               fold_assignment = c('Random', 'Stratified'),
               distribution = 'bernoulli')

search_criteria = list(strategy = 'RandomDiscrete',
                       stopping_metric = 'logloss',
                       stopping_rounds = 5,
                       max_models = 10,
                       seed = 11)

train_grid(algorithm = 'naivebayes', 
           data = train_featured_baked, 
           grid_id = 'nb_grid_featured', 
           params = params, 
           search_criteria = search_criteria)

nb_grid <- h2o.loadGrid('grids/nb_grid_featured/nb_grid_featured')
top_nb <- h2o.getModel(nb_grid@model_ids[[1]])

h2o.performance(top_nb, as.h2o(test_featured_baked))

h2o.varimp_plot(top_nb)
