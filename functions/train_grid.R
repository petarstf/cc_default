train_grid <- function(algorithm, data, grid_id, params, search_criteria) {
  y = 'default'
  x = setdiff(names(data), y)
  
  grid_dir = paste0('grids/', grid_id)
  
  grid <- h2o.grid(algorithm = algorithm,
                   x = x,
                   y = y,
                   training_frame = as.h2o(data),
                   grid_id = grid_id,
                   hyper_params = params,
                   search_criteria = search_criteria,
                   nfolds = 5,
                   keep_cross_validation_predictions = T,
                   seed = 11)
  
  if(!dir.exists(grid_dir)) {
    dir.create(grid_dir)
  }
  h2o.saveGrid(grid_directory = grid_dir,
               grid_id = grid_id)
  
  grid
}


