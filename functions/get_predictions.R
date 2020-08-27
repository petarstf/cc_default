get_predictions <- function(model, data) {
  as_tibble(h2o.predict(model, as.h2o(data))) %>% 
    bind_cols(test_featured_baked %>% select(default)) %>% 
    mutate(predict = factor(predict, levels = c(1, 0)))
}