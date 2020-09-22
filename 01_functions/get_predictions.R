get_predictions <- function(model, data) {
  as_tibble(h2o.predict(model, as.h2o(data))) %>% 
    mutate(predict = factor(predict, levels = c(1, 0)),
           predict05 = factor(ifelse(p1 > 0.5, 1, 0), levels = c(1, 0)),
           default = data$default)
}
