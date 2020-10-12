get_predictions <- function(model, data) {
  bind_cols(predict(model, data),
            predict(model, data, type = 'prob'),
            default = data$default) %>% 
    rename(predict = .pred_class, p1 = .pred_1, p0 = .pred_0) %>% 
    get_optimal_predictions()
}
