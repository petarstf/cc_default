get_optimal_predictions <- function(predictions) {
  r <- pROC::roc(predictions, default, p1)
  optimal_ts <- r$thresholds[which.max(r$sensitivities + r$specificities)]
  predictions %>% 
    mutate(optimal_ts = optimal_ts,
           p_optimal = factor(ifelse(p1 > optimal_ts, 1, 0), levels = c(1, 0)))
}