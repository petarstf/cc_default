evalf1 <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  labels <- factor(labels, levels = c(1, 0))
  preds <- factor(ifelse(preds > 0.5, 1, 0), levels = c(1, 0))
  
  f1 <- f_meas_vec(labels, preds)
  f1_score <- ifelse(is.na(f1), 0, f1)
  
  return(list(name = 'f1_score', 
              value = f1_score, 
              higher_better = T))
}
