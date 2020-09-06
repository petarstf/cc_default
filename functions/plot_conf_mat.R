plot_conf_mat <- function(estimate, truth, title) {
  as.data.frame(table(estimate,
                      truth)) %>% 
    ggplot() +
    geom_tile(aes(estimate, truth, fill = Freq)) +
    geom_label(aes(estimate, truth, label = Freq)) +
    labs(title = title,
         x = 'Prediction',
         y = 'Actual') +
    theme_bw()
}
