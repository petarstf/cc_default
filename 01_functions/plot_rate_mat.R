plot_rate_mat <- function(estimate, truth, title = 'NA') {
  as.data.frame(table(estimate,
                      truth)) %>% 
    group_by(truth) %>% 
    mutate(freq_pct = round(Freq / sum(Freq) * 100, 2)) %>% 
    ungroup() %>% 
    ggplot() +
    geom_tile(aes(estimate, truth, fill = Freq)) +
    geom_label(aes(estimate, truth, label = paste(Freq, paste0('(', freq_pct, '%', ')')))) +
    labs(title = title,
         x = 'Prediction',
         y = 'Actual') +
    theme_bw() +
    theme(legend.position = 'None')
}
