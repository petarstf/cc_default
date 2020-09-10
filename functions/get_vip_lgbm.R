get_vip_lgbm <- function(model, num_features = 10L, title = 'LGBM') {
  as_tibble(lgb.importance(model)) %>% 
    clean_names() %>% 
    arrange(gain) %>% 
    mutate(feature = factor(feature, levels = unique(feature))) %>% 
    slice_tail(n = num_features) %>% 
    ggplot() +
    geom_bar(aes(y = feature, x = gain, fill = feature), color = 'black', stat = 'identity') +
    labs(title = title,
         x = 'Importance',
         y = '') +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5),
          legend.position = 'None',
          axis.title = element_text(size = 13))
    
}
