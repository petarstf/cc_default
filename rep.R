library(janitor)
library(tictoc)
library(DataExplorer)
library(tidyverse)
library(tidymodels)
library(doParallel)
library(lightgbm)
library(vip)

source('functions/load_data.R')

registerDoParallel(cores = parallel::detectCores(logical = F))

load('env/h2o_met.RData')
load('env/xgb_met.RData')
load('env/lgbm_met.RData')
load('env/glm_met.RData')
load('env/lgbm_conf.RData')
load('env/xgb_conf.RData')
load('env/xgb_vip.RData')
load('env/glm_vip.RData')

data_featured %>% 
  correlationfunnel::binarize() %>%
  correlationfunnel::correlate(default__1) %>% 
  correlationfunnel::plot_correlation_funnel(interactive = T)

lgbm_metrics <- lgbm_metrics %>% 
  select(-.estimator)

lgbm_metrics <- bind_cols(lgbm_metrics,
          bind_rows(precision(lgbm_pred, default, .pred_class),
                    precision(lgbm_pred, default, .pred_class),
                    precision(lgbm_pred_down, default, .pred_class),
                    precision(lgbm_pred_down, default, .pred_class),
                    precision(lgbm_pred_up, default, .pred_class),
                    precision(lgbm_pred_up, default, .pred_class),
                    precision(lgbm_pred_smote, default, .pred_class),
                    precision(lgbm_pred_smote, default, .pred_class)))

load('env/lgbm_metrics.RData')

lgb.importance(lgbm_best$fit, )

total_metrics <- 
  bind_rows(bind_cols(bind_rows(xgb_metrics,
                                lgbm_metrics,
                                glm_metrics),
                      library = 'Parsnip'),
            bind_cols(h2o_metrics,
                      library = 'H2O')) %>% 
  select(model, library, everything())

lgbm_metrics %>% 
  mutate(precision = f_meas)

total_metrics
# 
# load('env/xgboost_metrics_1.RData')
# 
# 
# 
# load('env/lgbm_metrics.RData')
# save.image('env/glm_vip.RData')
# rm(list = (setdiff(ls(), ls(pattern = 'conf'))))
# rm(list = setdiff(ls(), ls(pattern = 'vip')))




vip_glm <- vip(glm, mapping = aes_string(fill = 'Variable')) +
  labs(title = 'GLM - Regular') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5,
                                  size = 15,
                                  color = '#272727'),
        axis.title = element_text(color = '#272727', size = 13),
        legend.position = 'none',
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 13))


vip_glm3 <- vip(glm3, mapping = aes_string(fill = 'Variable')) +
  labs(title = 'GLM - x3, x7, x9') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5,
                                  size = 15,
                                  color = '#272727'),
        axis.title = element_text(color = '#272727', size = 13),
        legend.position = 'none',
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 13))


vip_xgb_up <- vip(xgb_final_up, mapping = aes_string(fill = 'Variable')) +
  labs(title = 'XGBoost - Upsample') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5,
                                  size = 15,
                                  color = '#272727'),
        axis.title = element_text(color = '#272727', size = 13),
        legend.position = 'none',
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 13))


vip_xgb_smote <- vip(xgb_final_smote, mapping = aes_string(fill = 'Variable')) +
  labs(title = 'XGBoost - SMOTE') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5,
                                  size = 15,
                                  color = '#272727'),
        axis.title = element_text(color = '#272727', size = 13),
        legend.position = 'none',
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 13))

