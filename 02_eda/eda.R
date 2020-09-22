library(DataExplorer)
library(tidyverse)
library(tidymodels)
library(janitor)
library(plotly)
library(patchwork)

# Load Data ----
source('functions/load_data.R')

plot_proportion <- function(column, data = data_featured, title, xlabels = '') {
  column <- enquo(column)
  if(length(xlabels) > 1) {
    data <- data %>% 
      mutate(!!column := factor(!!column, labels = xlabels))
  }
  
  ggplot_pay <- data %>% 
    mutate(default = factor(default, labels = c('Yes', 'No'))) %>% 
    group_by(!!column, default) %>% 
    count(!!column) %>% 
    ungroup() %>% 
    group_by(!!column) %>% 
    mutate(prop = round(n/sum(n) * 100, 2),
           total = sum(n)) %>% 
    ggplot() +
    geom_col(aes(!!column, prop, fill = default, text = paste('Count: ', n,
                                                              '\nTotal count:', total,
                                                              '\nProp: ', prop,
                                                              '\nDefault: ', default)), color = 'black') + 
    labs(title = title,
         y = 'Count (%)',
         fill = 'Default') +
    theme_bw() +
    theme(axis.text = element_text(size = 11))
  ggplotly(ggplot_pay, tooltip = 'text')
}

# Plots ----

plot_class_balance <- data_featured %>% 
  mutate(default = factor(default, labels = c('Yes', 'No'))) %>% 
  count(default) %>% 
  ggplot() +
  geom_col(aes(default, n, fill = default), color = 'black') +
  geom_label(aes(default, n, label = n)) +
  labs(title = 'Class balance',
       y = 'Count') +
  theme_bw() + 
  theme(legend.position = 'None',
        plot.title = element_text(hjust = 0.5))

plot_pay_1 <- plot_proportion(pay_1, title = 'Pay_1 vs Default')
plot_pay_2 <- plot_proportion(pay_2, title = 'Pay_2 vs Default')
plot_pay_3 <- plot_proportion(pay_3, title = 'Pay_3 vs Default')
plot_pay_4 <- plot_proportion(pay_4, title = 'Pay_4 vs Default')
plot_pay_5 <- plot_proportion(pay_5, title = 'Pay_5 vs Default')
plot_pay_6 <- plot_proportion(pay_6, title = 'Pay_6 vs Default')

plot_education <- plot_proportion(education, xlabels = c('Graduate School', 'University', 'High school', 'Other'), title = 'Education vs Default')
plot_marriage <- plot_proportion(marriage, xlabels = c('Married', 'Single', 'Other'), title = 'Marriage vs Default')
plot_sex <- plot_proportion(sex, xlabels = c('Male', 'Female'), title = 'Sex vs Default')
plot_age <- plot_proportion(age, 
                xlabels = c('(20, 30]', '(30, 40]', '(40, 50]', '(50, 60]', '(60, 90]'),
                title = 'Age vs Default')
plot_months_not_paid_sum_cat <- plot_proportion(months_not_paid_sum_cat,
                                                xlabels = c('(-Inf, 1]', '(1,10]', '(10, Inf)'),
                                                title = 'Total of months not paid - Mapped')
plot_months_not_paid_sum <- plot_proportion(months_not_paid_sum, title = 'Total of months not paid')

plot_months_not_paid <- plot_proportion(months_not_paid, 
                xlabels = c(0, 1, 2, 3, 4, 5, 6),
                title = 'Months client was late with payment')

# Test ----

data_featured %>% 
  mutate(default = factor(default, labels = c('Yes', 'No')),
         education = factor(education, labels = c('Graduate school', 'University', 'Highschool', 'Other'))) %>% 
  group_by(education, default) %>% 
  count(education) %>% 
  ungroup() %>% 
  group_by(education) %>% 
  mutate(prop = round(n/sum(n) * 100, 2),
         total = sum(n))
  


# Save Image ----

save.image('env/eda_data.RData')