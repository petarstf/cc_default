library(janitor)
library(tidyverse)

data <- read_csv('00_data/UCI_Credit_Card.csv') %>% 
  clean_names() %>% 
  rename('default' = 'default_payment_next_month',
         'pay_1' = 'pay_0')

data_cleaned <- data %>%
  mutate_at(vars(contains('pay'), -contains('amt')), ~factor(ifelse(. %in% c(-2, -1, 0), 0, .))) %>% 
  mutate(sex = factor(sex),
         education = factor(ifelse(education %in% c(0, 4, 5, 6), 4, education)),
         marriage = factor(ifelse(marriage %in% c(0, 3), 3, marriage)),
         age = cut(age, breaks = c(20, 30, 40, 50, 60, 80), labels = seq(0, 4)),
         default = factor(default, levels = c(1, 0)))


# Feature engineering ----

data_featured <- data_cleaned %>% 
  mutate(education_sex = factor(str_c(education, sex, sep = '_'), labels = seq(0, 7)),
         sex_marriage = factor(str_c(sex, marriage, sep = '_'), labels = seq(1, 6)),
         education_marriage = factor(str_c(education, marriage, sep = '_'), labels = seq(0, 11)),
         education_marriage_sex = factor(str_c(education, marriage, sex, sep = '_'), labels = seq(0, 23)))



# Feature engineering 2 ----

temp <- data %>% 
  select(contains('pay'), -contains('amt'))

data_featured <- data_cleaned %>% 
  mutate(months_not_paid = (temp > 0) %>% rowSums,
         months_not_paid_sum = (temp*(temp > 0)) %>% rowSums,
         months_not_paid_sum_cat = cut(months_not_paid_sum, 
                                       breaks = c(-Inf, 1, 10, +Inf), 
                                       labels = c(0, 1, 2))) %>% 
  select(months_not_paid, months_not_paid_sum, months_not_paid_sum_cat, everything())


data_featured <- data_featured %>%
  mutate(rem_amt1 = bill_amt1 - pay_amt1,
         rem_amt2 = bill_amt2 - pay_amt2,
         rem_amt3 = bill_amt3 - pay_amt3,
         rem_amt4 = bill_amt4 - pay_amt4,
         rem_amt5 = bill_amt5 - pay_amt5,
         rem_amt6 = bill_amt6 - pay_amt6,
         pay_rate1 = pay_amt1 / limit_bal,
         pay_rate2 = pay_amt2 / limit_bal,
         pay_rate3 = pay_amt3 / limit_bal,
         pay_rate4 = pay_amt4 / limit_bal,
         pay_rate5 = pay_amt5 / limit_bal,
         pay_rate6 = pay_amt6 / limit_bal)

rm(temp)

# Train/Test split ----

set.seed(11)
split <- initial_split(data_cleaned, prop = 0.8, strata = default)
train_data <- split %>% training()
test_data <- split %>% testing()
# smote_data <- SMOTE(default ~ ., as.data.frame(train_data))

set.seed(11)
split_featured <- initial_split(data_featured, prop = 0.8, strata = default)
train_featured <- split_featured %>% training()
test_featured <- split_featured %>% testing()
# smote_featured <- SMOTE(default ~ ., as.data.frame(train_featured))