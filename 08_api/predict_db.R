#* Return prediction
#* @get /predict
#* @serializer unboxedJSON
#* @response 201 Class prediction
#* @response 400 Error message - string
function(req, res) {
  
  lgbm <- lgb.load('../05_saved_models/lightgbm.txt')
  
  # DB connection params
  dbname <- 'rating'
  db_table <- 'client_parameters_1' # change -> client_parameters
  username <- 'postgres'
  password <- 'passw0rd'
  host <- '192.168.150.235' # machine host -> 'host.docker.internal'
  port <- 5432
  
  # Testing purposes DB
  # dbname <- 'ccdefault' # testing db
  # db_table <- 'client_parameters' # testing table
  # username <- 'postgres'
  # password <- 'root' # testing password
  # host <- 'host.docker.internal' # way to communicate from docker container to localhost
  # port <- 5432
  
  # Establish connection to DB
  conn <- DBI::dbConnect(RPostgres::Postgres(),
                         dbname = dbname,
                         user = username,
                         password = password,
                         host = host,
                         port = port)
  
  # DB Table to work with
  data <- tbl(conn, db_table)
  
  # Select parameters column from client_parameters
  params <- data %>%
    as_tibble %>% 
    filter(is.na(ml_probability)) %>% 
    select(parameters)
  
  # Names of model input columns
  colnames_model <- c("ID_CLIENT",
                      "LIMIT_BAL",
                      "SEX",
                      "EDUCATION",
                      "MARRIAGE",
                      "AGE",
                      "PAY_0",
                      "PAY_2",
                      "PAY_3",
                      "PAY_4",
                      "PAY_5",
                      "PAY_6",
                      "BILL_AMT1",
                      "BILL_AMT2",
                      "BILL_AMT3",
                      "BILL_AMT4",
                      "BILL_AMT5",
                      "BILL_AMT6",
                      "PAY_AMT1",
                      "PAY_AMT2",
                      "PAY_AMT3",
                      "PAY_AMT4",
                      "PAY_AMT5",
                      "PAY_AMT6")
  
  
  # Wrangle data to model input format
  data <- jsonlite::stream_in(textConnection(params$parameters))
  if(dim(data)[1] > 0) {
    
    data <- data %>%
      furrr::future_map_dfc(~ select(., contains('name'), contains('value')) %>% 
                pivot_wider(names_from = 'parameter_name', values_from = 'parameter_value')) %>% 
      unnest(everything()) %>% 
      select(all_of(colnames_model))
    
    data <- data %>%
      as_tibble %>% 
      clean_names() %>%
      rename(pay_1 = pay_0,
             id = id_client)
  
  
    temp <- data %>% select(contains('pay'), -contains('amt'))
    
    data <- data %>%
      mutate_at(vars(contains('pay'), -contains('amt')), ~factor(ifelse(. %in% c(-2, -1, 0), 0, .))) %>%
      mutate(sex = factor(sex),
             education = factor(ifelse(education %in% c(0, 4, 5, 6), 4, education)),
             marriage = factor(ifelse(marriage %in% c(0, 3), 3, marriage)),
             age = factor(ifelse(age <= 30, 0,
                                 ifelse(age <= 40, 1,
                                        ifelse(age <= 50, 2,
                                               ifelse(age <= 60, 3, 4))))),
             months_not_paid = (temp > 0) %>% rowSums,
             months_not_paid_sum = (temp*(temp > 0)) %>% rowSums,
             months_not_paid_sum_cat = cut(months_not_paid_sum,
                                           breaks = c(-Inf, 1, 10, +Inf),
                                           labels = c(0, 1, 2)),
             rem_amt1 = bill_amt1 - pay_amt1,
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
    
    rm(temp, params)
    
    # Get model predictions
    data <- data %>%
      mutate(probability = stats::predict(lgbm, data.matrix(data %>% select(-id) %>% 
                                                              mutate_if(is.factor, as.character) %>% 
                                                              mutate_if(is.character, as.numeric))),
             default = ifelse(probability > 0.231887956925153, 1L, 0L)) %>% 
      select(id, probability, default)
    
    # Update db
    pwalk(data, ~ RPostgres::dbClearResult(RPostgres::dbSendQuery(conn = conn,
                                                                statement = paste('UPDATE', db_table,
                                                                                  'SET ml_probability =', ..2, 
                                                                                  ', "ml_class" =', ..3,
                                                                                  'WHERE "client_id" =', ..1))))
    
    res$status <- 201
    success <- T
    msg <- 'Successfully updated db.'
    res$body <- msg
    
  } else {
    
    res$status <- 200
    success <- F
    msg <- 'No new clients to predict'
    res$body <- msg
    
  }
  # End connection
  DBI::dbDisconnect(conn = conn)
  
  # Print our workspace memory usage
  # for(el in ls()) {
  #   print(paste(el, format(object.size(get(el)), units = 'KB')))
  # }
  
  # Clear out workspace memory
  rm(list = (setdiff(ls(), c('success', 'msg'))))
  
  # Call garbage collector
  gc()
  
  # Return response
  list(success = jsonlite::unbox(success),
       msg = jsonlite::unbox(msg))
  
}
