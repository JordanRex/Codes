# modelling snippets #

# xgboost function
xgb_model_fn = function(x) {
  temp = x
  
  temp_train = temp %>%
    filter(tt == "train") %>%
    select(-id, -tt)
  
  temp_test = temp %>%
    filter(tt == "test") %>%
    select(-id, -tt)
  
  test_ids = temp %>%
    filter(tt == "test") %>%
    .$id
  
  temp_train_weights = temp_train %>%
    mutate(weight = if_else(project_is_approved == 0, 3, 1)) %>%
    .$weight
  
  train_xgb = xgb.DMatrix(data = temp_train %>% select(-project_is_approved,) %>% data.matrix,
                          label = as.numeric(temp_train$project_is_approved))
  test_xgb = xgb.DMatrix(data = temp_test %>% select(-project_is_approved) %>% data.matrix)
  
  xgb_model = xgb.train(data = train_xgb,
                        nrounds = 1000,
                        print_every_n = 50,
                        verbose = 1,
                        eta = 0.01,
                        max_depth = 4,
                        objective = "binary:logistic",
                        eval_metric = "auc",
                        subsample = 0.5,
                        colsample_bytree = 0.4,
                        base_score = 0.49,
                        weight = temp_train_weights,
                        verbose = T)
  
  output = data.frame(id = test_ids, project_is_approved = as.numeric(predict(xgb_model, test_xgb))) %>%
    mutate(project_is_approved = if_else(project_is_approved > 0.95, 1,
                                         if_else(project_is_approved < 0.05, 0, project_is_approved)))
  assign("output", value = output, envir = .GlobalEnv)
  write.csv(output, 'output.csv', row.names = F)
  
  return("seems fine")
}