######################
# Modelling snippets #
######################

##################################################################################################################################
### function to create an xgboost model ###
# one argument = input dataset with both train and test datasets appended
# not fully dynamic: need to change name of response variable referenced internally,
# also hardcoded for binary classification, customize function as and when necessary
# most general params declared with arguments

# xgboost function
xgb_model_fn = function(x, nround = 1000, eta = 0.01, weight = c(3, 1), maxdepth = 4, subsample = 0.5, colsample = 0.4) {
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
    mutate_(weight = if_else(project_is_approved == 0, weight[1], weight[2])) %>%
    .$weight

  train_xgb = xgb.DMatrix(data = temp_train %>% select(-project_is_approved,) %>% data.matrix,
                          label = as.numeric(temp_train$project_is_approved))
  test_xgb = xgb.DMatrix(data = temp_test %>% select(-project_is_approved) %>% data.matrix)

  xgb_model = xgb.train(data = train_xgb,
                        nrounds = nround,
                        print_every_n = 50,
                        verbose = 1,
                        eta = eta,
                        max_depth = maxdepth,
                        objective = "binary:logistic",
                        eval_metric = "auc",
                        subsample = subsample,
                        colsample_bytree = colsample,
                        base_score = 0.49,
                        weight = temp_train_weights,
                        verbose = T)
  xgboost::xgb.save(xgb_model, fname = 'xgb_model')

  output = data.frame(id = test_ids, project_is_approved = as.numeric(predict(xgb_model, test_xgb))) %>%
    mutate(project_is_approved = if_else(project_is_approved > 0.99, 1,
                                         if_else(project_is_approved < 0.01, 0, project_is_approved)))
  assign("output", value = output, envir = .GlobalEnv)
  fwrite(output, 'output.csv')

  return("seems fine")
}
##################################################################################################################################


##################################################################################################################################
### function to get the importance matrix from an xgboost model and select n numer of top features alone ###
# function 1 -> does the modelling and then returns the important features as a vector and a dataframe with rank
# objective set to binary:logistic, can be customized for any/all models (currently not dynamic)
xgb_importance_fn_1 = function(train, response = "dep", id = "id", nround = 500, eta = 0.01, maxdepth = 5, n = 500) {
  train_xgb = xgb.DMatrix(data = data.matrix(train %>% select_(.dots = list(paste0("-", response, id)))),
                          label = data.matrix(as.numeric(train %>% select_(response))))

  xgb_feat_selection_model = xgboost(data = train_xgb,
                                     nrounds = nround,
                                     eta = eta,
                                     objective = "binary:logistic",
                                     verbose = 1,
                                     max_depth = maxdepth,
                                     print_every_n = 50,
                                     early_stopping_rounds = 10)

  xgb_importance = data.table(xgboost::xgb.importance(feature_names = setdiff(colnames(train), c(response, id)),
                                                      model = xgb_feat_selection_model))

  Importance_table <<- data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
    mutate(Rank = dense_rank(desc(Importance))) %>%
    filter(Rank <= n)
  top_n_features <<- as.vector(Importance_table$Feature)
}

# function 2 -> takes as input an xgb model (all/any model)
# returns simply the n required features and the dataframe with rank
xgb_importance_fn_2 = function(xgbmodel, train, response = "dep", id = "id", n = 500) {
  xgb_importance = data.table(xgboost::xgb.importance(feature_names = setdiff(colnames(train), c(response, id)),
                                                      model = xgbmodel))

  Importance_table <<- data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
    mutate(Rank = dense_rank(desc(Importance))) %>%
    filter(Rank <= n)
  top_n_features <<- as.vector(Importance_table$Feature)
}
##################################################################################################################################


##################################################################################################################################
### function to do ranger ###


##################################################################################################################################
