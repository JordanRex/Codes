######################
# ML snippets #
# model functions (for ranger, xgboost, h2o - gbm/randomforest/dnn)
# feature manipulation functions (for PCA, ICA, deviation encoding, bucketing, feature selection)

# Basic steps are to pre-process train and test data and split the response variable into a separate vector
# have only the independant features for use in modelling as part of the train and test (no id, response vars)
######################

## XGBOOST ##
##################################################################################################################################
### function to create an xgboost model ###
# one argument = input dataset with both train and test datasets appended
# not fully dynamic: need to change name of response variable referenced internally,
# also hardcoded for binary classification, customize function as and when necessary
# most general params declared with arguments

# xgboost function
xgb_model_fn = function(train, test, response = "dep", test_ids,
                        nround = 500,
                        params = list(eta = 0.01,
                                      maxdepth = 7,
                                      subsample = 0.65,
                                      colsample = 0.7),
                        evalmetric_fn = c("auc", "rmse", "mlogloss"),
                        objective_fn = c("binary:logistic", "reg:linear", "multi:softprob"),
                        printevery_n = 50) {
  
  # uncomment in case if you are doing classification and need custom weights to balance classes
  # weights = response %>%
  #   data.frame %>% set_colnames("response") %>%
  #   mutate(weight = if_else(response == 0, 3, 1)) %>%
  #   .$weight
  
  train_xgb = xgb.DMatrix(data = train %>% as.matrix,
                          label = response %>% as.numeric)
  test_xgb = xgb.DMatrix(data = test %>% as.matrix)

  xgb_model = xgb.train(data = train_xgb,
                        nrounds = nround,
                        print_every_n = printevery_n,
                        verbose = 1,
                        objective = objective_fn,
                        eval_metric = evalmetric_fn,
                        # base_score = 0.5, # applicable only for classification
                        # weight = weights, # applicable if need to balance classes for classification
                        verbose = T,
                        params = params)
  xgboost::xgb.save(xgb_model, fname = 'xgb_model')

  output = data.frame(id = test_ids, response = as.numeric(predict(xgb_model, test_xgb)))
  assign("output", value = output, envir = .GlobalEnv)
  fwrite(output, 'output.csv')

  return("seems fine")
}

#--------------------------------------------------------------------------------------------------------------------------------#

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

#--------------------------------------------------------------------------------------------------------------------------------#

### custom function modified from MLBayesOpt package for bayesian hyper-parameter tuning ###
# can perform tuning for classification (binary/multi) and regression
# needs more testing for special cases and other parameters
# provide ranges for parameters if required (defaults provided), needs what acquisition method to be used for the bayesian segment
# takes xgb.Dmatrix objects as input for simplicity (modify if only dataframe can be provided, not recommended)
xgb_opt_custom = function(train_df, test_df, test_label, objectfun,
                          evalmetric, eta_range = c(0.001, 0.2), max_depth_range = c(4L, 12L), nrounds_range = c(100L, 1000L), subsample_range = c(0.5, 0.9L), bytree_range = c(0.4, 0.8L), init_points = 20, n_iter = 10, acq = c("ucb", "ei", "poi"), kappa = 2.576, eps = 0, optkernel = list(type = "exponential", power = 2), classes = NULL) {
  dtrain <- train_df
  dtest <- test_df
  if (grepl("logi", objectfun) == TRUE) {
    xgb_holdout <- function(object_fun, eval_met, num_classes,
                            eta_opt, max_depth_opt, nrounds_opt, subsample_opt,
                            bytree_opt) {
      object_fun <- objectfun
      eval_met <- evalmetric
      model <- xgb.train(params = list(objective = object_fun,
                                       eval_metric = eval_met, nthread = 1, eta = eta_opt,
                                       max_depth = as.integer(round(max_depth_opt, digits = 0)), subsample = subsample_opt,
                                       colsample_bytree = bytree_opt), data = dtrain,
                         nrounds = nrounds_opt)
      t_pred <- predict(model, newdata = dtest)
      Pred <- sum(diag(table(data_test_label, t_pred)))/nrow(test_data)
      list(Score = Pred, Pred = Pred)
    }
  }
  else
    if (grepl("multi", objectfun) == TRUE) {
      xgb_holdout <- function(object_fun, eval_met, num_classes,
                              eta_opt, max_depth_opt, nrounds_opt, subsample_opt,
                              bytree_opt) {
        object_fun <- objectfun
        eval_met <- evalmetric
        num_classes <- classes
        model <- xgb.train(params = list(objective = object_fun,
                                         num_class = num_classes, nthread = 1, eval_metric = eval_met,
                                         eta = eta_opt, max_depth = as.integer(round(max_depth_opt, digits = 0)), subsample = subsample_opt,
                                         colsample_bytree = bytree_opt), data = dtrain,
                           nrounds = nrounds_opt)
        t_pred <- predict(model, newdata = dtest)
        Pred <- sum(diag(table(data_test_label, t_pred)))/nrow(test_data)
        list(Score = Pred, Pred = Pred)
      }
    }
  else {
    xgb_holdout <- function(object_fun, eval_met,
                            eta_opt, max_depth_opt, nrounds_opt, subsample_opt,
                            bytree_opt) {
      object_fun <- objectfun
      eval_met <- evalmetric
      model <- xgb.train(params = list(objective = object_fun, nthread = 1, eval_metric = eval_met,
                                       eta = eta_opt, max_depth = as.integer(round(max_depth_opt, digits = 0)), subsample = subsample_opt,
                                       colsample_bytree = bytree_opt), data = dtrain,
                         nrounds = nrounds_opt)
      t_pred_df = data.frame(actuals = test_label[, 1, drop = T], pred = predict(model, newdata = dtest)) %>%
        mutate(APE = abs((actuals - pred)/(if_else(actuals == 0, 1, actuals))),
               APE = if_else(APE > 1, 1, APE))
      Pred = 1 - mean(t_pred_df$APE)
      list(Score = Pred, Pred = Pred)
    }
  }
  opt_res <- rBayesianOptimization::BayesianOptimization(xgb_holdout, bounds = list(eta_opt = eta_range,
                                                                                    max_depth_opt = max_depth_range, nrounds_opt = nrounds_range,
                                                                                    subsample_opt = subsample_range, bytree_opt = bytree_range),
                                                         init_points, init_grid_dt = NULL, n_iter, acq, kappa,
                                                         eps, optkernel, verbose = TRUE)
  return(opt_res)
}

# example ->
{
# xgb_bayes_output = xgb_opt_custom(train_df = train_xgb, test_df = test_xgb, test_label = test_label,
#                                   objectfun = "reg:linear", evalmetric = "rmse",
#                                   eta_range = c(0.01, 0.3), max_depth_range = c(3, 6L),
#                                   nrounds_range = c(500, 1000L), subsample_range = c(0.3, 0.9),
#                                   bytree_range = c(0.4, 0.8), init_points = 5, n_iter = 5, acq = "ucb",
#                                   kappa = 2.576)
#
# # params from the bayesian optimized random search hyperparameter tuning
# params = list(booster = "gbtree",
#               eta = xgb_bayes_output$Best_Par["eta_opt"],
#               max_depth = as.integer(round(xgb_bayes_output$Best_Par["max_depth_opt"], digits = 0)),
#               subsample = xgb_bayes_output$Best_Par["subsample_opt"],
#               colsample_bytree = xgb_bayes_output$Best_Par["bytree_opt"],
#               obj = "reg:linear",
#               eval_metric = "rmse")
#
# xgb_model = xgb.train(params = params, nrounds = xgb_bayes_output$Best_Par["nrounds_opt"], data = train_xgb, verbose = T, print_every_n = 10)
}

#--------------------------------------------------------------------------------------------------------------------------------#

### function to do n-fold cv xgboost ###
# used for parameter tuning with grid/random search without a train/test split but based off on n-fold cv results
# define folds, default set here for classification. subtle changes required to generalize
# n-fold cv with random search on parameters (some default sensible ranges given)
nfold_xgb_fn = function(train_xgb, response, folds = 5, maximize = TRUE,
                        evalmetric = c("auc", "mlogloss", "rmse"), 
                        objective_fn = c("binary:logistic", "multi:softprob", "reg:linear"),
                        n = 20) {
  best_param = list()
  best_seednumber = 1234
  best_score = Inf
  best_score_index = 0

  for (iter in 1:n) {
    param = list(objective = objective_fn,
                 eval_metric = evalmetric,
                 num_class = length(unique(response[, 1])),
                 max_depth = sample(5:12, 1),
                 eta = runif(1, .001, .2),
                 gamma = runif(1, 0.0, 0.2),
                 subsample = runif(1, .6, .9),
                 colsample_bytree = runif(1, .5, .8),
                 min_child_weight = sample(1:10, 1),
                 max_delta_step = sample(1:10, 1)
    )
    cv.nround = sample(seq.int(100, 1000, 100), 1)
    cv.nfold = folds
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    mdcv <- xgb.cv(data=train_data_matrix, params = param, stratified = T, print_every_n = 50,
                   nfold = cv.nfold, nrounds = cv.nround,
                   verbose = T, early_stopping_rounds = 10, maximize = maximize)

    evalmetric_col = paste0("test_", evalmetric, "_mean")

    min_score = min(mdcv$evaluation_log[[evalmetric_col]])
    min_score_index = which.min(mdcv$evaluation_log[[evalmetric_col]])

    if (min_score < best_score) {
      best_score = min_score
      best_score_index = min_score_index
      best_seednumber = seed.number
      best_param = param
    }
    print(paste0("iter is ", i))
  }

  nround = best_score_index
  set.seed(best_seednumber)

  best_params_df = data.frame(best_param) %>%
    cbind(., data.frame(nround = nround))
  saveRDS(best_params_df, 'best_params_df.rds')

  return(best_params_df)
}

##################################################################################################################################

## RANGER(RF) ##
##################################################################################################################################


##################################################################################################################################

## HTS/FORECAST ##
##################################################################################################################################

### function to create an hts/gts object ###
# make a singular id column, singular time dimension, and specify characters convention used for creating the id column
# example -> id_col = store + department = S{1-9} + D{1-99} = S01D01 = characters(2,3)
train_hts_fn = function(train, id_col = "id", time_col = "date", dep_col = "dep") {
  acast_fn = paste0(time_col , '~', id_col)

  train_ts = x %>%
    select_(id_col, time_col, dep_col) %>%
    reshape2::acast(., formula = acast_fn) %>%
    ts %>%
    hts(., characters = c(2,3))

  return(train_ts)
}

#--------------------------------------------------------------------------------------------------------------------------------#

### function to create and return an hts model
# specify the train ts object, train/test xreg objects (only if you have regressors! else skip argument), fmethod to be used (ets|arima|hw?), seasonal = T or F, h = horizon to be predicted
hts_model_fn = function(train_ts, train_xreg = NULL, test_xreg = NULL, h = 12, fmethod = "arima", seasonal = T) {
  hts_model = forecast(object = train_ts, xreg = train_xreg, newxreg = test_xreg, h = h, fmethod = fmethod, seasonal = seasonal)
  return(hts_model)
}
# save to hts_model, predictions are saved in hts_model$bts. Bind with test_ts$bts to get output dataframe
# output_hts = rbind(hts_model$bts, test_ts %>% .$bts)

##################################################################################################################################

## GLMNET for text features selection ##
##################################################################################################################################

### function to select the top n relevant features after an NLP process
# set for classification (binary) -> modify accordingly for regression/multi-class classification
# needs to be tested for whether it works in the form of a function, intended for leveraging, not as in usage
glmnet_fn = function(train, train_dep, test_dep, response = "target") {
  NFOLDS = 5
  t1 = Sys.time()
  glmnet_classifier = cv.glmnet(x = train, y = train_dep[[response]],
                                family = 'binomial',
                                # L1 penalty
                                alpha = 1,
                                # interested in the area under ROC curve
                                type.measure = "auc",
                                # 5-fold cross-validation
                                nfolds = NFOLDS,
                                # high value is less accurate, but has faster training
                                thresh = 1e-5,
                                # again lower number of iterations for faster training
                                maxit = 1e7)
  print(difftime(Sys.time(), t1, units = 'sec'))

  plot(glmnet_classifier)
  print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

  preds = as.numeric(predict(glmnet_classifier, train, type = 'response'))

  glmnet:::auc(test_dep[[response]], preds)
}

##################################################################################################################################

## FEATURE ENGINEERING ##
##################################################################################################################################
### function to do deviation encoding ###
# the arguments are the train, test dataframes; and the character columns' subset and numeric dependant variable dataframes
# works best for regression problems (forecasting)
# for classification convert the dependant to a numeric (label encoded) column
# mode and skew needs to be tested, also select whatever functions ONLY if you know what you are doing
feateng_categtoDeviationenc_fn = function(train, test, dep_var) {
    train_char_data = train %>% select_if(function(col) is.character(col) | is.logical(col)) %>%
      mutate_all(as.character)

    for (i in 1:ncol(train_char_data)) {
      temp_col = colnames(train_char_data[, i, drop = F])

      dev_encod_cols = paste0(temp_col, c("_mean", "_min", "_max", "_sd", "_median", "_ndistinct", "_mode", "_skew"))
      dev_funs = paste0(c("mean(", "min(", "max(", "sd(", "median(", "n_distinct("), "dep", ")") %>%
        append(., c("mlv(dep, method = 'mfv')[['M']]", "mlv(dep, method = 'mfv')[['skewness']]")) %>% as.list

      temp = train_char_data[, i, drop = F] %>%
        bind_cols(., dep_var %>% set_colnames("dep")) %>%
        group_by_at(vars(-matches("dep"))) %>%
        mutate_(.dots = setNames(dev_funs, dev_encod_cols)) %>%
        ungroup %>%
        select(-dep) %>%
        distinct %>%
        mutate_at(vars(-matches(lazyeval::uq(temp_col))), as.numeric)

      train <<- left_join(train, temp)
      test <<- left_join(test, temp)
    }
    return(print("train and test have been generated"))
  }

#--------------------------------------------------------------------------------------------------------------------------------#

### function to do bucket feature creation ###
# unsupervised, create simply by specifying train set
feateng_bucket_fn_unsup = function(train) {
  temp = train

  temp_categ = temp %>%
    select_if(is.character) %>%
    summarise_all(.funs = n_distinct) %>%
    melt %>%
    .$variable

  feateng_bucket_col_fn = function(x) {
    x = as.character(x)

    temp_bucket_num = floor(length(unique(x)) /
                              floor(sqrt(length(unique(x)))))

    temp_x = temp %>%
      group_by_(.dots = list(x)) %>%
      summarise()
  }
}

# supervised, uses dependant variable mean and sd with each level in the categorical features
# preferred for a categorical variable if the number of levels it contains is < 50-100 (argument n gives this control)
feateng_bucket_fn_sup = function(train, dep, col) {
  temp = train %>% select_(indep = col, dep)
  temp_test = test %>% select_(indep = col)
  temp_extreme_cuts = append(temp$indep, temp_test$indep) %>% unique %>% range(.) #%>% .[. %in% c(max(.), min(.))]

  temp_model = rpart::rpart(data = temp, formula = paste0(dep, "~ indep"))
  temp_splits = temp_model$splits[ ,4] %>% as.numeric %>% append(., temp_extreme_cuts) %>% sort %>% unique

  if (is.null(temp_model$splits)) { temp_splits %<>% append(., mean(temp$indep)) %>% sort %>% unique }

  temp_train_cut = data.frame(temp = cut(temp$indep, breaks = temp_splits, labels = F, include.lowest = T)) %>%
    set_colnames(value = paste0("bucket_", col)) %>% bind_cols(., temp %>% select(indep) %>% set_colnames(col)) %>%
    distinct
  temp_test_cut = data.frame(temp = cut(temp_test$indep, breaks = temp_splits, labels = F, include.lowest = T)) %>%
    set_colnames(value = paste0("bucket_", col)) %>% bind_cols(., temp_test %>% select(indep) %>% set_colnames(col)) %>%
    distinct
  train %<>% left_join(., temp_train_cut)
  test %<>% left_join(., temp_test_cut)

  train <<- train
  test <<- test
}

# example usage ->
# bucket_feats = train %>%
#   summarise_all(n_distinct) %>%
#   melt %>%
#   set_colnames(c("column", "count")) %>%
#   mutate(bucket_feat_flag = if_else(count >= threshold_ndistinct, 1, 0)) %>%
#   filter(bucket_feat_flag == 1) %>% .$column %>% as.character
#
# for (i in 1:length(bucket_feats)) {
#   feateng_bucket_fn(train, bucket_feats[i])
#}

#--------------------------------------------------------------------------------------------------------------------------------#

### function to d0 label encoding ###
# label encode and store all character columns for algorithms that require a numeric matrix
labelencoder_fn = function(train, test, response) {
  categ_cols = train %>% select_if(is.character) %>% colnames
  rem_cols = setdiff(colnames(train), categ_cols)

  train_categ = train %>% select(categ_cols)
  test_categ = test %>% select(categ_cols)
  train_test_categ = bind_rows(train_categ, test_categ)

  train_rest = train %>% select(rem_cols) %>% mutate(tt = "train")
  test_rest = test %>% select(rem_cols) %>% mutate(tt = "test")
  train_test_rest = bind_rows(train_rest, test_rest)

  encoded_list = list()

  for (i in 1:length(categ_cols)) {
    encoded_list[[i]] = CatEncoders::LabelEncoder.fit(train_test_categ[, i])
    train_test_categ[, i] = CatEncoders::transform(encoded_list[[i]], train_test_categ[, i])
  }

  train_test_encoded = train_test_categ %>%
    bind_cols(., train_test_rest)

  return(train_test_encoded)
}

##################################################################################################################################
