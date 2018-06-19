## DEMAND FORECASTING

# Initialization
{
  gc()
  cat("\014")
  rm(list = setdiff(ls(), c()))

  packages = function(x) {
    x = as.character(match.call()[[2]])
    if (!require(x,character.only = TRUE)){
      install.packages(pkgs = x, repos = "http://cran.r-project.org", dependencies = T, quiet = T)
      require(x, character.only = TRUE)
    }
  }

  suppressMessages(
    {
      packages("withr")
      packages("data.table")
      packages("magrittr")
      packages("dplyr")
      packages("readr")
      packages("stringr")
      packages("xgboost")
      packages("rBayesianOptimization")
    })
}

# Input
{
  dataset = read_csv('XYZ_Dataset_B.csv', progress = T, trim_ws = T)
}

# Pre-Processing
{
  dummy_grid = expand.grid(Store = unique(dataset$Store),
                           Dept = unique(dataset$Dept),
                           Date = unique(dataset$Date),
                           stringsAsFactors = F)

  ads = left_join(dummy_grid, dataset) %>%
    mutate(IsHoliday = as.numeric(IsHoliday) %>% if_else(is.na(.), 0, .),
           Weekly_Sales = if_else(is.na(Weekly_Sales), 0, Weekly_Sales))
}

# Processing/Feature Engineering
{
  ads %<>%
    mutate(year = gsub(Date, pattern = "(.*)\\/", replacement = ""),
           day = str_extract(Date, pattern = regex("(?<=\\/).*(?=\\/(.*?))", perl = T)) %>%
             str_pad(., width = 2, side = "left", pad = "0"),
           month = gsub(Date, pattern = "\\/(.*)", replacement = "") %>%
             str_pad(., width = 2, side = "left", pad = "0"),
           year_month = paste0(year, month))

  ads_aggr = ads %>%
    group_by(year_month, Store, Dept) %>%
    summarise(Weekly_Sales = sum(Weekly_Sales),
              IsHoliday = sum(IsHoliday),
              year = unique(year)[1],
              month = unique(month)[1]) %>%
    ungroup %>%
    mutate(store_dept = paste0("S", Store, "D", str_pad(Dept, 2, "left", "0"))) %>%
    mutate_if(.predicate = colnames(.) %in% setdiff(colnames(.), "store_dept"),
              .funs = as.numeric)

  # split for validation
  # using 2012-06 as threshold/cut for train/test
  train = ads_aggr %>% filter(year_month <= 201204)
  test = ads_aggr %>% filter(year_month > 201204)

  # some more feat engineering
  # supervised engineering, hence use only train dataset for it
  {
    vector_1 = c("Store", "Dept", "IsHoliday", "store_dept")
    vector_2 = c("year", "month", "IsHoliday")

    vct = expand.grid(vector_1, vector_2, stringsAsFactors = F) %>% distinct %>% filter(Var1 != Var2)

    group_fn = function(train, test, i) {
      temp = train

      groupby_vars = vct %>% .[i, ] %>% as.character

        temp2 = temp %>%
          group_by_(.dots = groupby_vars) %>%
          summarise(mean_sales = mean(Weekly_Sales, na.rm = T),
                    sd_sales = sd(Weekly_Sales, na.rm = T),
                    perc_10_sales = quantile(Weekly_Sales, probs = c(0.1)),
                    perc_90_sales = quantile(Weekly_Sales, probs = c(0.9))) %>%
          ungroup %>%
          set_colnames(append(groupby_vars, paste0("mean_", paste0(groupby_vars, collapse = "_"))) %>%
                         append(., paste0("sd_", paste0(groupby_vars, collapse = "_"))) %>%
                         append(., paste0("perc_10_", paste0(groupby_vars, collapse = "_"))) %>%
                         append(., paste0("perc_90_", paste0(groupby_vars, collapse = "_"))))

        train <<- left_join(train, temp2)
        test <<- left_join(test, temp2)
    }

    for (i in 1:nrow(vct)) {
      group_fn(train, test, i)
    }

    for (i in 1:nrow(vct)) {
      temp = ads_aggr

      groupby_vars = vct %>% .[i, ] %>% as.character

      temp2 = temp %>%
        group_by_(.dots = groupby_vars) %>%
        summarise(mean_sales = mean(Weekly_Sales, na.rm = T),
                  sd_sales = sd(Weekly_Sales, na.rm = T),
                  perc_10_sales = quantile(Weekly_Sales, probs = c(0.1)),
                  perc_90_sales = quantile(Weekly_Sales, probs = c(0.9))) %>%
        ungroup %>%
        set_colnames(append(groupby_vars, paste0("mean_", paste0(groupby_vars, collapse = "_"))) %>%
                       append(., paste0("sd_", paste0(groupby_vars, collapse = "_"))) %>%
                       append(., paste0("perc_10_", paste0(groupby_vars, collapse = "_"))) %>%
                       append(., paste0("perc_90_", paste0(groupby_vars, collapse = "_"))))

      ads_aggr <<- left_join(ads_aggr, temp2)
    }
  }

  rm(ads, dataset, dummy_grid, vct, vector_1, vector_2, i)
}

# xgboost
{
  train %<>%
    mutate(store_dept = gsub(store_dept, pattern = "[[:alpha:]]", replacement = ""))
  test %<>%
    mutate(store_dept = gsub(store_dept, pattern = "[[:alpha:]]", replacement = ""))
  ads_aggr %<>%
    mutate(store_dept = gsub(store_dept, pattern = "[[:alpha:]]", replacement = ""))

  train_xgb = xgboost::xgb.DMatrix(data = train %>% select(-year_month, -Weekly_Sales) %>% data.matrix,
                                   label = train %>% .$Weekly_Sales %>% as.numeric)
  test_xgb = xgboost::xgb.DMatrix(data = test %>% select(-year_month, -Weekly_Sales) %>% data.matrix)
  test_label = test %>% select(Weekly_Sales)

  # custom function modified from MLBayesOpt package for bayesian hyper-parameter tuning
  xgb_opt_custom = function(train_df, test_df, test_label, objectfun,
                            evalmetric, eta_range = c(0.001, 0.5), max_depth_range = c(4, 8L), nrounds_range = c(50, 160L), subsample_range = c(0.1, 1L), bytree_range = c(0.4, 1L), init_points = 5, n_iter = 10, acq = "ucb", kappa = 2.576, eps = 0, optkernel = list(type = "exponential", power = 2), classes = NULL) {
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

  xgb_bayes_output = xgb_opt_custom(train_df = train_xgb, test_df = test_xgb, test_label = test_label,
          objectfun = "reg:linear", evalmetric = "rmse",
          eta_range = c(0.01, 0.3), max_depth_range = c(3, 6L),
          nrounds_range = c(500, 1000L), subsample_range = c(0.3, 0.9),
          bytree_range = c(0.4, 0.8), init_points = 5, n_iter = 5, acq = "ucb",
          kappa = 2.576)

  # params from the bayesian optimized random search hyperparameter tuning
  params = list(booster = "gbtree",
                eta = xgb_bayes_output$Best_Par["eta_opt"],
                max_depth = as.integer(round(xgb_bayes_output$Best_Par["max_depth_opt"], digits = 0)),
                subsample = xgb_bayes_output$Best_Par["subsample_opt"],
                colsample_bytree = xgb_bayes_output$Best_Par["bytree_opt"],
                obj = "reg:linear",
                eval_metric = "rmse")

  xgb_model = xgb.train(params = params, nrounds = xgb_bayes_output$Best_Par["nrounds_opt"], data = train_xgb, verbose = T, print_every_n = 10)

  output_xgb = predict(xgb_model, test_xgb) %>%
    data.frame %>%
    set_colnames("pred_xgb") %>%
    bind_cols(., test %>% select(store_dept, year_month, Weekly_Sales)) %>%
    mutate(ape = abs((Weekly_Sales - pred_xgb)/Weekly_Sales),
           ape = if_else(ape > 1 | is.infinite(ape), 1, ape))

  print(paste0("mape is: ", output_xgb %>% .$ape %>% mean * 100, "%"))

  # create model for whole train data and predict for 2012-11
  {
    ads_aggr_xgb = xgboost::xgb.DMatrix(data = ads_aggr %>% select(-year_month, -Weekly_Sales) %>% data.matrix,
                                        label = ads_aggr %>% .$Weekly_Sales %>% as.numeric)
    xgb_model = xgb.train(params = params, nrounds = x$Best_Par["nrounds_opt"], data = ads_aggr_xgb, verbose = T, print_every_n = 10)

    # creating the test set for new values for xgboost
    {
      test_xgb_new = expand.grid(Store = unique(ads_aggr$Store),
                                 Dept = unique(ads_aggr$Dept),
                                 year = 2012,
                                 month = 11,
                                 IsHoliday = 1,
                                 stringsAsFactors = F)

      test_xgb_new_filtered = ads_aggr %>%
        filter(Store %in% unique(test_xgb_new$Store)) %>%
        filter(Dept %in% unique(test_xgb_new$Dept)) %>%
        filter(month %in% unique(test_xgb_new$month)) %>%
        group_by(Store, Dept, month) %>%
        summarise_all(mean) %>%
        ungroup %>%
        mutate(store_dept = paste0(Store, str_pad(Dept, 2, "left", "0")),
               year = 2012:2012) %>%
        select(colnames(test))

      test_xgb_new = xgboost::xgb.DMatrix(data = test_xgb_new_filtered %>%
                                            select(-year_month, -Weekly_Sales) %>%
                                            data.matrix)
    }
    output_final_xgb = predict(xgb_model, test_xgb_new) %>%
      data.frame %>%
      set_colnames(c("2012-11-forecast")) %>%
      bind_cols(., test_xgb_new_filtered %>% select(store_dept))
    }
}

# hts
{
  library(forecast)
  library(hts)

  train_ts = train %>%
    select(store_dept, year_month, Weekly_Sales) %>%
    reshape2::acast(., formula = year_month ~ store_dept) %>%
    ts %>%
    hts(., characters = c(1,2))

  train_xreg = train %>%
    select(store_dept, year_month, IsHoliday, year, month) %>%
    group_by(year_month) %>%
    summarise(IsHoliday = mean(IsHoliday),
              year = max(year),
              month = max(month)) %>%
    select(-year_month)
  test_xreg = test %>%
    select(store_dept, year_month, IsHoliday, year, month) %>%
    group_by(year_month) %>%
    summarise(IsHoliday = mean(IsHoliday),
              year = max(year),
              month = max(month)) %>%
    select(-year_month)

  hts_model = forecast(object = train_ts, xreg = train_xreg, newxreg = test_xreg, h = 6, fmethod = "arima", seasonal = T)

  test_ts = test %>%
    select(store_dept, year_month, Weekly_Sales) %>%
    reshape2::acast(., formula = year_month ~ store_dept) %>%
    ts %>%
    hts(., characters = c(1,2))

  output_hts = rbind(hts_model$bts, test_ts %>% .$bts)
  x = hts_model$bts %>% data.frame %>% melt %>% mutate(store_dept = str_replace(variable, pattern = "X", ""),
                                                       year_month = rep_len(201207:201210, nrow(.))) %>%
    select(store_dept, year_month, pred_hts = value) %>%
    left_join(., output_xgb)
  #accuracy.gts(hts_model, test = test_ts)

  # lm
  {
    y = x %>% select(pred_xgb, pred_hts, Weekly_Sales) %>% mutate(avg = (pred_xgb + pred_hts)/2)

    y_lm = lm(formula = Weekly_Sales ~ .,
              data = y)
    # this returned a formula given in coefficients of the model
    # will use this model as the formula for the simple ensemble
    }
}

# hts - using whole train data and forecast for november
{
  library(forecast)
  library(hts)

  train_ts = ads_aggr %>%
    select(store_dept, year_month, Weekly_Sales) %>%
    reshape2::acast(., formula = year_month ~ store_dept) %>%
    ts %>%
    hts(., characters = c(2,3))

  train_xreg = ads_aggr %>%
    select(store_dept, year_month, IsHoliday, year, month) %>%
    group_by(year_month) %>%
    summarise(IsHoliday = mean(IsHoliday),
              year = max(year),
              month = max(month)) %>%
    select(-year_month)
  test_xreg = expand.grid(IsHoliday = 1,
                          year = 2012,
                          month = 11,
                          stringsAsFactors = F)

  hts_model = forecast(object = train_ts, xreg = train_xreg, newxreg = test_xreg, h = 1, fmethod = "arima", seasonal = T)

  output_hts = hts_model$bts %>% data.frame %>% `row.names<-`(c("201211"))
}
