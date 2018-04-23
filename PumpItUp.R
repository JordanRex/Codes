## Can you predict which water pumps are faulty? ##

# Using data from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional, which need some repairs, and which don't work at all?
# This is an intermediate-level practice competition. Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.


## MAIN CODE ##

# Initialization
{
  gc()
  cat("\014")
  rm(list = setdiff(ls(), c()))

  packages = function(x) {
    x = as.character(match.call()[[2]])
    if (!require(x,character.only = TRUE)) {
      install.packages(pkgs = x, repos = "http://cran.r-project.org", dependencies = T, quiet = T)
      require(x, character.only = TRUE)
    }
  }

  suppressMessages(
    {
      packages("data.table")
      packages("dplyr")
      packages("magrittr")
      packages("stringr")
      packages("DataExplorer")
      packages("xgboost")
    })
}

# Reading in the files
{
  train_feats = read.csv('train_feats.csv', header = T, stringsAsFactors = F, na.strings = c("", " ", "?", "NA"))
  test_feats = read.csv('test_feats.csv', header = T, stringsAsFactors = F, na.strings = c("", " ", "?", "NA"))

  train_dep = read.csv('train_dep.csv', header = T, stringsAsFactors = F, na.strings = c("", " ", "?", "NA"))
  sample_sub = read.csv('SubmissionFormat.csv')
}

# Processing
{
  # EDA
  # Using DataExplorer: an awesome package which generates a pdf with a single click on most/if not all information you would need for processing
  # DataExplorer::create_report(train_feats)

  train_feats %<>% select_if(function(x) length(unique(x)) > 1)
  test_feats %<>% select(colnames(train_feats))

  # missing value cols
  miss_cols = data.frame(miss = colSums(is.na(train_feats))) %>%
    mutate(col = row.names(.)) %>%
    filter(miss > 0) %>%
    .$col

  # treat each of the missing value columns
  fn_miss_treatment_categ = function(x, y = "unknown") {
    x[which(is.na(x))] = y
    return(x)
  }

  train_feats %<>% mutate_if(.predicate = colnames(.) %in% miss_cols,
                             .funs = fn_miss_treatment_categ)

  # label encode and store all character columns for immediate modelling
  categ_cols = train_feats %>% select_if(is.character) %>% colnames
  rem_cols = setdiff(colnames(train_feats), categ_cols)

  train_categ = train_feats %>% select(categ_cols) %>% mutate(tt = "train")
  test_categ = test_feats %>% select(categ_cols) %>% mutate(tt = "test")
  train_test_categ = bind_rows(train_categ, test_categ)

  train_rest = train_feats %>% select(rem_cols) %>% mutate(tt = "train")
  test_rest = test_feats %>% select(rem_cols) %>% mutate(tt = "test")
  train_test_rest = bind_rows(train_rest, test_rest)

  fn_labelencoder_df = function(x) {
    encoded_list = list()
    temp_df = x
    temp_df_cols = colnames(temp_df)

    for (i in 1:length(temp_df_cols)) {
      encoded_list[[i]] = CatEncoders::LabelEncoder.fit(temp_df[, i])
      x[, i] = CatEncoders::transform(encoded_list[[i]], x[, i])
    }
    return(x)
  }

  train_test_encoded = fn_labelencoder_df(train_test_categ) %>%
    mutate(tt = NULL) %>%
    cbind(., train_test_rest)

  rm(train_feats, test_feats, train_rest, test_rest, train_test_rest, train_categ, test_categ)
}

# Sample and Base Model
{
  gc()

  train = train_test_encoded %>%
    filter(tt == "train")
  test = train_test_encoded %>%
    filter(tt == "test")

  # encode the dep variable as well
  dep_encoded = CatEncoders::LabelEncoder.fit(train_dep$status_group)
  train_dep$status_group = CatEncoders::transform(dep_encoded, train_dep$status_group) - 1

  train_sample = sample_frac(tbl = train, size = 0.7) %>%
    select(-tt, -id)
  train_sample_dep_actual = train_sample %>%
    left_join(., train_dep) %>%
    select(id, status_group)
  test_sample = anti_join(train, train_sample, by = "id") %>%
    select(-tt, -id)
  test_sample_dep_actual = test_sample %>%
    left_join(., train_dep) %>%
    select(id, status_group)
  test_ids = test_sample_dep_actual %>%
    .$id

  train_data_matrix = xgb.DMatrix(data = as.matrix(train_sample),
                                  label = as.matrix(train_sample_dep_actual$status_group))
  test_data_matrix = xgb.DMatrix(data = as.matrix(test_sample),
                                 label = as.matrix(test_sample_dep_actual$status_group))

  numberOfClasses = length(unique(train_dep$status_group))

  # n-fold cv with random search on parameters
  {
    best_param = list()
    best_seednumber = 1234
    best_logloss = Inf
    best_logloss_index = 0

    for (iter in 1:25) {
      param = list(objective = "multi:softprob",
                   eval_metric = "mlogloss",
                   num_class = numberOfClasses,
                   max_depth = sample(10:15, 1),
                   eta = runif(1, .01, .2),
                   gamma = runif(1, 0.0, 0.2),
                   subsample = runif(1, .7, .9),
                   colsample_bytree = runif(1, .5, .7),
                   min_child_weight = sample(1:10, 1),
                   max_delta_step = sample(1:10, 1)
      )
      cv.nround = sample(seq.int(100, 500, 100), 1)
      cv.nfold = 5
      seed.number = sample.int(10000, 1)[[1]]
      set.seed(seed.number)
      mdcv <- xgb.cv(data=train_data_matrix, params = param, stratified = T, print_every_n = 50,
                     nfold = cv.nfold, nrounds = cv.nround,
                     verbose = T, early_stopping_rounds = 10, maximize = FALSE)

      min_logloss = min(mdcv$evaluation_log$test_mlogloss_mean)
      min_logloss_index = which.min(mdcv$evaluation_log$test_mlogloss_mean)

      if (min_logloss < best_logloss) {
        best_logloss = min_logloss
        best_logloss_index = min_logloss_index
        best_seednumber = seed.number
        best_param = param
      }
      print(paste0("iter is ", i))
    }

    nround = best_logloss_index
    set.seed(best_seednumber)
  }

  best_params_df = data.frame(best_param) %>%
    cbind(., data.frame(nround = nround))
  saveRDS(best_params_df, 'best_params_df.rds')

  train_full_matrix = xgb.DMatrix(data = as.matrix(train %>% select(-tt, -id)),
                                  label = as.matrix(train_dep$status_group))
  test_full_matrix = xgb.DMatrix(data = as.matrix(test %>% select(-tt, -id)))

  xgb_params = best_param
  nround = nround
  cv.nfold = 5

  # Fit cv.nfold * cv.nround XGB models and save OOF predictions
  cv_model = xgb.cv(params = xgb_params,
                    data = train_full_matrix,
                    nrounds = nround,
                    nfold = cv.nfold,
                    verbose = T,
                    prediction = T,
                    print_every_n = 25,
                    early_stopping_rounds = 10)

  OOF_prediction = data.frame(cv_model$pred) %>%
    mutate(max_prob = max.col(., ties.method = "last") - 1,
           label = train_dep$status_group,
           pred_flag = if_else(max_prob == label, 1, 0))

  md = xgb.train(data = train_full_matrix,
                 params = best_param,
                 nrounds = nround,
                 verbose = T,
                 print_every_n = 50,
                 early_stopping_rounds = 10)

  test_ids = test$id
  test_final_predictions = predict(md, test_full_matrix, reshape = T)
  output = data.frame(test_final_predictions) %>%
    mutate(status_group = max.col(., ties.method = "last")) %>%
    cbind(data.frame(id = test_ids), .) %>%
    select(id, status_group)
  output$status_group = CatEncoders::inverse.transform(dep_encoded, output$status_group)

  fwrite(output, 'output.csv')
}
