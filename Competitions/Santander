## santander

# Initialization
{
  gc()
  cat("\014")
  rm(list = setdiff(ls(), c()))

  options(scipen = 999)

  packages = function(x) {
    x = as.character(match.call()[[2]])
    if (!require(x,character.only = TRUE)) {
      install.packages(pkgs = x, repos = "http://cran.r-project.org", dependencies = T, quiet = T)
      require(x, character.only = TRUE)
    }
  }

  suppressMessages(
    {
      packages("xgboost")
      packages("stringr")
      packages("data.table")
      packages("magrittr")
      packages("readr")
      packages("dplyr")
    })
}

# Reading in the files
{
  train = read_csv('train.csv') %>% select(-ID)
  test = read_csv('test.csv') %>% select(-ID)
  test %<>% set_colnames(paste0("V", seq_len(ncol(.))))

  gc()
}

# Processing
{
  dep = train$target %>% as.numeric %>% log1p()
  train$target = NULL

  train %<>% mlr::removeConstantFeatures(obj = ., show.info = T)




  train_xgb = xgb.DMatrix(data = train %>% as.matrix, label = dep %>% as.numeric)
}

# Save test as sparse matrix
{
  #myform = as.formula(paste(" ~ ", paste(colnames(test), collapse= "+"),"-1"))
  #test_matrix = Matrix::sparse.model.matrix(object = myform, data = test)
  #saveRDS(test_matrix, 'test_matrix.rds')
  test_matrix = readRDS('test_matrix.rds')

  test_xgb = xgb.DMatrix(data = test_matrix)
}

# n-fold cv with random search on parameters
{
  best_param = list()
  best_seednumber = 1234
  best_score = 100
  best_score_index = 0

  for (iter in 1:15) {
    param = list(objective = "reg:linear",
                 eval_metric = "rmse",
                 booster = "gbtree",
                 max_depth = sample(12:15, 1),
                 eta = runif(1, .001, .05),
                 gamma = runif(1, 0.0, 0.2),
                 subsample = runif(1, .5, .7),
                 colsample_bytree = runif(1, .4, .7),
                 colsample_bylevel = runif(1, .7, .9),
                 min_child_weight = sample(8:10, 1),
                 lambda = runif(1, 0.9, 2))

    cv.nround = sample(seq.int(1000, 2000, 100), 1)
    cv.nfold = 4
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    mdcv = xgb.cv(data = train_xgb, params = param, stratified = T, print_every_n = 25,
                  nfold = cv.nfold, nrounds = cv.nround,
                  verbose = T, early_stopping_rounds = 50, maximize = F)

    max_score = min(mdcv$evaluation_log$test_rmse_mean)
    max_score_index = which.min(mdcv$evaluation_log$test_rmse_mean)

    if (max_score < best_score) {
      best_score = max_score
      best_score_index = max_score_index
      best_seednumber = seed.number
      best_param = param
    }
    print(paste0("iter is ", iter))
    gc()
  }

  nround = best_score_index
  set.seed(best_seednumber)
  saveRDS(nround, 'nround.rds')
  best_params_df = data.frame(best_param)
  saveRDS(best_params_df, 'best_params_df.rds')
}

# xgboost - main training and prediction
md = xgb.train(data = train_xgb,
               params = best_params_df,
               nrounds = nround,
               verbose = T,
               print_every_n = 50)


read_csv("sample_submission.csv") %>%
  mutate(target = expm1(predict(md, test_matrix))) %>%
  write_csv(paste0("santander_xgb", md$best_score, ".csv"))
