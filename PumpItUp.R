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
  train = train_test_encoded %>%
    filter(tt == "train")
  test = train_test_encoded %>%
    filter(tt == "test")

  train_sample = sample_frac(tbl = train, size = 0.75) %>%
    mutate(tt = NULL)
  train_sample_dep_actual = train_sample %>%
    left_join(., train_dep) %>%
    select(id, status_group)
  test_sample = anti_join(train, train_sample, by = "id") %>%
    mutate(tt = NULL)
  test_sample_dep_actual = test_sample %>%
    left_join(., train_dep) %>%
    select(id, status_group)

  train_data_matrix = xgb.DMatrix(data = as.matrix(train_sample),
                                  label = train_sample_dep_actual$status_group)
  test_data_matrix = xgb.DMatrix(data = as.matrix(test_sample),
                                 label = test_sample_dep_actual$status_group)

  numberOfClasses = length(unique(train_dep$status_group))

  xgb_params <- list("objective" = "multi:softprob",
                     "eval_metric" = "mlogloss",
                     "num_class" = numberOfClasses)
  nround = 100 # number of XGBoost rounds
  cv.nfold = 5

  # Fit cv.nfold * cv.nround XGB models and save OOF predictions
  cv_model = xgb.cv(params = xgb_params,
                     data = train_data_matrix,
                     nrounds = nround,
                     nfold = cv.nfold,
                     verbose = FALSE,
                     prediction = TRUE)
}
