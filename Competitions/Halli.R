## Halliburton RFP ##

## classification
{
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
        packages("modeest")
        packages("xgboost")
        packages("stringr")
        packages("rowr")
        packages("caret")
        # packages("ranger")
        # packages("mice")
        # packages("qdapTools")
        packages("fastICA")
        # packages("splitstackshape")
        # packages("text2vec")
        # packages("rpart")
        packages("caret")
        packages("readr")
        packages("DataExplorer")
        packages("data.table")
        packages("magrittr")
        packages("dplyr")
      })
  }

  # Input reading
  {
    train = read_csv('Training_Data_C.csv')
    test = read_csv('Validation_Data_C.csv')

    ## combining the train and test
    train_test = bind_rows(train %>% mutate(tt = "train"),
                           test %>% mutate(tt = "test"))

    # view the nulls present
    #View(data.frame(col = colSums(is.na(train_test))))
  }

  # EDA
  {
    # DataExplorer::create_report(train)

    ## notes from above resuls
    # columns level_1, level_2, region, criteria_1, criteria_2, type_1, length_1 are discrete (can be used for deviation encoding)
    # the depth_1, depth_2, depth_3 features have a lot of mutual information with the dependant feat, can be used for feat engineering
    # depth_1 and depth_3 have high correlation
  }

  # Pre-Processing
  {
    # 1. basic filtering
    {
    # select only columns with atleast 2 unique values
    train %<>% select_if(function(x) length(unique(x)) > 1)
    test %<>% select(colnames(train))
    }

    # 2. missing value imputation
    {
    # missing value cols
    miss_cols = data.frame(miss = colSums(is.na(train))) %>%
      mutate(col = row.names(.)) %>%
      filter(miss > 0) %>%
      .$col

    # treat each of the missing value columns
    fn_miss_treatment_categ = function(x, y = "unknown") {
      if (is.numeric(x)) x[which(is.na(x))] = mean(x, na.rm = T)
      if (!is.numeric(x)) x[which(is.na(x))] = y
      return(x)
    }

    train %<>% mutate_if(.predicate = colnames(.) %in% miss_cols,
                               .funs = fn_miss_treatment_categ)
    }

    # 3. summarizing for feature engineering
    {
      train_summary = train %>%
        summarise_all(n_distinct) %>%
        melt %>%
        set_colnames(c("column", "count")) %>%
        mutate(categ_flag = if_else(count < 10, 1, 0),
               bucket_feat_flag = if_else(count >= 100, 1, 0)) %>%
        filter(!column %in% c("Job_Status", "ID"))
    }
  }

  # Feature engineering
  {
    # deviation encoding (creating mean,sd,median,skew features of the different levels/labels in categorical features)
    {
      dev_encod_feats = train_summary %>% filter(categ_flag == 1) %>% .$column %>% as.character

      # encode the dep variable as well
      dep_encoded = CatEncoders::LabelEncoder.fit(train$Job_Status)
      train$Job_Status = CatEncoders::transform(dep_encoded, train$Job_Status) - 1

      # function to compute the deviation encoded features
      categtoDeviationenc = function(char_data,
                                     num_data)
      {
        train_char_data = char_data %>% data.frame() %>% mutate_all(as.character)
        train_num_data = num_data %>% data.frame() %>% mutate_all(as.character) %>% mutate_all(as.numeric)

        for (i in 1:ncol(train_char_data)) {
          temp_col = colnames(train_char_data[, i, drop = F])

          temp_cols = c(temp_col,
                        paste0(temp_col, "_mean"),
                        paste0(temp_col, "_sd"),
                        paste0(temp_col, "_median"),
                        paste0(temp_col, "_mode"),
                        paste0(temp_col, "_skew"))

          temp = train_char_data[, i, drop = F] %>%
            cbind(., train_num_data) %>%
            group_by_at(vars(-matches("Job_Status"))) %>%
            mutate(mean = mean(Job_Status),
                   sd = sd(Job_Status),
                   median = median(Job_Status),
                   mode = mlv(Job_Status, method = "mfv")[['M']],
                   skew = mlv(Job_Status, method = "mfv")[['skewness']]) %>%
            ungroup %>%
            select(temp_col, mean, sd, median, mode, skew) %>%
            set_colnames(temp_cols) %>%
            distinct %>%
            mutate_all(as.numeric)

          train <<- left_join(train, temp)
          test <<- left_join(test, temp)
        }

        return(print("train and test have been generated"))
      }

      categtoDeviationenc(char_data = train %>% select(dev_encod_feats), num_data = train %>% select(Job_Status))
    }

    # remove useless cols generated (with near zero variance)
    {
      useless_cols = caret::nearZeroVar(x = train, freqCut = 10, uniqueCut = 20, names = T)

      train %<>% select(setdiff(colnames(.), useless_cols))
      test %<>% select(colnames(train))
    }

    # bucket feats (binning features with rpart decision trees for continuous ordered features)
    {
      bucket_feats = train_summary %>% filter(bucket_feat_flag == 1) %>% .$column %>% as.character

      feateng_bucket_fn = function(train, col) {
        temp = train %>% select_(indep = col, "Job_Status")
        temp_test = test %>% select_(indep = col)
        temp_extreme_cuts = append(temp$indep, temp_test$indep) %>% unique %>% range(.) #%>% .[. %in% c(max(.), min(.))]

        temp_model = rpart::rpart(data = temp, formula = Job_Status ~ indep)
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

      for (i in 1:length(bucket_feats)) {
        feateng_bucket_fn(train, bucket_feats[i])
      }
    }

    rm(train_summary, train_test)

    # PCA components
    {
      train_num = train %>%
        mutate_all(.funs = as.numeric) %>% select(-ID, -Job_Status)
      test_num = test %>%
        mutate_all(.funs = as.numeric) %>% select(-ID, -Job_Status)

      pca_feats = prcomp(x = train_num, retx = T, center = T, tol = 0, scale. = T)
      expl.var = round(pca_feats$sdev^2/sum(pca_feats$sdev^2)*100)
      # top 2 components itself explains the whole variance

      # scree plot
      {
        std_dev = pca_feats$sdev
        pca_var = std_dev^2
        prop_var = pca_var/sum(pca_var)
        plot(cumsum(prop_var), xlab = "PC", ylab = "Prop Var Exp", type = "b")
      }
      # ~20 is the threshold

      pca_feats_to_be_added = data.frame(pca_feats$x[, 1:20])
      test_pca_pred = data.frame(predict(pca_feats, newdata = test_num) %>% .[, 1:20])

      train %<>% bind_cols(., pca_feats_to_be_added)
      test %<>% bind_cols(., test_pca_pred)

      rm(pca_feats, pca_feats_to_be_added, pca_var, expl.var, prop_var, std_dev, test_pca_pred)
    }

    # ICA
    {
      train_num %<>% scale
      test_num %<>% scale

      train_ica = fastICA(train_num, n.comp = 15, maxit = 100, verbose = T, tol = 1e-04, method = "C")
      train %<>% bind_cols(., train_ica$S %>% data.frame %>% set_colnames(paste0("ica_", 1:15)))

      test_ica_df = as.matrix(test_num) %*% train_ica$K %*% train_ica$W %>%
        data.frame %>%
        set_colnames(paste0("ica_", 1:15))
      test %<>% bind_cols(., test_ica_df)

      rm(train_num, test_num, test_ica_df, train_ica)
    }
  }

  # xgboost
  {
    gc()

    # train sampling (80-20)
    {
    train_sample = sample_frac(tbl = train, size = 0.8)
    train_sample_dep_actual = train_sample %>%
      select(ID, Job_Status)

    test_sample = anti_join(train, train_sample, by = "ID")
    test_sample_dep_actual = test_sample %>%
      select(ID, Job_Status)

    train_sample %<>% select(-ID, -Job_Status)
    test_sample %<>% select(-ID, -Job_Status)

    test_ids = test_sample_dep_actual %>%
      .$ID
    }

    train_data_matrix = xgb.DMatrix(data = as.matrix(train_sample),
                                    label = as.matrix(train_sample_dep_actual$Job_Status))
    test_data_matrix = xgb.DMatrix(data = as.matrix(test_sample))

    numberOfClasses = length(unique(train_sample_dep_actual$Job_Status))

    # n-fold cv with random search on parameters
    {
      best_param = list()
      best_seednumber = 1234
      best_logloss = Inf
      best_logloss_index = 0

      result_matrix = data.frame(model = 0,
                                 cv_mlogloss = 0,
                                 test_accuracy = 0)

      for (iter in 1:25) {
        param = list(objective = "multi:softprob",
                     eval_metric = "mlogloss",
                     num_class = numberOfClasses,
                     max_depth = sample(3:7, 1),
                     eta = runif(1, .001, .2),
                     gamma = runif(1, 0.0, 0.2),
                     subsample = runif(1, .6, .9),
                     colsample_bytree = runif(1, .5, .9),
                     min_child_weight = sample(5:10, 1),
                     max_delta_step = sample(1:10, 1))
        cv.nround = sample(seq.int(25, 300, 25), 1)
        cv.nfold = 5
        seed.number = sample.int(10000, 1)[[1]]
        set.seed(seed.number)
        mdcv = xgb.cv(data = train_data_matrix, params = param, stratified = T, print_every_n = 50,
                       nfold = cv.nfold, nrounds = cv.nround,
                       verbose = T, early_stopping_rounds = 10, maximize = FALSE)

        min_logloss = min(mdcv$evaluation_log$test_mlogloss_mean)
        min_logloss_index = which.min(mdcv$evaluation_log$test_mlogloss_mean)

        #plot(log(mdcv$evaluation_log$test_mlogloss_mean), type = 'l')

        # testing it on the seperate validation set as well (no cv folds, so entire train is taken, simply to know the real life predictive power of the model. Not used for tuning, only cv is used for the same)
        {
          xgb_model = xgb.train(data = train_data_matrix, params = param, nrounds = min_logloss_index, print_every_n = 50)

          output = predict(xgb_model, test_data_matrix, reshape = T) %>%
            data.frame %>%
            mutate(pred = max.col(., ties.method = "last") - 1) %>%
            cbind(data.frame(ID = test_ids), .) %>%
            left_join(., test_sample_dep_actual) %>% select(ID, pred, Job_Status) %>%
            mutate(acc = if_else(pred == Job_Status, 1, 0))

          # filling up the result matrix
          result_matrix[iter, "model"] = iter
          result_matrix[iter, "cv_mlogloss"] = min_logloss
          result_matrix[iter, "test_accuracy"] = 100*sum(output$acc)/nrow(output)
        }

        if (min_logloss < best_logloss) {
          best_logloss = min_logloss
          best_logloss_index = min_logloss_index
          best_seednumber = seed.number
          best_param = param
        }
        print(paste0("iter is ", iter))
      }

      nround = best_logloss_index
      set.seed(best_seednumber)
    }

    best_params_df = data.frame(best_param) %>%
      cbind(., data.frame(nround = nround))
    saveRDS(best_params_df, 'best_params_df.rds')

    train_full_matrix = xgb.DMatrix(data = as.matrix(train %>% select(-ID, -Job_Status)),
                                    label = as.matrix(train$Job_Status))
    test_full_matrix = xgb.DMatrix(data = as.matrix(test %>% select(-ID, -Job_Status)))

    xgb_params = best_param
    nround = nround
    cv.nfold = 5

    # Fit cv.nfold * cv.nround XGB models and save OOF predictions for just seeing the model performance (not that necessary, testing set validation served the same purpose)
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
             label = train$Job_Status,
             pred_flag = if_else(max_prob == label, 1, 0))

    md = xgb.train(data = train_full_matrix,
                   params = best_param,
                   nrounds = nround,
                   verbose = T,
                   print_every_n = 50)

    test_ids = test$ID
    test_final_predictions = predict(md, test_full_matrix, reshape = T)
    output = data.frame(test_final_predictions) %>%
      mutate(job_status = max.col(., ties.method = "last")) %>%
      cbind(data.frame(ID = test_ids), .) %>%
      select(ID, job_status)
    output$job_status = CatEncoders::inverse.transform(dep_encoded, output$job_status)

    fwrite(output, 'output.csv')
  }
}
