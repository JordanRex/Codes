## WIDS 2018 DATATHON ##

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
      packages("data.table")
      packages("dplyr")
      packages("xgboost")
      packages("stringr")
      packages("magrittr")
      packages("rowr")
      packages("ranger")
      packages("mice")
      packages("qdapTools")
      packages("fastICA")
    })
}

# Processing
{
  test = read.csv('test.csv', header = T, stringsAsFactors = F, na.strings = c("NA", "", " ", "?"))

  train = read.csv('train.csv', header = T, stringsAsFactors = F, na.strings = c("NA", "", " ", "?"))

  ##test_train_cols = cbind.fill(data.frame(test = colnames(test)), data.frame(train = colnames(train)), fill = NA)

  ##wids_dd = readxl::read_xlsx('WiDS data dictionary v2.xlsx', sheet = 1)
}

# Pre-Processing
{
  ## remove columns deemed unnecessary because of some condition
  {
    # function to flag which columns are pointless without efficient NA imputation
    useless_cols_fn = function(x) {
      y = sum(is.na(x))

      if (y != length(x)) {
      y = if_else(y < ceiling(0.3 * length(x)), 1, 0)

      if ((y == 1) & (length(unique(x)) == 1)) {
        y = 0
      }

      if (sort(table(x), decreasing = T)[1] >= (0.99 * length(x))) {
        y = 0
      }
      }

      return(as.character(y))
    }

    # remove columns not satisfying the threshold
    train_useless_cols = train %>%
      summarise_all(useless_cols_fn)
    train_useless_cols_df = data.frame(col = colnames(train_useless_cols), flag = t(train_useless_cols), stringsAsFactors = F) %>%
      filter(flag == 1) %>%
      .$col

    train_1 = train %>% select(train_useless_cols_df)
    test_1 = test %<>% select(append(intersect(train_useless_cols_df, colnames(test)), "test_id"))
  }

  ## find out which columns are truly numeric
  {
    # function to find columns that are actually numeric, not stored as character
    cols_class_fn = function(x) {
      y = as.numeric(x)

      if (sum(is.na(y)) == length(x)) {
        x = 0
      } else
        if (is.numeric(x) == T) {
          x = 1
        } else
          if (sum(is.na(y)) > sum(is.na(x))) {
            x = 0
          } else
            if (sum(is.na(y)) == sum(is.na(x))) {
              x = 1
            }

      return(as.character(x))
    }

    # flag every column as 1 (numeric) and 0 (character)
    train_class = train_1 %>%
      summarise_all(cols_class_fn)
    train_class_df = data.frame(col = colnames(train_1), flag = t(train_class), stringsAsFactors = F) %>%
      filter(flag == 0) %>%
      .$col

    print(train_class_df)

    #View(train %>% select(train_class_df))
  }
  # 146 columns are character originally, after previous step there are none since they contain too many NAs

  rm(train, test)

  ## categorization of all remaining columns
  {
    train_df_summary = train_1 %>%
      select(-train_id)

    train_cols_df_1 = data.frame(COUNT_NULLS = colSums(is.na(train_df_summary))) %>%
      mutate(COLUMN = as.character(row.names(.)),
             COUNT_NULLS_RANK = if_else(COUNT_NULLS == 0, 0, if_else(COUNT_NULLS < 0.1 * nrow(train_df_summary), 1, 2)))

    train_cols_df_2 = summarise_all(train_df_summary, n_distinct) %>%
      melt %>%
      set_colnames(c("COLUMN", "COUNT_DISTINCT")) %>%
      mutate(COLUMN = as.character(COLUMN)) %>%
      left_join(., train_cols_df_1) %>%
      mutate(COUNT_DISTINCT_RANK = if_else(COUNT_NULLS_RANK == 0,
                                           if_else(COUNT_DISTINCT == 2, "binary",
                                                   if_else(COUNT_DISTINCT < 25, "categ",
                                                           if_else(COUNT_DISTINCT < 54, "rem_factors", "continuous"))),
                                           "continuous"))
  }

  ## missing value imputation
  # using MICE
  {
    # train_1 %<>% mutate_all(as.numeric)
    # test_1 %<>% mutate_all(as.numeric)
    #
    # mdpattern = md.pattern(train_1)
    #
    # train_impute = mice(train_1, m = 3, maxit = 10, method = "pmm")
  }

  ## treat NAs in the remaining columns - rough imputation
  {
  # mode function to fill NAs with mode of column
  mode_fn = function(x) {
    y = attr(sort(table(x), decreasing = T)[1], which = "name")
    x[is.na(x)] = y
    return(x)
  }

  train_2 = train_1 %>%
    group_by(AA3) %>%
    mutate_all(mode_fn) %>%
    ungroup
  test_2 = test_1 %>%
    group_by(AA3) %>%
    mutate_all(mode_fn) %>%
    ungroup
  }

  ## datatype treatment - 1
  {
    train_2 %<>% mutate_all(as.numeric)
    test_2 %<>% mutate_all(as.numeric)
  }
}

# Feature-Engineering
{
  # PCA
  {
    train_num_f = train_2 %>% mutate_all(.funs = as.numeric) %>% select(-train_id, -is_female)
    test_num_f = test_2 %>% mutate_all(.funs = as.numeric) %>% select(-test_id)

    pca_feats = prcomp(x = train_num_f, retx = T, center = T, tol = 0, scale. = T)
    expl.var = round(pca_feats$sdev^2/sum(pca_feats$sdev^2)*100)
    # top 2 components itself explains the whole variance

    # scree plot
     {
      std_dev = pca_feats$sdev
      pca_var = std_dev^2
      prop_var = pca_var/sum(pca_var)
      plot(cumsum(prop_var), xlab = "PC", ylab = "Prop Var Exp", type = "b")
     }

    pca_feats_to_be_added = data.frame(pca_feats$x[, 1:50])
    train_2 %<>% cbind(., pca_feats_to_be_added)

    test_pca_pred = data.frame(predict(pca_feats, newdata = test_num_f) %>% .[, 1:50])
    test_2 %<>% cbind(., test_pca_pred)

    rm(pca_feats, pca_feats_to_be_added, pca_var, expl.var, prop_var, std_dev, train_useless_cols, train_useless_cols_df, train_class_df, train_num_f, test_num_f, test_pca_pred)
  }

  # ICA
  {
    train_num_f = train_2 %>% mutate_all(.funs = as.numeric) %>% select(-train_id, -is_female) %>% scale
    test_num_f = test_2 %>% mutate_all(.funs = as.numeric) %>% select(-test_id)

    train_ica = fastICA(train_num_f, n.comp = 50, maxit = 50, verbose = T, tol = 1e-04)

    train_2 %<>% cbind(., train_ica$S %>% data.frame %>% set_colnames(paste0("ica_", 1:50)))

    train_ica_df = as.matrix(test_num_f) %*% train_ica$K %*% train_ica$W %>%
      data.frame %>%
      set_colnames(paste0("ica_", 1:50))
    test_2 %<>% cbind(., train_ica_df)

    rm(train_num_f, test_num_f, train_ica_df, train_ica)
  }

  # DEVIATION_ENCODING
  {
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
                      paste0(temp_col, "_median"))

        temp = train_char_data[, i, drop = F] %>%
          cbind(., train_num_data) %>%
          group_by_at(vars(-matches("is_female"))) %>%
          mutate(mean = mean(is_female),
                 sd = sd(is_female),
                 median = median(is_female)) %>%
          ungroup %>%
          select(temp_col, mean, sd, median) %>%
          set_colnames(temp_cols) %>%
          distinct %>%
          mutate_all(as.numeric)

        train_2 <<- left_join(train_2, temp)
        test_2 <<- left_join(test_2, temp)
      }

      return(print("train_2 and test_2 have been generated"))
    }

    categ_variables = train_cols_df_2 %>%
      filter(COUNT_DISTINCT_RANK == "categ" | COUNT_DISTINCT_RANK == "rem_factors") %>%
      .$COLUMN

    categtoDeviationenc(char_data = train_2 %>% select(categ_variables), num_data = train_2 %>% select(is_female))
    }
}

# Feature-Selection
{
  train_xgb = xgb.DMatrix(data = data.matrix(train_2 %>% select(-is_female, -train_id)), label = data.matrix(as.numeric(train_2[,'is_female'])))

  xgb_feat_selection_model = xgboost(data = train_xgb,
                                     nrounds = 500,
                                     eta = 0.01,
                                     objective = "binary:logistic",
                                     verbose = 1,
                                     max_depth = 12,
                                     print_every_n = 50,
                                     early_stopping_rounds = 5)

  xgb_importance = data.table(xgboost::xgb.importance(feature_names = setdiff(colnames(train_2), c("is_female", "train_id")),
                                                      model = xgb_feat_selection_model))

  Importance_table = data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
    mutate(Rank = dense_rank(desc(Importance))) %>%
    filter(Rank <= 350)
  colnames_features_brands = as.vector(Importance_table$Feature)


  ## subset for the required columns alone
  train_3 = train_2 %>% select(append(colnames_features_brands, c("is_female", "train_id")))
  test_3 = test_2 %>% select(append(colnames_features_brands, c("test_id")))

  rm(xgb_feat_selection_model, xgb_importance, train_df_summary)
}

# Processing - Ranger
{
  # datatype treatment
  {
    # function to make columns factors if unique values are less than 50, else numeric
    factor_fn = function(x) {
      unique_temp = length(unique(x))

      if (unique_temp <= 53) {
        x = as.factor(x)
      } else {
        x = as.numeric(x)
      }
    }

    train_3 %<>%
      mutate_all(factor_fn) %>%
      mutate(is_female = as.numeric(as.character(is_female)))

    test_3 %<>%
      mutate_all(factor_fn)
  }
}

# Modelling - Ranger
{
  # the final treated datasets are train_3 and test_3
  # creating their equivalents without the id columns and a dummy dependant in test for predictions
  train_4 = train_3 %>%
    mutate(train_id = NULL)
  test_4 = test_3 %>%
    mutate(test_id = NULL,
           is_female = 0)

  test_rows = sample(nrow(train_3), size = ceiling(0.6 * nrow(train_3)), replace = F)
  train_temp = train_4[test_rows, ] %>% mutate_if(is.factor, droplevels)
  test_temp = train_4[-test_rows, ] %>%
    mutate(is_female = 0) %>% mutate_if(is.factor, droplevels)

  actual_train_test = train_3[-test_rows, ] %>% select(test_id = train_id, is_female_act = is_female)

  accuracy_grid_1 = data.frame(model = "", accuracy = "", num_trees = "", mtry = "", stringsAsFactors = F)

  ranger_grid = expand.grid(num_trees = c(350, 500, 1000),
                            mtry = c(ceiling(ncol(train_3)/3), ceiling(ncol(train_3)/2)))

  output_grid = data.frame(ID = actual_train_test$test_id)

  for (i in 1:nrow(ranger_grid)) {
    num_trees_temp = ranger_grid[i, 1]
    mtry_temp = ranger_grid[i, 2]

    model = ranger::ranger(is_female ~ ., num.trees = num_trees_temp, mtry = mtry_temp, splitrule = "gini", data = train_temp, probability = T, respect.unordered.factors = T)

    output = data.frame(predict(model, test_temp)[1]) %>%
      select(predictions.1) %>%
      set_colnames("is_female")
    output = cbind(data.frame(test_id = actual_train_test$test_id), output) %>%
      left_join(., actual_train_test) %>%
      mutate(prediction = if_else(is_female > 0.5 & is_female_act == 1, 1,
                                  if_else(is_female < 0.5 & is_female_act == 0, 1, 0)))

    accuracy = sum(output$prediction)/nrow(output)

    accuracy_grid_1[i, "model"] = i
    accuracy_grid_1[i, "accuracy"] = accuracy
    accuracy_grid_1[i, "num_trees"] = num_trees_temp
    accuracy_grid_1[i, "mtry"] = mtry_temp

    output_grid %<>% left_join(., output %>% select(ID = test_id, is_female), by = "ID")

    print(paste0("Model ", i, " is running"))
  }

  # Actual modelling for ranger
  {
    top_acc = accuracy_grid_1 %>%
      top_n(accuracy, n = 3) %>%
      select(num_trees, mtry) %>%
      mutate(model = row_number())

    output_grid = data.frame(ID = test_3$test_id)

    for (i in 1:nrow(top_acc)) {
      num_trees_temp = as.numeric(top_acc[i, 1])
      mtry_temp = as.numeric(top_acc[i, 2])

      # train_3
      # test_3

      model = ranger::ranger(is_female ~ ., num.trees = num_trees_temp, mtry = mtry_temp, splitrule = "gini", data = train_4, probability = T, respect.unordered.factors = T)

      output = data.frame(predict(model, test_4)[1]) %>%
        select(predictions.1) %>%
        set_colnames("is_female")
      output = cbind(data.frame(test_id = test_3$test_id), output)

      output_grid %<>% left_join(., output %>% select(ID = test_id, is_female), by = "ID")
    }

    # aggregate and generate final output
    {
      is_female_cols = grep(pattern = "is_female", x = colnames(output_grid), value = T)

      final_output = output_grid %>%
        select(is_female_cols) %>%
        as.matrix %>%
        rowMeans(.[,])

      final_output = data.frame(test_id = test_3$test_id, is_female = final_output)
    }
  }

  write.csv(final_output, "output_1.csv", row.names = F)
}

# Modelling - XGBOOST
{
  # the input datasets are train_3 and test_3
  # train_3
  # test_3

  test_rows = sample(nrow(train_3), size = ceiling(0.6 * nrow(train_3)), replace = F)
  train_temp = xgb.DMatrix(data = data.matrix(train_3[test_rows, ] %>% select(-is_female, -train_id)), label = data.matrix(as.numeric(train_3[test_rows, 'is_female'])))
  test_temp = train_3[-test_rows, ] %>%
    select(-is_female, -train_id) %>%
    as.matrix

  actual_train_test = train_3[-test_rows, ] %>% select(test_id = train_id, is_female_act = is_female)

  accuracy_grid = data.frame(model = "", accuracy = "", nrounds = "", eta = "", max_depth = "", stringsAsFactors = F)

  xgboost_grid = expand.grid(nrounds = c(500, 1000, 1500),
                             eta = c(0.05, 0.1, 0.3),
                             max_depth = c(7, 13))

  output_grid = data.frame(ID = actual_train_test$test_id)

  for (i in 1:nrow(xgboost_grid)) {
    nrounds_temp = xgboost_grid[i, 1]
    eta_temp = xgboost_grid[i, 2]
    max_depth_temp = xgboost_grid[i, 3]

    model = xgboost(data = train_temp, nrounds = nrounds_temp, eta = eta_temp, max_depth = max_depth_temp, verbose = 1, print_every_n = 50, objective = "binary:logistic", subsample = 0.7, colsample_bytree = 0.7)

    output = predict(model, test_temp) %>%
      data.frame %>%
      set_colnames("is_female")
    output = cbind(data.frame(test_id = actual_train_test$test_id), output) %>%
      left_join(., actual_train_test) %>%
      mutate(prediction = if_else(is_female > 0.5 & is_female_act == 1, 1,
                                  if_else(is_female < 0.5 & is_female_act == 0, 1, 0)))

    accuracy = sum(output$prediction)/nrow(output)

    accuracy_grid[i, "model"] = i
    accuracy_grid[i, "accuracy"] = accuracy
    accuracy_grid[i, "nrounds"] = nrounds_temp
    accuracy_grid[i, "eta"] = eta_temp
    accuracy_grid[i, "max_depth"] = max_depth_temp

    output_grid %<>% left_join(., output %>% select(ID = test_id, is_female), by = "ID")

    print(paste0("Model ", i, " is running"))
  }

  # Actual modelling for xgboost
  {
    top_acc = accuracy_grid %>%
      top_n(accuracy, n = 5) %>%
      select(nrounds, eta, max_depth) %>%
      mutate(model = row_number())

    output_grid = data.frame(ID = test_3$test_id)

    train_xgb = xgb.DMatrix(data = data.matrix(train_3 %>% select(-is_female, -train_id)), label = data.matrix(as.numeric(train_3[, 'is_female'])))
    test_xgb = as.matrix(test_3 %>% select(-test_id))

    for (i in 1:nrow(top_acc)) {
      nrounds_temp = as.numeric(top_acc[i, 1])
      eta_temp = as.numeric(top_acc[i, 2])
      max_depth_temp = as.numeric(top_acc[i, 3])

      # train_3
      # test_3

      model = xgboost(data = train_xgb, nrounds = nrounds_temp, eta = eta_temp, max_depth = max_depth_temp, verbose = 1, print_every_n = 50, objective = "binary:logistic", subsample = 0.7, colsample_bytree = 0.7)

      output = data.frame(predict(model, test_xgb)) %>%
        set_colnames("is_female")
      output = cbind(data.frame(test_id = test_3$test_id), output)

      output_grid %<>% left_join(., output %>% select(ID = test_id, is_female), by = "ID")
    }

    # aggregate and generate final output
    {
      is_female_cols = grep(pattern = "is_female", x = colnames(output_grid), value = T)

      final_output = output_grid %>%
        select(is_female_cols) %>%
        as.matrix %>%
        rowMeans(.[,])

      final_output = data.frame(test_id = test_3$test_id, is_female = final_output)
    }
  }

  write.csv(final_output, "output_2.csv", row.names = F)
}

# Ensemble
{
  output_1 = read.csv('output_1.csv', header = T, stringsAsFactors = F)
  output_2 = read.csv('output_2.csv', header = T, stringsAsFactors = F)
}
