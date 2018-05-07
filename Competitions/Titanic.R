## Titanic ##

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
      packages("splitstackshape")
      packages("text2vec")
      packages("rpart")
    })
}

# Processing
{
  test = read.csv('test.csv', header = T, stringsAsFactors = F, na.strings = c("NA", "", " ", "?"))

  train = read.csv('train.csv', header = T, stringsAsFactors = F, na.strings = c("NA", "", " ", "?"))

  sample_sub = read.csv('gender_submission.csv', header = T, stringsAsFactors = F, na.strings = c("NA", "", " ", "?"))
}

# Pre-Processing
{
  ## combining the train and test and fixing the missing values
  train_test = bind_rows(train %>% mutate(tt = "train"),
                         test %>% mutate(tt = "test"))

  # view the nulls present
  #View(data.frame(col = colSums(is.na(train_test))))
}

# Feature-Engineering
{
  # creating some base natural features
  {
    train_test_new = train_test %>%
      mutate(alias_name = gsub(str_extract(Name, pattern = "\\(.*\\)"), pattern = "\\(|\\)", replacement = ""),
             alias_name_flag = if_else(is.na(alias_name), 0, 1),
             Name = gsub(x = Name, pattern = "\\(.*\\)", replacement = ""),
             cabin_count = str_count(Cabin, pattern = " ") + 1) %>%
      cSplit(., splitCols = "Name", direction = "wide", sep = ",", type.convert = F, drop = F) %>%
      cSplit(., splitCols = "Name_2", direction = "wide", sep = ".", type.convert = F, drop = F) %>%
      mutate(Pclass_low_med = if_else(Pclass < 3, 1, 0),
             Pclass_med_high = if_else(Pclass > 1, 1, 0),
             tot_family = SibSp + Parch,
             cabin_flag = if_else(is.na(Cabin), 0, 1)) %>%
      rowwise %>%
      mutate(Ticket = gsub(Ticket, pattern = ".", replacement = "", fixed = T)) %>%
      ungroup %>%
      cSplit(., splitCols = "Ticket", direction = "wide", sep = " ", type.convert = F, drop = F) %>%
      mutate(ticket_no = if_else(is.na(Ticket_3),
                                 if_else(is.na(Ticket_2),
                                         Ticket_1,
                                         Ticket_2),
                                 Ticket_3)) %>%
      mutate_if(is.factor, as.character) %>%
      ungroup
  }

  # NLP features
  {
    nlp_feats_df = train_test_new %>%
      filter(tt == "train") %>%
      mutate(name_full_nlp = trimws(gsub(paste(Name_2_2, Name_1, alias_name, sep = " "), pattern = "NA", replacement = "", fixed = T), which = "both")) %>%
      select(PassengerId, tt, name_full_nlp)

    prep_fun = tolower
    tok_fun = word_tokenizer

    it_train = itoken(nlp_feats_df$name_full_nlp,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = nlp_feats_df$PassengerId,
                      progressbar = FALSE)

    vocab = create_vocabulary(it_train)
    vectorizer = vocab_vectorizer(vocab)
    dtm_train = create_dtm(it_train, vectorizer) %>%
      as.matrix

    dtm_freq = data.frame(count = colSums(dtm_train)) %>%
      mutate(token = row.names(.)) %>%
      top_n(., count, n = 3) %>%
      .$token

    dtm_train %<>%
      data.frame %>%
      select(dtm_freq)

    nlp_feats_df_test = train_test_new %>%
      filter(tt == "test") %>%
      mutate(name_full_nlp = trimws(gsub(paste(Name_2_2, Name_1, alias_name, sep = " "), pattern = "NA", replacement = "", fixed = T), which = "both")) %>%
      select(PassengerId, tt, name_full_nlp)

    it_test = itoken(nlp_feats_df_test$name_full_nlp,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = nlp_feats_df_test$PassengerId,
                      progressbar = FALSE)

    vocab = create_vocabulary(it_test)
    vectorizer = vocab_vectorizer(vocab)
    dtm_test = create_dtm(it_test, vectorizer) %>%
      as.matrix %>% as.data.frame %>%
      select(dtm_freq)

    dtm_full = bind_rows(dtm_train, dtm_test)

    train_test_new %<>% cbind(., dtm_full)

    rm(dtm_test, dtm_train, dtm_freq, nlp_feats_df, nlp_feats_df_test, vocab, it_test, it_train, dtm_full)
  }

  # some treatment
  {
    useless_cols = c("Name", "alias_name", "Name_1", "Name_2", "Name_2_2", "Ticket")
    cols_to_retain = setdiff(colnames(train_test_new), useless_cols)
    train_test_new %<>% select(cols_to_retain)

    # mode function to fill NAs with mode of column
    mode_fn = function(x) {
      y = attr(sort(table(x), decreasing = T)[1], which = "name")
      x[is.na(x)] = y
      return(x)
    }

    train_test_new_v2 = train_test_new %>%
      rowwise %>%
      mutate(Ticket_1 = gsub(Ticket_1, pattern = ticket_no, replacement = ""),
             Ticket_2 = gsub(Ticket_2, pattern = ticket_no, replacement = "")) %>%
      ungroup %>%
      mutate(Ticket_1 = toupper(gsub(paste0(Ticket_1, Ticket_2), pattern = "/", replacement = "", fixed = T)),
             Ticket_categ = Ticket_1) %>%
      mutate(Ticket_categ = if_else(Ticket_categ %in% c("CA", "PC"), Ticket_categ, "NA")) %>%
      rename(title = Name_2_1) %>%
      select(-Ticket_1, -Ticket_2, -Ticket_3) %>%
      group_by(Sex, tt, Pclass) %>%
      mutate(Age = if_else(is.na(Age), mean(Age, na.rm = T), Age),
             Fare = if_else(is.na(Fare), mean(Fare, na.rm = T), Fare),
             Embarked = if_else(is.na(Embarked), mode_fn(Embarked), Embarked),
             Cabin = if_else(is.na(Cabin), "None", Cabin),
             cabin_count = if_else(is.na(cabin_count), 0, cabin_count))

    # view the nulls present
    View(data.frame(col = colSums(is.na(train_test_new_v2))))
  }

  # some random features
  {
    train = train_test_new_v2 %>% filter(tt == "train")

    # for age
    age_bins_model = rpart(data = train, formula = Survived ~ Age)
    age_bins_model
    # splits happened at 6.5 and 26.5

    # for fare
    fare_bins_model = rpart(data = train, formula = Survived ~ Fare)
    fare_bins_model
    # splits happened at 10.5, 52.5, 69.5, 74.5

    # for fare
    train %<>%
      mutate(ticket_no = as.numeric(ticket_no),
             ticket_no = if_else(is.na(ticket_no), 0, ticket_no))
    ticketno_bins_model = rpart(data = train, formula = Survived ~ ticket_no)
    ticketno_bins_model
    # splits happened at 2650, 8855, 17590, 312993

    train_test_new_v2 %<>%
      mutate(age_bins = if_else(Age < 6.5, 0,
                                if_else(Age < 26.5, 1, 2)),
             fare_bins = if_else(Fare < 10.5, 0,
                                 if_else(Fare < 52.5, 1,
                                         if_else(Fare < 69.5, 2,
                                                 if_else(Fare < 74.5, 3, 4)))),
             ticket_bins = if_else(ticket_no < 2650, 0,
                                   if_else(ticket_no < 8855, 1,
                                           if_else(ticket_no < 17590, 2,
                                                   if_else(ticket_no < 312993, 3, 4))))) %>%
      ungroup
  }

  # create the fully numeric matrix
  {
    # label encode and store all character columns for immediate modelling
    categ_cols = train_test_new_v2 %>% select_if(is.character) %>% colnames
    rem_cols = setdiff(colnames(train_test_new_v2), categ_cols)

    train_categ = train_test_new_v2 %>% select(categ_cols) %>% filter(tt == "train")
    test_categ = train_test_new_v2 %>% select(categ_cols) %>% filter(tt == "test")
    train_test_categ = bind_rows(train_categ, test_categ)

    train_rest = train_test_new_v2 %>% select(rem_cols) %>% filter(tt == "train")
    test_rest = train_test_new_v2 %>% select(rem_cols) %>% filter(tt == "test")
    train_test_rest = bind_rows(train_rest, test_rest)

    fn_labelencoder_df = function(x) {
      encoded_list = list()
      temp_df = x
      temp_df_cols = colnames(temp_df)

      for (i in 1:length(temp_df_cols)) {
        encoded_list[[i]] = CatEncoders::LabelEncoder.fit(temp_df[, i, drop = T])
        x[, i] = CatEncoders::transform(encoded_list[[i]], x[, i, drop = T])
      }
      return(x)
    }

    train_test_encoded = fn_labelencoder_df(train_test_categ) %>%
      mutate(tt = NULL) %>%
      bind_cols(., train_test_rest)
   }

  # PCA
  {
    train_num_f = train_test_new_v2 %>%
      filter(tt == "train") %>%
      mutate_all(.funs = as.numeric) %>% select(-PassengerId, -Survived, -tt)
    test_num_f = test_1 %>% mutate_all(.funs = as.numeric) %>% select(-test_id)

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

# the nnet segment
{

## Partitioning Data into Train and Test datasets in 70:30
library(caret)
set.seed(1234567)
train1<-createDataPartition(bank_full$subscribed,p=0.7,list=FALSE)
train<-bank_full[train1,]
test<-bank_full[-train1,]

## CLASSIFICATION USING NEURAL NETWORK

library(nnet)

## 1. Fit a Single Hidden Layer Neural Network using Least Squares
train.nnet<-nnet(subscribed~.,train,size=3,rang=0.07,Hess=FALSE,decay=15e-4,maxit=250)
## Use TEST data for testing the trained model
test.nnet<-predict(train.nnet,test,type=("class"))
## MisClassification Confusion Matrix
table(test$subscribed,test.nnet)
## One can maximize the Accuracy by changing the "size" while training the neural network. SIZE refers to the number of nodes in the hidden layer.
which.is.max(test.nnet)  ## To Fine which row break ties at random (Maximum position in vector)

##2. Use Multinomial Log Linear models using Neural Networks
train.mlln<-multinom(subscribed~.,train)
##USe TEST data for testing the trained model
test.mlln<-predict(train.mlln,test)
##Misclassification or Confusion Matrix
table(test$subscribed,test.mlln)
##3. Training Neural Network Using BACK PROPOGATION
install.packages("neuralnet")
library(neuralnet)
## Check for all Input Independent Variables to be Integer or Numeric or complex matrix or vector arguments. If they are not any one of these, then tranform them accordingly
str(train)
str(test)
## It can be observed that all are either integer or factor. Now these factors have to be transformed to numeric.
## One cannot use directly as.numeric() to convert factors to numeric as it has limitations.
## First, Lets convert factors having character levels to numeric levels
str(bank_full)
bank_full_transform<-bank_full
bank_full_transform$marital=factor(bank_full_transform$marital,levels=c("single","married","divorced"),labels=c(1,2,3))
str(bank_full_transform)
## Now convert these numerical factors into numeric
bank_full_transform$subscribed<-as.numeric(as.character(bank_full_transform$subscribed))

str(bank_full_transform)
## Now all the variables are wither intergers or numeric
## Now we shall partition the data into train and test data
library(caret)
set.seed(1234567)
train2<-createDataPartition(bank_full_transform$subscribed,p=0.7,list=FALSE)
trainnew<-bank_full_transform[train2,]
testnew<-bank_full_transform[-train2,]
str(trainnew)
str(testnew)
## Now lets run the neuralnet model on Train dataset
trainnew.nnbp<-neuralnet(subscribed~age+balance+lastday+lastduration+numcontacts+pdays+pcontacts+marital+education+housingloan+personalloan+lastmonth+poutcome+lastcommtype,data=bank_full_transform,hidden=5,threshold=0.01,err.fct="sse",linear.output=FALSE,likelihood=TRUE,stepmax=1e+05,rep=1,startweights=NULL,learningrate.limit=list(0.1,1.5),learningrate.factor=list(minus=0.5,plus=1.5),learningrate=0.5,lifesign="minimal",lifesign.step=1000,algorithm="backprop",act.fct="logistic",exclude=NULL,constant.weights=NULL)
## Here, Back Propogation Algorithm has been used. One can also use rprop+ (resilient BP with weight backtracking),rprop- (resilient BP without weight backtracking), "sag and "slr" as modified global convergent algorithm
## Accordingly the accuracy can be checked for each algorithm
summary(train.nnbp)
## Plot method for the genralised weights wrt specific covariate (independent variable) and response target variable
gwplot(trainnew.nnbp,selected.covariate="balance")
##(Smoother the Curve- Better is the model prediction)
## Plot the trained Neural Network
plot(trainnew.nnbp,rep="best")
## To check your prediction accuracy of training model
prediction(trainnew.nnbp)
print(trainnew.nnbp)
## Now use the TEST data set to test the trained model
## Make sure that target column in removed from the test data set
columns=c("age","job","marital","education","balance","housingloan","personalloan","lastcommtype","lastday","lastmonth","lastduration","numcontacts","pdays","poutcome")
testnew2<-subset(testnew,select=columns)
testnew.nnbp<-compute(trainnew.nnbp,testnew2,rep=1)
## MisClassification Confusion Matrix
table(testnew$subscribed,testnew.nnbp$net.result)
cbind(testnew$subscribed,testnew.nnbp$net.result)
print(testnew.nnbp)
}
