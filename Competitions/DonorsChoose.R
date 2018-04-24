## DonorsChoose.org Application Screening ##

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
      packages("xgboost")
      packages("glmnet")
      packages("stringr")
      packages("rowr")
      packages("ranger")
      packages("qdapTools")
      packages("CatEncoders")
      packages("dummies")
      packages("fastICA")
      packages("splitstackshape")
      packages("qdap")
      packages("tidytext")
      packages("tm")
      packages("text2vec")
      packages("data.table")
      packages("magrittr")
      packages("dplyr")
    })

  # '%nin%' <- Negate('%in%')
}

# Reading in the files
{
  train = read.csv('train.csv', header = T, stringsAsFactors = F)
  test = read.csv('test.csv', header = T, stringsAsFactors = F)

  resources = read.csv('resources.csv', header = T, stringsAsFactors = F)
  sample_sub = read.csv('sample_submission.csv')
}

# Processing
{
  # 1. initial processing
  {
    resources %<>%
      filter(id %in% unique(append(train$id, test$id)))

    train %<>% mutate(tt = "train")
    test %<>% mutate(tt = "test")

    train_test = bind_rows(train, test)

    cols_to_be_used_now = colnames(train_test) %>% .[c(1:8,15:17)]

    resources_aggr = resources %>%
      mutate(total_price = quantity * price) %>%
      group_by(id) %>%
      summarise(tot_quantity = sum(quantity),
                tot_price = sum(total_price),
                tot_items = n(),
                min_price = min(total_price),
                max_price = max(total_price),
                min_unit_price = min(price),
                max_unit_price = max(price),
                mean_unit_price = mean(price)) %>%
      mutate(avg_quantity = tot_quantity/tot_items,
             avg_price_item = tot_price/tot_items,
             avg_price_quantity = tot_price/tot_quantity)

    resources_aggr_2 = resources %>%
      select(id, price, quantity) %>%
      mutate(price = price * quantity,
             quantity = NULL,
             zero_price_flag = if_else(price == 0, 1, 0),
             price_1_flag = if_else(price <= 10, 1, 0),
             price_2_flag = if_else(price > 10 & price <= 50, 1, 0),
             price_3_flag = if_else(price > 50 & price <= 100, 1, 0),
             price_4_flag = if_else(price > 100 & price <= 500, 1, 0),
             price_5_flag = if_else(price > 500 & price <= 1000, 1, 0),
             price_6_flag = if_else(price > 1000 & price <= 5000, 1, 0),
             price_7_flag = if_else(price > 5000, 1, 0),
             price = NULL) %>%
      group_by(id) %>%
      summarise_all(sum)

    teacher_ids = CatEncoders::LabelEncoder.fit(train_test$teacher_id)
    train_test$teacher_id = transform(teacher_ids, train_test$teacher_id)

    train_test_main = train_test %>% select(cols_to_be_used_now)
    train_test_side = train_test %>% select(setdiff(colnames(.), cols_to_be_used_now), project_subject_categories, project_subject_subcategories, tt, id, project_is_approved)

    train_test_main %<>% left_join(., resources_aggr) %>% left_join(., resources_aggr_2)

    rm(resources, resources_aggr, resources_aggr_2, teacher_ids, train, test)
  }

  # function to paste 2 columns and make numeric equivalent
  pasteTonum = function(x, y) {
    z = as.numeric(paste0(x, y))
    return(z)
  }

  train_test_main %<>%
    splitstackshape::cSplit(., splitCols = "project_submitted_datetime", sep = " ", direction = "wide") %>%
    rename(project_submitted_date = project_submitted_datetime_1) %>%
    mutate(project_submitted_date = as.character(project_submitted_date),
           project_year = str_sub(project_submitted_date, 1, 4),
           project_month = str_sub(project_submitted_date, 6, 7),
           project_day = str_sub(project_submitted_date, 9, 10),
           project_year_month = pasteTonum(project_year, project_month),
           project_year_day = pasteTonum(project_year, project_day),
           project_month_day = pasteTonum(project_month, project_day),
           project_year_month_day = as.numeric(gsub(project_submitted_date, pattern = "-", replacement = "")),
           project_day_hour = as.numeric(str_sub(project_submitted_datetime_2, 1, 2)),
           project_daytime_flag = if_else(project_day_hour < 13, 0, 1),
           project_submitted_datetime_2 = NULL,
           project_submitted_weekday = weekdays(as.Date(project_submitted_date)))
}

#save.image('processed.RData')
#load('processed.RData')

# The NLP segment
{
  # create separate models across:
  # 1. project title
  # 2. project essay 1 + 2+3+4
  # 3. project_resource_summary
  # 4. project_categories
  ################################

  # basic features with respect to the nlp feats
  {
    train_test_side %<>%
      mutate(project_essay = paste(project_essay_1, project_essay_2, project_essay_3, project_essay_4, sep = " "),
             project_categories = paste0(project_subject_categories, " ", project_subject_subcategories),
             pt_len = nchar(project_title),
             pt_essay1_len = nchar(project_essay_1),
             pt_essay2_len = nchar(project_essay_2),
             pt_essay3_len = nchar(project_essay_3),
             pt_essay4_len = nchar(project_essay_4),
             pt_res_summary_len = nchar(project_resource_summary),
             pt_essay_len = nchar(project_essay))

    train_test_side_num = train_test_side[, c(9:11, 14:20)]
    train_test_nlp = train_test_side[, c(1, 6, 9:13)]
  }

  # define the basic functions to be used across the various NLP feats
  {
    all_stopwords = stopwords(kind = "en") %>%
      append(., stop_words$word) %>%
      append(., stopwords::data_stopwords_smart$en) %>%
      append(., stopwords::data_stopwords_stopwordsiso$en) %>%
      stemDocument(language = "english") %>%
      unique %>%
      tolower %>%
      gsub(trimws(.), pattern = "[[:punct:]]|[[:digit:]]", replacement = "") %>%
      .[. != ""] %>%
      paste0(" ", ., " ")

    all_stopwords_pat_1 = data.frame(words = all_stopwords) %>%
      .$words %>%
      .[1:500] %>%
      paste0(., collapse = "|")
    all_stopwords_pat_2 = data.frame(words = all_stopwords) %>%
      .$words %>%
      .[501:1089] %>%
      paste0(., collapse = "|")

    text_feat_treat_fn = function(x) {
      x = as.character(x) %>%
        tolower %>%
        bracketX %>%
        rm_nchar_words(., n = "1,3", trim = T, clean = T) %>%
        # removeWords(all_stopwords) %>%
        # replace_number %>%
        # replace_symbol %>%
        # replace_contraction %>%
        # replace_ordinal %>%
        # replace_abbreviation %>%
        removeNumbers %>%
        removePunctuation %>%
        stemDocument %>%
        gsub(., pattern = all_stopwords_pat_1, replacement = " ", ignore.case = T) %>%
        gsub(., pattern = all_stopwords_pat_2, replacement = "", ignore.case = T) %>%
        str_trim %>%
        str_squish

      return(x)
    }

    # define the text2vec function
    # define preprocessing function and tokenization function
    prep_fun = tolower
    tok_fun = word_tokenizer

    text2vec_fn = function(x1, x2, y, n) {
      it_train = itoken(x1[, y],
                        preprocessor = prep_fun,
                        tokenizer = tok_fun,
                        ids = x1$id,
                        progressbar = FALSE)

      vocab = create_vocabulary(it_train, stopwords = all_stopwords, ngram = c(1,3)) %>%
        prune_vocabulary(., term_count_min = 10, doc_proportion_max = 0.75, doc_proportion_min = 0.001, vocab_term_max = n)
      vectorizer = vocab_vectorizer(vocab)

      dtm_train = create_dtm(it_train, vectorizer)

      # define tfidf model
      tfidf = TfIdf$new()
      # fit model to train data and transform train data with fitted model
      dtm_train_tfidf <<- fit_transform(dtm_train, tfidf)

      # fit for test now
      it_test = itoken(x2[, y],
                       preprocessor = prep_fun,
                       tokenizer = tok_fun,
                       ids = x2$id,
                       progressbar = FALSE)

      # apply pre-trained tf-idf transformation to test data
      dtm_test_tfidf = create_dtm(it_test, vectorizer)
      dtm_test_tfidf <<- fit_transform(dtm_test_tfidf, tfidf)

      return("all ok")
    }
  }

  # train_test_nlp features - the NLP modelling module
  {
    # process the data for nlp
    t1 = Sys.time()
    train_test_nlp %<>%
      mutate(project_categories = text_feat_treat_fn(project_categories),
             project_essay = text_feat_treat_fn(project_essay),
             project_resource_summary = text_feat_treat_fn(project_resource_summary),
             project_title = text_feat_treat_fn(project_title))
    print(difftime(Sys.time(), t1, units = 'sec'))
    saveRDS(train_test_nlp, file = 'train_test_nlp.rds')

    #train_test_nlp = readRDS('train_test_nlp.rds')

    train_test_nlp %<>%
      select(-project_essay) %>%
      mutate(nlp_feat = paste(project_title, project_resource_summary, project_categories, sep = " ")) %>%
      select(nlp_feat, id, project_is_approved, tt)

    nlp_feats = c("project_title", "project_resource_summary", "project_categories")

    train_side_1 = train_test_nlp %>%
      filter(tt == "train") %>%
      select(nlp_feat, id, project_is_approved)

    # testing for the best parameters in glmnet
    train_nlp_1 = sample_frac(tbl = train_side_1, size = 0.75)
    test_nlp_1 = anti_join(train_side_1, train_nlp_1, by = "id")

    text2vec_fn(x1 = train_nlp_1, x2 = test_nlp_1, y = 1, n = 200)

    # training and validating with the 200 feats
    # glmnet
    {
      NFOLDS = 5
      t1 = Sys.time()
      glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = train_nlp_1[['project_is_approved']],
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

      preds = as.numeric(predict(glmnet_classifier, dtm_test_tfidf, type = 'response'))

      glmnet:::auc(test_nlp_1$project_is_approved, preds)
    }
    ## tried multiple parameters, ended up with 125~130 features being best
    ## validating same and getting the list from xgboost

    # xgboost
    {
      x = as.matrix(dtm_train_tfidf) %>%
        data.frame %>%
        mutate(id = row.names(.)) %>%
        left_join(., train_test_side_num %>% select(-tt)) %>%
        left_join(., train_nlp_1[, c("id", "project_is_approved"), drop = F])
      x_cols = setdiff(colnames(x), c("id", "project_is_approved"))

      y = as.matrix(dtm_test_tfidf) %>%
        data.frame %>%
        mutate(id = row.names(.)) %>%
        left_join(., train_test_side_num %>% select(-tt)) %>%
        left_join(., train_nlp_1[, c("id", "project_is_approved"), drop = F])

      train_nlp_xgb = xgb.DMatrix(data = as.matrix(x %>% select(-id, -project_is_approved)),
                                  label = as.numeric(x$project_is_approved))
      test_nlp_xgb = xgb.DMatrix(data = as.matrix(y %>% select(-id, -project_is_approved)))

      nlp_xgb = xgboost(data = train_nlp_xgb,
                        nrounds = 500,
                        early_stopping_rounds = 25,
                        print_every_n = 50,
                        verbose = 1,
                        eta = 0.005,
                        max_depth = 5,
                        objective = "binary:logistic",
                        eval_metric = "auc",
                        subsample = 0.8,
                        colsample_bytree = 0.8)

      xgb_importance = data.table(xgboost::xgb.importance(feature_names = x_cols,
                                                          model = nlp_xgb))
      Importance_table = data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
        mutate(Rank = dense_rank(desc(Importance))) %>%
        filter(Rank <= 200)
      nlp_colnames_features = as.vector(Importance_table$Feature)
    }

    # creating the final nlp train and test set
    # nlp feats to be taken stored as nlp_colnames_features (128 features alone)
    {
      train_test_nlp %<>%
        select(id, project_is_approved, nlp_feat, tt)
      train_nlp = train_test_nlp %>%
        filter(tt == "train") %>%
        mutate(tt = NULL)
      test_nlp = train_test_nlp %>%
        filter(tt == "test") %>%
        mutate(tt = NULL)

      text2vec_fn(x1 = train_nlp, x2 = test_nlp, y = 3, n = 200)

      train_nlp = as.matrix(dtm_train_tfidf) %>%
        data.frame %>%
        mutate(id = row.names(.)) %>%
        left_join(., train_test_side_num %>% select(-tt)) %>%
        select(nlp_colnames_features, id, project_is_approved)
      test_nlp = as.matrix(dtm_test_tfidf) %>%
        data.frame %>%
        mutate(id = row.names(.)) %>%
        left_join(., train_test_side_num %>% select(-tt)) %>%
        select(nlp_colnames_features, id, project_is_approved)

      save(list = c("train_nlp", "test_nlp"), file = 'nlp_backup.rda')
      load('nlp_backup.rda')
    }
  }

  rm(dtm_test_tfidf, dtm_train_tfidf, glmnet_classifier, Importance_table, nlp_xgb,
     test_nlp_1, test_nlp_xgb, train_nlp_1, train_nlp_xgb, train_side_1, train_test_nlp, train_test_side_num, train_test_side,
     xgb_importance, x, y, preds, all_stopwords, all_stopwords_pat_1, all_stopwords_pat_2)
  gc()
}

#save.image('after_nlp.RData')
#load('after_nlp.RData')

# Feature Engineering
{
  # existing feature transformations/engineering
  {
    # PCA
    {
      train_test_pca = train_test_main %>%
        select_if(is.numeric) %>%
        cbind(., train_test_main %>% select(tt))

      train_num_f = train_test_pca %>% filter(tt == "train") %>% select(-tt, -project_is_approved) %>% mutate_all(.funs = as.numeric)
      test_num_f = train_test_pca %>% filter(tt == "test") %>% select(-tt, -project_is_approved) %>% mutate_all(.funs = as.numeric)

      pca_feats = prcomp(x = train_num_f, retx = T, center = T, tol = 0, scale. = T)
      expl.var = round(pca_feats$sdev^2/sum(pca_feats$sdev^2)*100)
      # top 2 components itself explains the whole variance

      # scree plot
      {
        std_dev = pca_feats$sdev^2
        pca_var = std_dev^2
        prop_var = pca_var/sum(pca_var)
        plot(cumsum(prop_var), xlab = "PC", ylab = "Prop Var Exp", type = "b")
      }

      pca_feats_to_be_added = data.frame(pca_feats$x[, 1:10])
      test_pca_pred = data.frame(predict(pca_feats, newdata = test_num_f) %>% .[, 1:10])
      pca_preds = bind_rows(pca_feats_to_be_added, test_pca_pred)
      train_test_main %<>% cbind(., pca_preds)

      train_test_ica = train_test_pca
      rm(pca_feats, pca_feats_to_be_added, pca_var, expl.var, prop_var, std_dev, train_num_f, test_num_f, test_pca_pred, pca_preds)
    }

    # ICA
    {
      train_num_f = train_test_ica %>% filter(tt == "train") %>% select(-tt, -project_is_approved) %>% mutate_all(.funs = as.numeric)
      test_num_f = train_test_ica %>% filter(tt == "test") %>% select(-tt, -project_is_approved) %>% mutate_all(.funs = as.numeric)

      train_ica = fastICA(train_num_f, n.comp = 8, maxit = 20, verbose = T, tol = 1e-04)

      train_ica_cols = train_ica$S %>% data.frame %>% set_colnames(paste0("ica_", 1:8))

      test_ica_cols = as.matrix(test_num_f) %*% train_ica$K %*% train_ica$W %>%
        data.frame %>%
        set_colnames(paste0("ica_", 1:8))
      train_test_ica_cols =  bind_rows(train_ica_cols, test_ica_cols)

      train_test_main %<>% cbind(., train_test_ica_cols)

      rm(train_num_f, test_num_f, train_ica, train_ica_cols, test_ica_cols, train_test_ica_cols)
    }

    rm(train_test_ica, train_test_pca)

    # Deviation encoding
    {
      # split train and test to ensure no leakage
      train = train_test_main %>%
        filter(tt == "train") %>%
        mutate(tt = NULL)
      test = train_test_main %>%
        filter(tt == "test") %>%
        mutate(tt = NULL)

      train_summary = summarise_all(train, n_distinct) %>%
        melt %>%
        set_colnames(c("COLUMN", "COUNT_DISTINCT")) %>%
        mutate(char_flag = if_else(COUNT_DISTINCT < 500, 1, 0))
      categ_variables = train_summary %>%
        filter(char_flag == 1) %>%
        .$COLUMN %>%
        as.character %>%
        setdiff(., c("project_is_approved"))

      rm(train_summary)

      train %<>% mutate_if(colnames(.) %in% categ_variables,
                           as.character)
      test %<>% mutate_if(colnames(.) %in% categ_variables,
                          as.character)

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
                        paste0(temp_col, "_count"))

          temp = train_char_data[, i, drop = F] %>%
            cbind(., train_num_data) %>%
            group_by_at(vars(-matches("project_is_approved"))) %>%
            mutate(mean = mean(project_is_approved),
                   sd = sd(project_is_approved),
                   count = n()) %>%
            ungroup %>%
            select(temp_col, mean, sd, count) %>%
            set_colnames(temp_cols) %>%
            distinct

          train <<- left_join(train, temp)
          test <<- left_join(test, temp)
        }

        return(print("train and test have been generated"))
      }

      categtoDeviationenc(char_data = train %>% select(categ_variables), num_data = train %>% select(project_is_approved))
    }

    ## find out which columns are truly numeric, and typecast them
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
      train_class = train %>%
        summarise_all(cols_class_fn)
      train_class_df = data.frame(col = colnames(train), flag = t(train_class), stringsAsFactors = F)
      train_num_cols = train_class_df %>% filter(flag == 1) %>% .$col
      train_char_cols = train_class_df %>% filter(flag == 0) %>% .$col

      train_test_main = bind_rows(train %>% mutate(tt = "train"),
                                  test %>% mutate(tt = "test")) %>%
        mutate_if(colnames(.) %in% train_char_cols, as.character) %>%
        mutate_if(colnames(.) %in% train_num_cols, as.numeric)
    }

    # Random
    {
      train_test_main %<>%
        cSplit(., splitCols = "project_subject_categories", sep = ",", direction = "wide", drop = F, stripWhite = T) %>%
        cSplit(., splitCols = "project_subject_subcategories", sep = ",", direction = "wide", drop = F, stripWhite = T) %>%
        mutate(count_project_subject_categories = str_count(project_subject_categories, pattern = ",") + 1,
               count_project_subject_subcategories = str_count(project_subject_subcategories, pattern = ",") + 1)

            # teacher_stats = train_test_main %>%
            #   filter(tt == "train") %>%
            #   group_by(teacher_id) %>%
            #   summarise(teacher_count_of_all_projects = n(),
            #             teacher_count_of_approved_projects = sum(project_is_approved),
            #             teacher_prob_of_approval = n()/sum(project_is_approved),
            #             teacher_normalized_score_nologic = teacher_prob_of_approval * teacher_count_of_approved_projects,
            #             cum_tot_quantity = sum(tot_quantity),
            #             cum_tot_price = sum(tot_price),
            #             cum_tot_items = sum(tot_items))
            #
            # train_test_main %<>%
            #   left_join(., teacher_stats) %>%
            #   mutate(teacher_prob_of_approval = if_else(is.na(teacher_prob_of_approval), 0.5, teacher_prob_of_approval))
    }

    # treating NULLS
    {
      View(colSums(is.na(train_test_main)))

      remove_largely_null_cols_fn = function(x) {
        x = as.vector(x)
        if (sum(is.na(x)) > (0.35 * length(x))) {
          x = 1
        } else {x = 0}
        return(x)
      }

      train_test_main_null_cols_not = train_test_main %>%
        summarise_all(remove_largely_null_cols_fn) %>%
        melt %>%
        filter(value == 0) %>%
        .$variable %>% as.character

      train_test_main %<>%
        select(train_test_main_null_cols_not)
    }

    rm(train_class, train_class_df, train, test, train_test)
  }
}

# some stuff
{
  rm(train_char_cols, train_num_cols)

  # repeat the class test for dummification for FS
  # flag every column as 1 (numeric) and 0 (character)

  train_class = train_test_main %>%
    summarise_all(cols_class_fn)
  train_class_df = data.frame(col = colnames(train_test_main), flag = t(train_class), stringsAsFactors = F)
  train_num_cols = train_class_df %>% filter(flag == 1) %>% .$col
  train_char_cols = train_class_df %>% filter(flag == 0) %>% .$col

  train_test_main %<>% mutate_if(colnames(.) %in% train_char_cols, as.character) %>%
    mutate_if(colnames(.) %in% train_num_cols, as.numeric)
}

# Feature selection
{
  train = train_test_main %>%
    filter(tt == "train") %>%
    mutate(tt = NULL)

  train_dummy_cols = setdiff(train_char_cols, c("tt", "id", "project_submitted_date"))
  train_dummy = dummy.data.frame(data = train, names = train_dummy_cols, sep = "_")
  train_xgb = xgb.DMatrix(data = data.matrix(train_dummy %>% select(-project_is_approved, -id, -project_submitted_date)),
                          label = data.matrix(as.numeric(train_dummy[,'project_is_approved'])))

  xgb_feat_selection_model = xgboost(data = train_xgb,
                                     nrounds = 400,
                                     early_stopping_rounds = 25,
                                     print_every_n = 50,
                                     verbose = 1,
                                     eta = 0.01,
                                     max_depth = 5,
                                     objective = "binary:logistic",
                                     eval_metric = "auc",
                                     subsample = 0.3,
                                     colsample_bytree = 0.3)

  xgb_importance = data.table(xgboost::xgb.importance(feature_names = setdiff(colnames(train_dummy), c("project_is_approved", "id", "project_submitted_date")),
                                                      model = xgb_feat_selection_model))

  Importance_table = data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
    mutate(Rank = dense_rank(desc(Importance))) %>%
    filter(Rank <= 500)
  colnames_features_main = as.vector(Importance_table$Feature)


  ## subset for the required columns alone
  train_test_main_dummy = dummy.data.frame(data = train_test_main, names = train_dummy_cols, sep = "_")

  train_x = train_test_main_dummy %>% filter(tt == "train") %>%
    mutate(tt = NULL) %>%
    select(append(colnames_features_main, c("project_is_approved", "id")))
  test_x = train_test_main_dummy %>% filter(tt == "test") %>%
    mutate(tt = NULL) %>%
    select(append(colnames_features_main, c("id")))

  rm(xgb_feat_selection_model, xgb_importance, train_dummy, train_test_main_dummy, train_class, train_class_df, train, train_test_main)
}

# combine the main and nlp features for a single dataset as well. 3 models to be created for ensemble
{
  train_all_feats = inner_join(train_x, train_nlp)
  test_all_feats = inner_join(test_x, test_nlp)

  train_nlp = train_nlp
  test_nlp = test_nlp

  train_non_nlp = train_x
  test_non_nlp = test_x
  rm(train_x, test_x)

  View(data.frame(colnames(train_all_feats)))

  train_test_all_feats = bind_rows(train_all_feats %>%
                                     mutate(tt = "train"),
                                   test_all_feats %>%
                                     mutate(tt = "test"))
  train_test_nlp = bind_rows(train_nlp %>%
                               mutate(tt = "train"),
                             test_nlp %>%
                               mutate(tt = "test"))
  train_test_non_nlp = bind_rows(train_non_nlp %>%
                                   mutate(tt = "train"),
                                 test_non_nlp %>%
                                   mutate(tt = "test"))

  # treat NAs in the datasets
  num_cols_nas_fix_fn = function(x) {
    x = as.numeric(x)
    x[which(is.na(x))] = mean(x, na.rm = T)
    return(x)
  }

  train_test_all_feats %<>%
    mutate_if(is.numeric, num_cols_nas_fix_fn)
  train_test_nlp %<>%
    mutate_if(is.numeric, num_cols_nas_fix_fn)
  train_test_non_nlp %<>%
    mutate_if(is.numeric, num_cols_nas_fix_fn)
}

#save.image('backup.RData')
#load('backup.RData')
rm(train_all_feats, test_all_feats, train_nlp, test_nlp, train_non_nlp, test_non_nlp)

# Model - xgboost (3 models)
{
  gc()

  ## the 3 datasets ##
  # train_test_all_feats
  # train_test_nlp
  # train_test_non_nlp

  # applied the best params from grid search
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

  xgb_model_fn(train_test_all_feats)

  # final only single model was made since the others had no value
}

# Finals
{
  gc()
fwrite(train_test_all_feats, 'final_ads.csv')
fwrite(train_test_all_feats %>%
         filter(tt == "train") %>%
         select(-id, -tt) %>%
         mutate(project_is_approved = as.character(project_is_approved)) %>%
         select(project_is_approved, everything()),
        'final_train.csv', showProgress = T)
fwrite(train_test_all_feats %>%
         filter(tt == "test") %>%
         select(-id, -tt, -project_is_approved) %>%
         select(project_is_approved, everything()),
       'final_test.csv', showProgress = T)
}

# h2o deep learning
{
  # Initialization
  {
    gc()
    cat("\014")
    rm(list = setdiff(ls(), c("output", "train_test_all_feats")))

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
        packages("magrittr")
        packages("dplyr")
      })

    # '%nin%' <- Negate('%in%')
  }

  # Load the H2O R package:
  library(h2o)

  #### Start H2O
  #Start up a 1-node H2O server on your local machine, and allow it to use all CPU cores and up to 4GB of memory:

  h2o.init(nthreads = 3, max_mem_size = "6G")
  h2o.removeAll() ## clean slate - just in case the cluster was already running

  output = fread('output.csv')

  train_full.hex = h2o.importFile(path = 'final_train.csv', destination_frame = 'final_train')
  test_full.hex = h2o.importFile(path = 'final_test.csv', destination_frame = 'final_test')
  names(train_full.hex)

  train_full.hex[, 1] = as.factor(train_full.hex[, 1])

  # splits = h2o.splitFrame(train_full.hex, c(0.6,0.2), seed = 1234)
  # train  = h2o.assign(splits[[1]], "train.hex") # 60%
  # valid  = h2o.assign(splits[[2]], "valid.hex") # 20%
  # test   = h2o.assign(splits[[3]], "test.hex")  # 20%

  response = "project_is_approved"
  predictors = setdiff(names(train_full.hex), response)
  weight = paste0("C", as.character(length(names(train_full.hex)) + 1))

  # define the splits
  h2o_splits = h2o.splitFrame(train_full.hex, c(0.6, 0.2), seed = 1234)
  h2o_DstTrain  = h2o.assign(h2o_splits[[1]], "train.hex") # 60%
  h2o_DstTest  = h2o.assign(h2o_splits[[2]], "test.hex") # 20%
  h2o_DstValid = h2o.assign(h2o_splits[[3]], "valid.hex") # 20%

  # Number of CV folds (to generate level-one data for stacking)
  cvfolds <- 5

  get_auc <- function(x) h2o.auc(h2o.performance(h2o.getModel(x), newdata = h2o_DstValid))

  # Train & Cross-validate a GBM
  {
    my_gbm <- h2o.gbm(x = predictors,
                      y = response,
                      training_frame = h2o_DstTrain,
                      distribution = "bernoulli",
                      max_depth = 7,
                      min_rows = 3,
                      learn_rate = 0.1,
                      ntrees = 200,
                      nfolds = cvfolds,
                      fold_assignment = "Stratified",
                      keep_cross_validation_predictions = TRUE,
                      seed = 1,
                      stopping_metric = "AUC",
                      balance_classes = T,
                      max_after_balance_size = 3,
                      sample_rate = 0.7,
                      col_sample_rate = 0.6,
                      calibrate_model = T,
                      verbose = F,
                      calibration_frame = h2o_DstTest,
                      stopping_rounds = 10)

    h2o.saveModel(my_gbm, path = "./", force = TRUE)
    #my_gbm <- h2o.loadModel('GBM_model_R_1524547384365_773')

    # Measure auc
    get_auc(my_gbm@model_id)
  }

  # Train & Cross-validate a RF
  {
    my_rf <- h2o.randomForest(x = predictors,
                              y = response,
                              training_frame = h2o_DstTrain,
                              nfolds = cvfolds,
                              fold_assignment = "Stratified",
                              keep_cross_validation_predictions = TRUE,
                              seed = 1,
                              #balance_classes = T,
                              #max_after_balance_size = 2,
                              ntrees = 300,
                              max_depth = 7,
                              stopping_rounds = 10,
                              stopping_metric = "AUC",
                              verbose = F,
                              calibrate_model = T,
                              calibration_frame = h2o_DstTest,
                              mtries = 10,
                              binomial_double_trees = T)

    h2o.saveModel(my_rf, path = "./", force = TRUE)
    #my_rf <- h2o.loadModel('DRF_model_R_1524547384365_991')

    # Measure auc
    get_auc(my_rf@model_id)
  }

  # Train & Cross-validate a DNN
  {
    h2o.colnames(h2o_DstTrain)
    h2o_DstTrain[, length(names(h2o_DstTrain)) + 1] = ifelse(h2o_DstTrain[, 1] == 0, 3, 1)

    my_dl <- h2o.deeplearning(x = predictors,
                              y = response,
                              training_frame = h2o_DstTrain,
                              l1 = 0.001,
                              l2 = 0.001,
                              hidden = c(50, 75, 75, 50),
                              nfolds = cvfolds,
                              fold_assignment = "Stratified",
                              keep_cross_validation_predictions = TRUE,
                              seed = 1,
                              activation = "RectifierWithDropout",
                              stopping_rounds = 10,
                              stopping_metric = "AUC",
                              variable_importances = F,
                              balance_classes = T,
                              max_after_balance_size = 2,
                              input_dropout_ratio = 0.5,
                              epochs = 1,
                              weights_column = weight)

    h2o.saveModel(my_dl, path = "./", force = TRUE)
    #my_dl <- h2o.loadModel('DeepLearning_model_R_1524547384365_1243')

    # Measure auc
    get_auc(my_dl@model_id)
  }

  # Create a Stacked Ensemble
  # To maximize predictive power, will create an H2O Stacked Ensemble from the models we created above and print the performance gain the ensemble has over the best base model.
  # Train a stacked ensemble using the H2O and XGBoost models from above
  base_models <- list(my_gbm@model_id, my_rf@model_id, my_dl@model_id)

  h2o_DstTrain[, length(names(h2o_DstTrain)) + 1] = NULL

  ensemble <- h2o.stackedEnsemble(x = predictors,
                                  y = response,
                                  training_frame = h2o_DstTrain,
                                  base_models = base_models)
  # Eval ensemble performance on a test set
  perf <- h2o.performance(ensemble, newdata = h2o_DstTest)

  # Compare to base learner performance on the test set
  baselearner_aucs <- sapply(base_models, get_auc)
  baselearner_best_auc_test <- max(baselearner_aucs)
  ensemble_auc_test <- h2o.auc(perf)

  # Compare the test set performance of the best base model to the ensemble.
  print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
  print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))

  # predict with the model
  predictFinal <- h2o.predict(ensemble, test_full.hex)

  # convert H2O format into data frame and save as csv
  predictFinal.df <- as.data.frame(predictFinal)

  output_final = data.frame(id = train_test_all_feats %>%
                              filter(tt == "test") %>%
                              select(id),
                            project_is_approved = predictFinal.df$p1)
  # write the submition file
  fwrite(output_final, "Result.csv")

  output_ensemble = left_join(output, output_final, by = "id") %>%
    mutate(project_is_approved = (project_is_approved.x + project_is_approved.y)/2) %>%
    select(id, project_is_approved)
  fwrite(output_ensemble, 'output_ens.csv')

  # shut down virtual H2O cluster
  h2o.shutdown(prompt = FALSE)
}