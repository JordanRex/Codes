###############
## Functions ##
###############

############
# Contents #
############

# minor functions
# 1. initialization block
# 2.

# major functions
# 1.


# MINOR
{
# 1. initialization block
# gc, rm envir, load most common packages (install if not present)
{
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
        # packages("xgboost")
        # packages("stringr")
        # packages("qdapTools")
        # packages("CatEncoders")
        # packages("dummies")
        # packages("fastICA")
        # packages("splitstackshape")
        # packages("qdap")
        packages("data.table")
        packages("magrittr")
        packages("dplyr")
      })

  }
}

# 2. negate operator and the equivalent function
'%nin%' = Negate('%in%')
'%!in%' = function(x,y)!('%in%'(x,y))

# 3. function to paste 2 columns and make numeric equivalent
pasteTonum = function(x, y) {
  z = as.numeric(paste0(x, y))
  return(z)
}

# 4. stopwords for NLP
{
  all_stopwords = function(x = NULL) {
    x = stopwords(kind = "en") %>%
      append(., stop_words$word) %>%
      append(., stopwords::data_stopwords_smart$en) %>%
      append(., stopwords::data_stopwords_stopwordsiso$en) %>%
      stemDocument(language = "english") %>%
      unique %>%
      tolower %>%
      gsub(trimws(.), pattern = "[[:punct:]]|[[:digit:]]", replacement = "") %>%
      .[. != ""] %>%
      paste0(" ", ., " ")

    return(x)
  }
}

# 5. function to flag which columns are pointless
useless_cols_fn = function(x) {
  y = sum(is.na(x))

  if (y != length(x)) {
    y = if_else(y < ceiling(0.3 * length(x)), 1, 0)
    if ((y == 1) & (length(unique(x)) == 1)) {
      y = 0
    }
    if (sort(table(x), decreasing = T)[1] >= (0.98 * length(x))) {
      y = 0
    }
  }

  return(as.character(y))
}

# 6. function to find columns that are actually numeric, not stored as character
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

# 7. mode function to fill NAs with mode of column
mode_fn = function(x) {
  y = attr(sort(table(x), decreasing = T)[1], which = "name")
  x[is.na(x)] = y
  return(x)
}

# 8. function to make numeric columns factors if unique values are less than n
factor_fn = function(x, n = 53) {
  unique_temp = length(unique(x))

  if (unique_temp <= n) {
    x = as.factor(x)
  } else {
    x = x
  }

  return(x)
}

# 9.
}


# MAJOR
{
  # 1. text cleaning/treatment function
  {
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
  }

  # 2. text2vec function
  {
    text2vec_fn = function(x1, x2, y, vocab_term_max_n = 300, ngram_max = 3) {
      # define preprocessing function and tokenization function
      prep_fun = tolower
      tok_fun = word_tokenizer

      it_train = itoken(x1[, y],
                        preprocessor = prep_fun,
                        tokenizer = tok_fun,
                        ids = x1$id,
                        progressbar = FALSE)

      vocab = create_vocabulary(it_train, stopwords = all_stopwords, ngram = c(1, ngram_max)) %>%
        prune_vocabulary(., term_count_min = 10, doc_proportion_max = 0.75, doc_proportion_min = 0.001, vocab_term_max = vocab_term_max_n)
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

  # 4. PCA function
  {
    pca_fn = function(train, test, response, id, no_of_pca_feats = 10) {
        temp_train = train %>% mutate_all(.funs = as.numeric) %>% select_(.dots = paste0("-", list(id, response)))
        temp_test = test %>% mutate_all(.funs = as.numeric) %>% select_(.dots = paste0("-", list(id)))

        pca_feats = prcomp(x = temp_train, retx = T, center = T, tol = 0, scale. = T)
        expl.var = round(pca_feats$sdev^2/sum(pca_feats$sdev^2)*100)

        # scree plot
        {
          std_dev = pca_feats$sdev
          pca_var = std_dev^2
          prop_var = pca_var/sum(pca_var)
          plot(cumsum(prop_var), xlab = "PC", ylab = "Prop Var Exp", type = "b")
        }

        pca_feats_to_be_added = data.frame(pca_feats$x[, 1:no_of_pca_feats])
        train %<>% cbind(., pca_feats_to_be_added)

        test_pca_pred = data.frame(predict(pca_feats, newdata = test_num_f) %>% .[, 1:no_of_pca_feats])
        test %<>% cbind(., test_pca_pred)

        assign(x = "pca_feats_added_train", value = train, envir = .GlobalEnv)
        assign(x = "pca_feats_added_test", value = test, envir = .GlobalEnv)

        return("two dataframes: pca_feats_added_train and pca_feats_added_test have been created in environment")
      }
  }

  # 5. ICA function
  {
    ica_gn = function(train, test, response, id, no_of_ica_feats = 10) {
      train_num_f = train %>% mutate_all(.funs = as.numeric) %>% select_(.dots = paste0("-", list(id, response)))
      test_num_f = test %>% mutate_all(.funs = as.numeric) %>%  select_(.dots = paste0("-", list(id)))

      train_ica = fastICA(train_num_f, n.comp = no_of_ica_feats, maxit = 50, verbose = T, tol = 1e-04)

      train %<>% cbind(., train_ica$S %>% data.frame %>% set_colnames(paste0("ica_", 1:no_of_ica_feats)))

      train_ica_df = as.matrix(test_num_f) %*% train_ica$K %*% train_ica$W %>%
        data.frame %>%
        set_colnames(paste0("ica_", 1:no_of_ica_feats))
      test %<>% cbind(., train_ica_df)

      assign(x = "ica_feats_added_train", value = train, envir = .GlobalEnv)
      assign(x = "ica_feats_added_test", value = test, envir = .GlobalEnv)

      return("two dataframes: ica_feats_added_train and ica_feats_added_test have been created in environment")
    }
  }

  # 6. cor function
  # returns a vector with column names of the highly correlated and removable columns
  {
    cor_fn = function(x, cor_threshold = 0.8) {
      cor_matrix = cor(x)
      cor_matrix[is.na(cor_matrix)] <- 0

      # find attributes that are highly corrected (ideally > 0.8, change if necessary)
      highlyCorrelated = findCorrelation(correlationMatrix, cutoff = cor_threshold)

      # print indexes of highly correlated attributes
      print("Correlated features")
      print(colnames(ADS_cor[ , highlyCorrelated, drop = F]))

      highlyCorrelated = colnames(ADS_cor[ , highlyCorrelated, drop = F])

      return(highlyCorrelated)
  }
  }

  # 7.
}
