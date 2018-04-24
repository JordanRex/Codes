## Functions ##

############
# Contents #
############

# 1. initialization block
# 2.




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
        packages("xgboost")
        packages("stringr")
        packages("qdapTools")
        packages("CatEncoders")
        packages("dummies")
        packages("fastICA")
        packages("splitstackshape")
        packages("qdap")
        packages("data.table")
        packages("magrittr")
        packages("dplyr")
      })

  }
}

# 2. negate operator
'%nin%' = Negate('%in%')

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

# 5. text cleaning/treatment function
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

# 6. text2vec function
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