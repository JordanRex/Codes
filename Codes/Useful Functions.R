###############
## Functions ##
## AUTHOR -> VARUN V
## Thanks to https://github.com/raredd for some functions used here
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
pasteTonum_fn = function(x, y) {
  z = as.numeric(paste0(x, y))
  return(z)
}

# 4. stopwords for NLP
  all_stopwords_fn = function(x = NULL) {
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

# 7. NA treatment functions
{
# mode function to fill NAs with mode of column
mode_fn = function(x) {
  y = attr(sort(table(x), decreasing = T)[1], which = "name")
  x[is.na(x)] = y
  return(x)
}

miss_treatment_categ_fn = function(x, y = "unknown") {
  if (is.numeric(x)) x[which(is.na(x))] = mean(x, na.rm = T)
  if (!is.numeric(x)) x[which(is.na(x))] = y
  return(x)
}
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

# 9. Get RAM size in MB according to OS
ram_memory_fn = function() {
  OSType = Sys.info()['sysname']
  if(OSType == "Windows")
    RAMSize = memory.limit() else if (OSType == "Linux")
    RAMSize = floor(as.numeric(system("awk '/MemTotal/ {print $2}' /proc/meminfo",intern=TRUE))/1024) else if (OSType == "Darwin")
      cmdOutput = system("sysctl hw.memsize",intern = TRUE)
    RAMSize = substr(cmdOutput,13,nchar(cmdOutput))
    RAMSize = as.numeric(RAMSize)/(1024*1024) else
    RAMSize = NULL

  return(RAMSize)
}

# 10. create year, month, day and year-month columns from a Date column
date_multi_feat_fn = function(dataset, date_column = "Date") {
  temp = dataset

  ## you can use different methods to escape special characters
  # either '[/]' or '\\/' or '\\Q/\\E'
  # the first one is better because you wont have to deal with those extra escape characters being a problem when you have to save the regex pattern inside a string
  date_fn_year = paste0('gsub(', date_column, ', pattern = "(.*)[/]", replacement = "")')
  date_fn_month = paste0('gsub(', date_column, ', pattern = "[/](.*)", replacement = "") %>% str_pad(., width = 2, side = "left", pad = "0")')
  date_fn_day = paste0('str_extract(', date_column, ', pattern = regex("(?<=[/]).*(?=[/](.*?))", perl = T)) %>% str_pad(., width = 2, side = "left", pad = "0")')
  date_fn_year_month = paste0('paste0(', paste0(date_column, c("_year", "_month"), collapse = ","), ')')
  date_cols = paste0(date_column, c("_year", "_month", "_day", "_year_month"))

  temp %<>%
    mutate_(.dots = setNames(list(date_fn_year, date_fn_month, date_fn_day, date_fn_year_month),
                             date_cols))

  return(temp)

  ## example --->> dataset %<>% date_multi_feat_fn(., date_column = "Date")
}

# 11. function to get the range of values inside a range by giving an input of the range index to be returned
'%:%' <- function(object, range) {
  FUN <- if (!is.null(dim(object))) {
    if (is.matrix(object)) colnames else names
  } else identity
  wh <- if (is.numeric(range))
    range else which(FUN(object) %in% range)
  FUN(object)[seq(wh[1L], wh[2L])]
}
# example -> 4:9 %:% c(3,5) = (6, 7, 8)
# example -> letters %:% c('e', 'n') = ("e", "f", "g", "h", "i", "j", "k", "l", "m", "n")

# 12. pairwise sum of 2 vectors
psum_fn = function(..., na.rm = FALSE) {
  dat = do.call('cbind', list(...))
  res = rowSums(dat, na.rm = na.rm)

  idx = !rowSums(!is.na(dat))
  res[idx] = NA

  return(res)
}
# example -> psum(1:10, 10:19, 20:29) = (31, 34, 37, 40, 43, 46, 49, 52, 55, 58)

# 13. rescaling function adopted from scales package
rescaler_fn = function (x, to = c(0, 1), from = range(x, na.rm = TRUE)) {
  zero_range = function(x, tol = .Machine$double.eps * 100) {
    if (length(x) == 1L)  return(TRUE)
    if (length(x) != 2L)  stop('\'x\' must be length one or two')
    if (any(is.na(x)))    return(NA)
    if (x[1L] == x[2L])   return(TRUE)
    if (all(is.infinite(x))) return(FALSE)
    m = min(abs(x))
    if (m == 0) return(FALSE)
    abs((x[1L] - x[2L]) / m) < tol
  }

  if (zero_range(from) || zero_range(to))
    return(rep(mean(to), length(x)))

  (x - from[1L]) / diff(from) * diff(to) + to[1L]
}

# 14. multi-recursive join
multi_join_fn = function(list_dfs, ...) {
  Reduce(function(x, y) left_join(x, y, ...), list_dfs)
}

# 15. utils::View shortcut function
view = function(x, title) {
  utils::View(x, title)
}

# 16. convenience functions for col_to_rownames and the reverse
{
  rownames_to_column <- function(data, column = 'rownames', where = 1L) {
    column <- make.unique(c(colnames(data), column))[ncol(data) + 1L]
    data   <- insert(data, col = where, repl = rownames(data))

    colnames(data)[where] <- column
    rownames(data) <- NULL

    data
  }

  column_to_rownames <- function(data, column = 'rownames', where = 1L) {
    where <- if (missing(where))
      which(colnames(data) %in% column) else where

    stopifnot(
      length(where) == 1L,
      where %in% seq.int(ncol(data))
    )

    rownames(data) <- make.unique(as.character(data[, where]))

    data[, -where, drop = FALSE]
  }
}

# 17. EDA function
get_basic_eda_fn = function(df) {
  DataExplorer::create_report(df)
}

# 18. remove columns with nearZero variance (existing function from caret)
caret::nearZeroVar(x = train, freqCut = 10, uniqueCut = 20, names = T)

# 19.
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

  # 3.

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

  # 7. ram size function
  # Get RAM size in MB according to OS
  ram_memory = function() {
    OSType = Sys.info()['sysname']
    RAMSize = NULL

    if(OSType == "Windows")
      RAMSize = memory.limit()
    if (OSType == "Linux")
      RAMSize = floor(as.numeric(system("awk '/MemTotal/ {print $2}' /proc/meminfo",intern=TRUE))/1024)
    if (OSType == "Darwin") {
      cmdOutput = system("sysctl hw.memsize",intern = TRUE)
      RAMSize = substr(cmdOutput,13,nchar(cmdOutput))
      RAMSize = as.numeric(RAMSize)/(1024*1024)
    }

    return(RAMSize)
  }

  # 8.
}
