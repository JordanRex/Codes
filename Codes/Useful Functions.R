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
  
  # 19. vlookup function = run the below code snippet
  {
    vlookup = function(ref, #the value or values that you want to look for
                       table, #the table where you want to look for it; will look in first column
                       column, #the column that you want the return data to come from,
                       range=FALSE, #if there is not an exact match, return the closest?
                       larger=FALSE) #if doing a range lookup, should the smaller or larger key be used?)
    {
      if(!is.numeric(column) & !column %in% colnames(table)) {
        stop(paste("can't find column",column,"in table"))
      }
      if(range) {
        if(!is.numeric(table[,1])) {
          stop(paste("The first column of table must be numeric when using range lookup"))
        }
        table <- table[order(table[,1]),] 
        index <- findInterval(ref,table[,1])
        if(larger) {
          index <- ifelse(ref %in% table[,1],index,index+1)
        }
        output <- table[index,column]
        output[!index <= dim(table)[1]] <- NA
        
      } else {
        output <- table[match(ref,table[,1]),column]
        output[!ref %in% table[,1]] <- NA #not needed?
      }
      dim(output) <- dim(ref)
      output
    }
  }
  
  # 20. normalized gini metric
  {
    normalizedGini <- function(aa, pp) {
      Gini <- function(a, p) {
        if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
        temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
        temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
        population.delta <- 1 / length(a)
        total.losses <- sum(a)
        null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
        accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
        gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
        sum(gini.sum) / length(a)
      }
      Gini(aa,pp) / Gini(aa,aa)
    }
  }
  
  # 21. function to get period for timeseries
  find_freq_fn = function(x)
  {
    n <- length(x)
    spec <- spec.ar(c(x),plot = FALSE)
    if (max(spec$spec) > 10) # Arbitrary threshold chosen by trial and error.
    {
      period <- round(1/spec$freq[which.max(spec$spec)])
      if (period == Inf) # Find next local maximum
      {
        j <- which(diff(spec$spec) > 0)
        if (length(j) > 0)
        {
          nextmax <- j[1] + which.max(spec$spec[j[1]:500])
          period <- round(1/spec$freq[nextmax])
        }
        else
          period <- 1
      }
    }
    else
      period <- 1
    return(period)
  }
  
  # 22. Hyndman function for identifying outliers in time series
  tsoutliers_fn = function(x,plot=FALSE)
  {
    x <- as.ts(x)
    if (frequency(x) > 1)
      resid <- stl(x, s.window = "periodic", robust = TRUE)$time.series[,3]
    else
    {
      tt <- 1:length(x)
      resid <- residuals(loess(x ~ tt))
    }
    resid.q <- quantile(resid,prob = c(0.25,0.75))
    iqr <- diff(resid.q)
    limits <- resid.q + 1.5*iqr*c(-1,1)
    score <- abs(pmin((resid - limits[1])/iqr,0) + pmax((resid - limits[2])/iqr,0))
    if (plot)
    {
      plot(x)
      x2 <- ts(rep(NA,length(x)))
      x2[score > 0] <- x[score > 0]
      tsp(x2) <- tsp(x)
      points(x2,pch = 19,col = "red")
      return(invisible(score))
    }
    else
      return(score)
  }
  
  # 23. To use addins in RStudio
  {
    devtools::install_github("rstudio/addinexamples", type = "source")
  }
  
  # 24. statistical method of evaluating if log transform should be used for arima
  {
    if (gqtest(ads_ts ~ 1)$p.value < 0.1) {...}
    # can be used with an OR (||) operator with below expression as well
    if (abs(BoxCox.lambda(ads_ts) < 0.1)) {...}
  }
  
  # 25. To remove files from a directory with a pattern
  {
    do.call(file.remove, list(list.files("C:/Temp", full.names = TRUE)))
    # use grep to get the pattern out
  }
  
  # 24. setting wd to current file location (two methods given)
  {
    # A: works only when the r script is called through the terminal
    this.dir <- dirname(parent.frame(2)$ofile)
    setwd(this.dir)
    
    # B: works in RStudio
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }
  
  # 25. user-defined functions to get the rolling mean and standard deviation, and the rolling percentile values
  {
    #
    my.rollapply <- function(vec, width, FUN)
      sapply(seq_along(vec),
             function(i) if (i < width) NA else FUN(vec[i:(i - width + 1)]))
    
    my.rollapply.quantile <- function(vec, width,x)
      sapply(seq_along(vec),
             function(i) if (i < width) NA else quantile(vec[i:(i - width + 1)],probs = x))
  }
  
  # 26: for updating R within RStudio
  {
    if (!require(installr)) {
      install.packages("installr"); require(installr)} #load / install+load installr
    
    # using the package:
    updateR()
  }
  
  # 27. Various ways of printing content
  {
    my_string = "hehe"
    
    # print()
    print(my_string, quote = F)
    
    # noquote()
    noquote(my_string)
    
    # cat()
    # used for concatenation
    cat(my_string,"hehe", sep = ".", fill = 5)
    
    # format()
    # justify = c("c", "l", "r", "n")
    format(my_string, width = 10, justify = "r")
    
    # sprintf()
    sprintf("%05.1f", pi)
    
    # toString()
    # can be used a helper for the above commands to pass a vector of multiple strings, separated by commas
    toString(c("Bonjour", 123, TRUE, NA, log(exp(1))))
  }
  
  # 28. Random regex codes
  # Need to make them functions later
  {
    # using a sample df to display usage of some regex functions
    df <- USArrests
    df.var = rownames(df)
    
    df.name = deparse(substitute(df))
    df.var.name = deparse(substitute(df.var))
    
    # Abbreviating Strings
    df.var.abbr = abbreviate(df.var, minlength = 6, use.classes = F)
    
    # Getting the longest string in a column
    char.df.var = nchar(df.var)
    assign(paste0(df.name,".longest.string"),df.var[which(char.df.var == max(char.df.var))], envir = .GlobalEnv)
    
    # Get all values in a column that contain the given string (either small or large)
    df.contains.values = grep(pattern = "[Ww]", x = df.var, value = TRUE)
    # insert [] brackets around the string if you want to search for all values that contain both upper and lower case values for the string. Here it displays all values with either w or W in the string
    df.contains.values = grep(pattern = "w", x = df.var, value = TRUE, ignore.case = T)
    #above is a similar way of doing the same operation
    
    # Get all the values in a column that start with a given character
    df.start.value = subset(df.var,substr(df.var,1,1) %in% c("W","w"))
    
    # To get the total number of times a particular character appears in all the strings in a vector
    string.list = c("a","e","i","o","u")
    num.string.list = vector(mode = "integer", length = 5)
    for (j in seq_along(string.list)) {
      num.temp = str_count(tolower(df.var), string.list[j])
      num.string.list[j] = sum(num.temp)
    }
    names(num.string.list) = string.list
  }
  
  # 29. A one-liner that removes all objects except for functions:
  rm(list = setdiff(ls(), lsf.str()))
  
  # 30. some more random functions
  {
    # 1: To clear the console and clear the environment
    {
      # x == 1, only environment
      # x == 2, console and environment
      {
        Cln.Env.Con <- function(x) {
          switch(x,
                 `1` = {
                   rm(list = setdiff(ls(pos = ".GlobalEnv"),c("Cln.Env.Con")), pos = ".GlobalEnv") },
                 `2` = {
                   cat("\014")
                   rm(list = setdiff(ls(pos = ".GlobalEnv"),c("Cln.Env.Con")), pos = ".GlobalEnv") })
          # garbage cleaning
          gc()
        }
      }
    }
    
    # 2: To get the statistics on the unique values in a column/columns
    {
      # if x == 1, get the number of unique values in a column
      # if x == 2, get the unique values in a column and store them in a variable
      # if x == 3, get the number of unique values in all columns
      # if x == 4, get the unique values in all columns and store them in a variable
      count.unique <- function(df,x,y = "")
      {
        df.name = deparse(substitute(df))
        df.y = paste0("df$",y)
        
        if (x == 1) {
          unique.count.df.y <- data.frame(nrow(unique(data.frame(eval(parse(text = df.y))))))
          colnames(unique.count.df.y) = c("#")
          assign(paste("unique.count",df.name,y,sep = "."),unique.count.df.y,envir = .GlobalEnv)
        }
        
        if (x == 2) {
          unique.df.y <- unique(data.frame(eval(parse(text = df.y))))
          colnames(unique.df.y) = c("Unique.Values")
          assign(paste("unique",df.name,y,sep = "."),unique.df.y, envir = .GlobalEnv)
        }
        
        if (x == 3) {
          unique.count.df <- data.frame(rapply(as.list(df),function(x)length(unique(x))))
          colnames(unique.count.df) = c("Unique.Table")
          assign(paste("unique.count",df.name,sep = "."),unique.count.df,envir = .GlobalEnv)
        }
        
        if (x == 4) {
          unique.df <- sapply(df,function(x)unique(x))
          unique.df = cbind.fill(unique.df)
          assign(paste("unique",df.name,sep = "."),unique.df, envir = .GlobalEnv)
        }
      }
      
    }
    
    # 3: To get univariate statistics based on different techniques
    {
      # x == 1, "whatis" from YaleToolkit
      # x == 2, "stat.desc" from pastecs (for continuous only)
      # x == 3, "Hmisc.Describe"
      #
    }
    
    # 4: To get percentiles, deciles, quantiles
    {
      # x == 1, to get percentiles (from 10% to 100%, in increments of 10) for every continuous column in the dataframe
      # x == 2, to get the quartiles (25%, 50%, 75%)
      
      tile.df <- function(df, x)
      {
        df_num <- df[ , (sapply(df, is.numeric))]
        df_char <- df[ , (sapply(df, is.character))]
        df.name = deparse(substitute(df))
        
        if (x == 1) {
          percentiles_num <- data.frame(lapply(df_num, quantile, probs = seq(0,1,0.1), na.rm = T))
          assign(paste(df.name, "percentiles", sep = "."), percentiles_num, envir = .GlobalEnv)
        }
        
        if (x == 2) {
          percentiles_num <- data.frame(lapply(df_num, quantile, na.rm = T))
          assign(paste(df.name, "percentiles", sep = "."), percentiles_num, envir = .GlobalEnv)
        }
        
        if (x == 3) {
          
        }
      }
    }
  }
  
  # 31. some kaggle feature exploration codes
{
# remove duplicate columns
remDupcols <- function(data){
  rem = which(!(names(data) %in% colnames(unique(as.matrix(data), MARGIN=2))))
  return(rem)
}

# fast parallel cor
bigcorPar <- function(x, nblocks = 10, verbose = TRUE, ncore="all", ...){
  library(ff, quietly = TRUE)
  require(doMC)
  if(ncore=="all"){
    ncore = multicore:::detectCores()
    registerDoMC(cores = ncore)
  } else{
    registerDoMC(cores = ncore)
  }

  NCOL <- ncol(x)

  ## test if ncol(x) %% nblocks gives remainder 0
  if (NCOL %% nblocks != 0){stop("Choose different 'nblocks' so that ncol(x) %% nblocks = 0!")}

  ## preallocate square matrix of dimension
  ## ncol(x) in 'ff' single format
  corMAT <- ff(vmode = "single", dim = c(NCOL, NCOL))

  ## split column numbers into 'nblocks' groups
  SPLIT <- split(1:NCOL, rep(1:nblocks, each = NCOL/nblocks))

  ## create all unique combinations of blocks
  COMBS <- expand.grid(1:length(SPLIT), 1:length(SPLIT))
  COMBS <- t(apply(COMBS, 1, sort))
  COMBS <- unique(COMBS)

  ## iterate through each block combination, calculate correlation matrix
  ## between blocks and store them in the preallocated matrix on both
  ## symmetric sides of the diagonal
  results <- foreach(i = 1:nrow(COMBS)) %dopar% {
    COMB <- COMBS[i, ]
    G1 <- SPLIT[[COMB[1]]]
    G2 <- SPLIT[[COMB[2]]]
    if (verbose) cat("Block", COMB[1], "with Block", COMB[2], "\n")
    flush.console()
    COR <- cor(x[, G1], x[, G2], ...)
    corMAT[G1, G2] <- COR
    corMAT[G2, G1] <- t(COR)
    COR <- NULL
  }

  gc()
  return(corMAT)
}

# remove highly correlated features from data
remHighcor <- function(data, cutoff, ...){
  data_cor <- cor(sapply(data, as.numeric), ...)
  data_cor[is.na(data_cor)] <- 0
  rem <- findCorrelation(data_cor, cutoff=cutoff, verbose=T)
  return(rem)
}

# remove highly correlated features from data faster
remHighcorPar <- function(data, nblocks, ncore, cutoff, ...){
  data_cor = bigcorPar(data, nblocks = nblocks, ncore = ncore, ...)
  data_cor = data.matrix(as.data.frame(as.ffdf(data_cor)))
  data_cor[is.na(data_cor)] <- 0
  rem <- findCorrelation(data_cor, cutoff=cutoff, verbose=T)
  return(rem)
}

# remove features from data which are highly correlated to those in main
remHighcor0 <- function(data, main, cutoff, ...){
  res = cor(sapply(data, as.numeric), sapply(main, as.numeric), ...) # res is names(data) X names(main) matrix
  res[is.na(res)] <- 0
  res = res > cutoff
  rem = unname(which(rowSums(res) > 0))
  return(rem)
}

# one-hot/dummy encode factors
# does not depend on train/test split as long as # of factors is same
categtoOnehot <- function(data, fullrank=T, ...){
  data[,names(data)] = lapply(data[,names(data)] , factor) # convert character to factors
  if (fullrank){
    res = as.data.frame(as.matrix(model.matrix( ~., data, ...)))[,-1]
  } else {
    res = as.data.frame(as.matrix(model.matrix(~ ., data, contrasts.arg = lapply(data, contrasts, contrasts=FALSE), ...)))[,-1]
  }
  return(res)
}

# orthogonal polynomial encoding for ordered factors - use only for a few 2-5 levels !
# not all features may actually be important and must be removed
# include dependence on train/test split ?
categtoOrthPoly <- function(data, fullrank=T, ...){
  data[,names(data)] = lapply(data[,names(data)] , ordered) # convert character to factors
  if (fullrank){
    res = as.data.frame(as.matrix(model.matrix( ~., data, ...)))[,-1]
  } else {
    res = as.data.frame(as.matrix(model.matrix(~ ., data, contrasts.arg = lapply(data, contrasts, contrasts=FALSE), ...)))[,-1]
  }
  return(res)
}

# Out-of-fold mean/sd/median deviation encoding.
# Useful for high cardinality categorical variables.
# always full rank
categtoDeviationenc <- function(char_data, num_data, traininds=NULL, funcs = funs(mean(.,na.rm=T), sd(.,na.rm=T), 'median' = median(.,na.rm=T))){

  if(length(traininds) == 0){
    train_char_data = char_data
    train_num_data = num_data
  } else {
    train_char_data = char_data[traininds, ]
    train_num_data =num_data[traininds, ]
  }

  res = list()
  for(x in names(train_char_data)){
    res[[x]] = train_num_data %>% group_by(.dots=train_char_data[,x]) %>% summarise_each(funcs) # calculate mean/sd/median encodings
    res[[x]][,-1] = apply(res[[x]][,-1], 2, scale, scale=FALSE, center=TRUE) # calculate deviances of mean/sd/median encodings
    # rename columns
    colnames(res[[x]])[1] = x
    if (ncol(train_num_data) == 1)
      colnames(res[[x]])[-1] = paste0(names(train_num_data),'_',names(res[[x]])[-1])
    res[[x]] <- merge(char_data[,x,drop=F], res[[x]], all.x=T, by=x)[,-1] # apply encodings to all data
  }
  res = data.frame(do.call(cbind, res))
  return(res)
}

# function to remove equivalent factors
remEquivfactors <- function(x.data, ref.data = NULL){
  if(length(ref.data) == 0L){
    all = x.data
  } else{
    all = data.frame(cbind(ref.data, x.data))
  }
  all[,names(all)] = lapply(all[,names(all), drop=F], function(l){
    as.numeric(reorder(x=l, X=seq(1:nrow(all)), FUN=mean))
  })
  rem = which(!(names(x.data) %in% colnames(unique(as.matrix(all), MARGIN=2, fromLast = F)))) # removal of cols towards end will be preferred
  return(rem)
}

# function to create n-way categorical-categorical interactions
nwayInterac <- function(char_data, n){
  nway <- as.data.frame(combn(ncol(char_data), n, function(y) do.call(paste0, char_data[,y])))
  names(nway) = combn(ncol(char_data), n, function(y) paste0(names(char_data)[y], collapse='.'))
  rem = remEquivfactors(x.data = nway, ref.data=NULL)
  if(length(rem) > 0)
    nway = nway[,-rem]
  return(nway)
}

# create xgbfi xlsx output
xgbfi <- function(xgbfi_path = 'code/xgbfi/bin/XgbFeatureInteractions.exe', # need to clone code from xgbfi repo
                  model_path,
                  output_path, # if filename then without xlsx extension
                  d = 3, # upper bound for extracted feature interactions depth
                  g = -1, # upper bound for interaction start deepening (zero deepening => interactions starting @root only)
                  t = 100, #upper bound for trees to be parsed
                  k = 100, # upper bound for exported feature interactions per depth level
                  s = 'Gain', # score metric to sort by (Gain, Fscore, wFScore, AvgwFScore, AvgGain, ExpGain)
                  h = 10 # amounts of split value histograms
){

  system(paste0('mono ',xgbfi_path,' -m ',model_path,' -o ',output_path,
                ' -d ',d,' -g ',g,' -t ',t,' -k ',k,' -s ',s,' -h ',h)) # saves output .xlsx file in given ouput directory

}
}
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
