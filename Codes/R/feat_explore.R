# some kaggle feature exploration codes

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


# function to extract calendar features
# function to call in holidays open data / other open data from python/othersources ?
# function to calcualte mode?

# time series outliers
# time series outliers
tsoutliers <- function(x,plot=FALSE)
{
  x <- as.ts(x)
  if(frequency(x)>1)
    resid <- stl(x,s.window="periodic",robust=TRUE)$time.series[,3]
  else
  {
    tt <- 1:length(x)
    resid <- residuals(loess(x ~ tt))
  }
  resid.q <- quantile(resid,prob=c(0.25,0.75))
  iqr <- diff(resid.q)
  limits <- resid.q + 1.5*iqr*c(-1,1)
  score <- abs(pmin((resid-limits[1])/iqr,0) + pmax((resid - limits[2])/iqr,0))
  if(plot)
  {
    plot(x)
    x2 <- ts(rep(NA,length(x)))
    x2[score>0] <- x[score>0]
    tsp(x2) <- tsp(x)
    points(x2,pch=19,col="red")
    return(invisible(score))
  }
  else
    return(score)
}
