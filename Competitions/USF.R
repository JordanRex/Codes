# All_Markets_doParallel_foreach_non-seasonal.R

# Details
# 1. All markets (no division filter)
# 2. Recreate = 1 (always from now, the variable will not be created henceforth)
# 3. Parallel - True (using doParallel)
# 4. Arima as a function


# Clear the environment and console once before start
# Do garbage memory cleaning as well to free memory before running
rm(list = setdiff(ls(), c()))
cat("\014")
gc()

# Setting the working directory to the current folder in all the instances so that the model output files are saved in the same folders
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Installing and loading packages
{
  packages <- function(x) {
    x <- as.character(match.call()[[2]])
    if (!require(x,character.only = TRUE)) {
      install.packages(pkgs = x, repos = "http://cran.r-project.org")
      require(x,character.only = TRUE)
    }
  }
  suppressMessages({
    #packages('rstudioapi')
    packages('forecast')
    packages('tseries')
    packages('sqldf')
    packages('MASS')
    packages('gtools')
    #packages('RODBC')
    packages('data.table')
    packages('plyr')
    packages('dplyr')
    #packages('Rmpi')
    #packages('doParallel')
    packages('doSNOW')
  })
}

#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#setwd(dirname(rstudioapi::callFun("getActiveDocumentContext")$path))

# Primary variables
{
  # Week number for which forecast should start
  forecast_week <- 201640
  # Current week is the latest week for which complete sales data is available
  # It is computed based on the above user input
  current_week <- forecast_week - 1

  # Date should be in mm-dd-yyyy format (Week start Date for forecast week - Should be a Sunday)
  week_start_date <- as.Date('10-02-2016',"%m-%d-%Y")

  # Market/Division name for which forecasts are to be generated - Use only one Market at a time.
  div_nm <- c("ALL_Markets")

  # File name if the data has already been downloaded from snowflake
  base_data_file_name <- paste0("base_data_snowflake_till_",week_start_date - 1,".csv")
}

################ Outlier treatment ########################
{
# Packages and Pre-Requisites
{
  source('vlookup.R')

  #Getting week mapping file
  weekly_mapping = read.csv('Week Mapping.csv', header = T, as.is = T)
}

#Getting treated data in base table
ads = read.csv(base_data_file_name, header = T, as.is = T)

#dummy_dataframe(ads)

# Creating the dummy skeleton to get all combinations and all weeks
{
  # Markets
  {
    market_id = data.frame(unique(ads$DIV_NM))
    market_id = market_id$unique.ads.DIV_NM.

    market_nbr = data.frame(unique(subset(ads, select = c("DIV_NBR","DIV_NM"))))
    market_nbr = market_nbr[,c(2,1)]
  }

  # Pyramid_Segments
  {
    Pyr_Segments = data.frame(unique(ads$SEGMENT))
    Pyr_Segments = Pyr_Segments$unique.ads.SEGMENT.

  }

  # PIM_Classes
  {
    PIM_Classes = data.frame(unique(ads$CLASS))
    PIM_Classes = PIM_Classes$unique.ads.CLASS.

    PIM_id = data.frame(unique(subset(ads, select = c("CLASS","PIM_CLS_ID"))))
    PIM_id = PIM_id[,c(2,1)]
  }

  # Year and Fiscal Week
  {
    YEAR = data.frame(unique(ads$YEAR))
    YEAR = YEAR$unique.ads.YEAR.

    Fiscal_week = c(1:52)
  }

  # Skeleton with all combinations
  {
    xxx = expand.grid(market_id,Pyr_Segments,PIM_Classes,YEAR,Fiscal_week)
    colnames(xxx) = c("DIV_NM","SEGMENT","CLASS","YEAR","Fiscal_Week")
    #Adding 53rd week to 2015 record
    xxx_2 = expand.grid(market_id,Pyr_Segments,PIM_Classes,2015,53)
    colnames(xxx_2) = c("DIV_NM","SEGMENT","CLASS","YEAR","Fiscal_Week")
    xxx = rbind(xxx,xxx_2)
    rm(xxx_2)

    xxx$DIV_NM = as.character(xxx$DIV_NM)
    xxx$SEGMENT = as.character(xxx$SEGMENT)
    xxx$CLASS = as.character(xxx$CLASS)
    xxx$YEAR = as.character(xxx$YEAR)
    xxx$Fiscal_Week = as.character(xxx$Fiscal_Week)
    xxx$Fiscal_Week = ifelse(nchar(xxx$Fiscal_Week)<2,paste0("0",xxx$Fiscal_Week),xxx$Fiscal_Week)
    xxx$FISC_YR_WK = paste0(xxx$YEAR,xxx$Fiscal_Week)

    xxx$Fiscal_Week = as.numeric(gsub(" ", "", xxx$Fiscal_Week, fixed = TRUE))
    xxx$FISC_YR_WK = as.numeric(gsub(" ", "", xxx$FISC_YR_WK, fixed = TRUE))
    xxx$YEAR = as.numeric(xxx$YEAR)
  }
}

# Next steps after creating skeleton -> Joining to get the dummy records, formatting to make a clean dataset, 52vs53 week treatment, the rolling week no
{
  # Initialising sales and cases to 0 for the empty records, creating year_month
  {
    # Creating fiscal week, month column for mapping, joining raw to dummy,
    ads$WEEK <- format(as.Date(ads$WEEK,"%Y-%m-%d"),"%d-%b-%y")
    ads$Fiscal_Week = vlookup(ads$WEEK, weekly_mapping, 2)

    weekly_mapping = weekly_mapping[,c(3,1,2,4:6)]
    xxx$MONTH = as.numeric(vlookup(xxx$FISC_YR_WK, weekly_mapping, 6))

    #Merging skeleton and actuals
    ads_test = full_join(xxx, ads)

    ads_test[is.na(ads_test$CASES) == T,"CASES"] = 0
    ads_test[is.na(ads_test$SALES) == T,"SALES"] = 0

    ads_test$CLNDR_WK_STRT_DT = NULL

    # create a year-month column
    ads_test = mutate(ads_test, YEAR_MONTH = YEAR*100 + MONTH)
  }

  # Intermediary dataset
  ADS = ads_test

  # Loading packages again to foolproof process
  suppressMessages(library(plyr))
  suppressMessages(library(dplyr))

  # 52vs53 WEEK treatment - applied only to the Training dataset which contains the year 2015
  {
    ADS_52vs53 = ADS[which(((ADS[,"Fiscal_Week"] == "52") | (ADS[,"Fiscal_Week"] == "53")) & (ADS[,"YEAR"] == "2015")),]
    ADS_x <- rbind(ADS, ADS_52vs53)
    ADS_x = ADS_x[! duplicated(ADS_x, fromLast = TRUE) & seq(nrow(ADS_x)) <= nrow(ADS), ]

    ADS_52vs53_2 = ddply(ADS_52vs53,c('DIV_NM','SEGMENT',"CLASS"),summarise,CASES = mean(CASES))

    ADS_52vs53_3 = merge(ADS_52vs53,ADS_52vs53_2, by = c("DIV_NM","SEGMENT","CLASS"))
    ADS_52vs53_3$CASES.x <- NULL

    ADS_52vs53_3 = subset(ADS_52vs53_3, ADS_52vs53_3$Fiscal_Week == 52)
    ADS_52vs53_3$CASES = ADS_52vs53_3$CASES.y
    ADS_52vs53_3$CASES.y = NULL

    ADS_x <- ADS_x[,order(colnames(ADS_x),decreasing = TRUE)]
    ADS_52vs53_3 <- ADS_52vs53_3[,order(colnames(ADS_52vs53_3),decreasing = TRUE)]
    ADS = rbind(ADS_x,ADS_52vs53_3)
  }

  # Cumulative Week number
  {
    ADS = ADS %>%
      dplyr::arrange(DIV_NM,SEGMENT,CLASS,YEAR,FISC_YR_WK) %>%
      dplyr::group_by(DIV_NM,SEGMENT,CLASS) %>%
      dplyr::mutate(CUM_FW = row_number())
  }

  rm(ADS_52vs53,ADS_52vs53_2,ADS_52vs53_3,ADS_x,Fiscal_week,market_id,PIM_Classes,Pyr_Segments,xxx,YEAR)

}

# OUTLIER TREATMENT
{
  # Process flow for Outlier
  {
    # Initial Cut
    # Î¼+3Ï = UCL
    # Î¼â3Ï = LCL
    # Relevance Test
    # Check of negatives - not used here
    # Null cases check
    # Trend/Seasonality check
    # Final treatment
  }

  #Getting the mean, sd, the outlier limits(right and left tails) and the 98th and 2nd percentiles
  {
    # user-defined functions to get the rolling mean and standard deviation, and the rolling percentile values
    my.rollapply <- function(vec, width, FUN)
      sapply(seq_along(vec),
             function(i) if (i < width) NA else FUN(vec[i:(i - width + 1)]))

    my.rollapply.quantile <- function(vec, width,x)
      sapply(seq_along(vec),
             function(i) if (i < width) NA else quantile(vec[i:(i - width + 1)],probs = x))

    ADS_avg_sd_2013 = ADS %>%
      dplyr::filter(YEAR == 2013) %>%
      dplyr::group_by(DIV_NM,SEGMENT,CLASS) %>%
      dplyr::mutate(AVG = mean(CASES)) %>%
      dplyr::mutate(STDEV = sd(CASES)) %>%
      dplyr::mutate(P_98 = quantile(CASES,0.98)) %>%
      dplyr::mutate(P_2 = quantile(CASES,0.2))

    ADS_avg_sd_rem = ADS %>%
      dplyr::arrange(DIV_NM,SEGMENT,CLASS,FISC_YR_WK) %>%
      dplyr::group_by(DIV_NM,SEGMENT,CLASS) %>%
      dplyr::mutate(AVG = dplyr::if_else(YEAR != 2013, my.rollapply(CASES,52,mean), 0)) %>%
      dplyr::mutate(STDEV = dplyr::if_else(YEAR != 2013, my.rollapply(CASES,52,sd), 0)) %>%
      dplyr::mutate(P_98 = dplyr::if_else(YEAR != 2013, my.rollapply.quantile(CASES,52,0.98), 0)) %>%
      dplyr::mutate(P_2 = dplyr::if_else(YEAR != 2013, my.rollapply.quantile(CASES,52,0.2), 0)) %>%
      dplyr::filter(YEAR != 2013)

    ADS = rbind(ADS_avg_sd_2013,ADS_avg_sd_rem)

    ADS$RT_value = (ADS$AVG) + (3*ADS$STDEV)
    ADS$LT_value = (ADS$AVG) - (3*ADS$STDEV)
  }

  #####SEASONALITY CHECK######
  {
    ADS = ADS %>%
      dplyr::mutate(RT = dplyr::if_else(CASES > RT_value, 1, 0)) %>%
      dplyr::mutate(LT = dplyr::if_else(LT_value > CASES, 1, 0))

    ADS$OT_2013 <- ifelse(ADS$RT == "1" & ADS$YEAR == "2013","Right tail",NA)
    left <- subset(ADS,is.na(ADS$OT_2013))
    right <- subset(ADS,!is.na(ADS$OT_2013))
    left$OT_2013 <- ifelse(left$LT == "1" & left$YEAR == "2013","Left tail",NA)
    ADS <- rbind(right,left)

    ADS$OT_2014 <- ifelse(ADS$RT == "1" & ADS$YEAR == "2014","Right tail",NA)
    left <- subset(ADS,is.na(ADS$OT_2014))
    right <- subset(ADS,!is.na(ADS$OT_2014))
    left$OT_2014 <- ifelse(left$LT == "1" & left$YEAR == "2014","Left tail",NA)
    ADS <- rbind(right,left)

    ADS$OT_2015 <- ifelse(ADS$RT == "1" & ADS$YEAR == "2015","Right tail",NA)
    left <- subset(ADS,is.na(ADS$OT_2015))
    right <- subset(ADS,!is.na(ADS$OT_2015))
    left$OT_2015 <- ifelse(left$LT == "1" & left$YEAR == "2015","Left tail",NA)
    ADS <- rbind(right,left)

    ADS$OT_2016 <- ifelse(ADS$RT == "1" & ADS$YEAR == "2016","Right tail",NA)
    left <- subset(ADS,is.na(ADS$OT_2016))
    right <- subset(ADS,!is.na(ADS$OT_2016))
    left$OT_2016 <- ifelse(left$LT == "1" & left$YEAR == "2016","Left tail",NA)
    ADS <- rbind(right,left)

    rm(left,right)

    ADS_1 <- setNames(aggregate(RT ~ DIV_NM + SEGMENT + CLASS + Fiscal_Week, ADS, sum),
                      c("DIV_NM","SEGMENT","CLASS","Fiscal_Week","RT_sum"))

    ADS_2 <- setNames(aggregate(LT ~ DIV_NM + SEGMENT + CLASS + Fiscal_Week, ADS, sum),
                      c("DIV_NM","SEGMENT","CLASS","Fiscal_Week","LT_sum"))

    ADS <- merge(ADS, ADS_1, by = c("DIV_NM","SEGMENT","CLASS","Fiscal_Week"))
    ADS <- merge(ADS, ADS_2, by = c("DIV_NM","SEGMENT","CLASS","Fiscal_Week"))

    ADS$Seasonality_check <- ifelse(ADS$RT_sum > 1, 1, ifelse(ADS$LT_sum > 1, 1, 0))
  }

  #####AVG CHECK####
  ADS = mutate(.data = ADS, Avg_check = ifelse(AVG < 15,1,0))

  #####NULL CHECK####
  {
    ADS = mutate(.data = ADS, Null_cases = ifelse(CASES == 0,1,0))
    ADS_1 <- aggregate(ADS$Null_cases,by = list(ADS$YEAR,ADS$DIV_NM,ADS$SEGMENT,ADS$CLASS),FUN = sum)
    colnames(ADS_1) <- c("YEAR","DIV_NM","SEGMENT","CLASS","NULL_TEST")

    ADS <- merge(ADS,ADS_1,by = c("YEAR","DIV_NM","SEGMENT","CLASS"))
    ADS$NULL_check <- ifelse(ADS$NULL_TEST > 3,1,0)
    ADS$NULL_TEST <- NULL
    rm(ADS_1)
  }

  #####TREATMENT######
  {
    ADS$TREAT_REQ <- ifelse(ADS$Avg_check == 0 & ADS$NULL_check == 0 & ADS$Seasonality_check == 0,"YES","NO")
    ADS_1 <- subset(ADS,ADS$TREAT_REQ == "NO")
    ADS_1$Treated_cases <- ADS_1$CASES
    ADS_2 <- subset(ADS,ADS$TREAT_REQ == "YES")

    ADS_2$TREAT_REQ <- ifelse((ADS_2$CASES > (ADS_2$AVG + (3*ADS_2$STDEV))),"Replace with P_98","NO")
    ADS_2_RT <- subset(ADS_2,ADS_2$TREAT_REQ == "Replace with P_98")
    ADS_2_RT$Treated_cases <- ADS_2_RT$P_98

    ADS_2_LT <- subset(ADS_2,ADS_2$TREAT_REQ=="NO")
    ADS_2_LT$TREAT_REQ <- ifelse(((ADS_2_LT$AVG - (3*ADS_2_LT$STDEV))>ADS_2_LT$CASES),"Replace with P_2","NO")
    ADS_2_LT$Treated_cases <- ifelse(ADS_2_LT$TREAT_REQ=="Replace with P_2",ADS_2_LT$P_2,ADS_2_LT$CASES)

    ADS_2 <- rbind(ADS_2_RT,ADS_2_LT)

    ADS <- rbind(ADS_1,ADS_2)

    rm(ADS_1,ADS_2,ADS_2_LT,ADS_2_RT)

    ADS$Diff <- ADS$Treated_cases - ADS$CASES
    ADS$Exact <- ifelse(ADS$Diff != 0,"FALSE","TRUE")
  }

  ADS$KEY = NULL

  # Next intermediary dataset creation
  ADS_Treated = ADS
}

# Adding Regressors
{
  # Take ADS_Treated
  ADS_Treated_Regressor = ADS_Treated
  regressor = read.csv('Regressors.csv', header = T, as.is = T)
  regressor[is.na(regressor)] = 0
  weekly_mapping = read.csv('Week Mapping.csv', header = T, as.is = T)
  weekly_mapping = weekly_mapping[,c(3,1:2,4:6)]
  ADS_Treated_Regressor$WEEK = NULL
  ADS_Treated_Regressor$CLNDR_WK_STRT_DT = vlookup(ADS_Treated_Regressor$FISC_YR_WK, weekly_mapping, 2)

  #Merging treated data with regressors
  ADS_Treated_Regressor = inner_join(ADS_Treated_Regressor, regressor)

  # Applying Business Rules on regressors -
  {
    {
      # 1.       IF SEGMENT NE "IND" THEN MK = 0;
      # 2.       IF CLASS = "CHEMICALS & CLEANING AGENTS" OR CLASS = "MEAT SUBSTITUTE" THEN MK = 0;
      # 3.       IF CLASS = "CHEMICALS & CLEANING AGENTS" OR CLASS = "MEAT SUBSTITUTE" THEN SUPERBOWL = 0;
      # 4.       IF CLASS = "CHEMICALS & CLEANING AGENTS" OR CLASS = "MEAT SUBSTITUTE" THEN marmad= 0;
      # 5.       IF CLASS = "EQUIPMENT & SUPPLIES" THEN NF = 0;
      # 6.       IF CLASS = "EQUIPMENT & SUPPLIES" THEN SUPERBOWL = 0;
      # 7.       IF CLASS = "EQUIPMENT & SUPPLIES" THEN marmad = 0;
    }

    ADS_Treated_Regressor$mk = ifelse(ADS_Treated_Regressor$SEGMENT == "IND", ADS_Treated_Regressor$mk,0)
    ADS_Treated_Regressor$mk = ifelse((ADS_Treated_Regressor$CLASS == "CHEMICALS & CLEANING AGENTS") |
                                        (ADS_Treated_Regressor$CLASS == "MEAT SUBSTITUTE"), 0, ADS_Treated_Regressor$mk)
    ADS_Treated_Regressor$superbowl = ifelse((ADS_Treated_Regressor$CLASS == "CHEMICALS & CLEANING AGENTS") |
                                               (ADS_Treated_Regressor$CLASS == "MEAT SUBSTITUTE"), 0, ADS_Treated_Regressor$superbowl)
    ADS_Treated_Regressor$marmad = ifelse((ADS_Treated_Regressor$CLASS == "CHEMICALS & CLEANING AGENTS") |
                                            (ADS_Treated_Regressor$CLASS == "MEAT SUBSTITUTE"), 0, ADS_Treated_Regressor$marmad)
    ADS_Treated_Regressor$nf = ifelse(ADS_Treated_Regressor$CLASS == "EQUIPMENT & SUPPLIES",
                                      0, ADS_Treated_Regressor$nf)
    ADS_Treated_Regressor$superbowl = ifelse(ADS_Treated_Regressor$CLASS == "EQUIPMENT & SUPPLIES",
                                             0, ADS_Treated_Regressor$superbowl)
    ADS_Treated_Regressor$marmad = ifelse(ADS_Treated_Regressor$CLASS == "EQUIPMENT & SUPPLIES",
                                          0, ADS_Treated_Regressor$marmad)
  }
}

# Adding Div_nbr and PIM_id
{
  ADS_Treated_Regressor$DIV_NBR <- NULL
  ADS_Treated_Regressor$PIM_CLS_ID <- NULL
  ADS_Treated_Regressor =  inner_join(ADS_Treated_Regressor,market_nbr)
  ADS_Treated_Regressor =  inner_join(ADS_Treated_Regressor,PIM_id)
}

# Initialising variable that deletes all unnecessary columns
rem_cols <- c("AVG",	"STDEV",	"RT_value",	"LT_value",	"P_98",	"P_2",	"RT",	"LT",	"OT_2013",	"OT_2014",	"OT_2015",	"OT_2016",	"RT_sum",	"LT_sum",	"Seasonality_check",	"Avg_check",	"Null_cases",	"NULL_check",	"TREAT_REQ",	"Diff",	"Exact")

#Filtering final columns
ADS_Treated_Regressor = ADS_Treated_Regressor[-which(colnames(ADS_Treated_Regressor) %in% rem_cols)]
ads_treated_regressor_data_file_name <- paste0(div_nm,'_ADS_Treated_Regressor_data',current_week,'.csv')

# Remove all objects in environment that is not needed
rm(list = setdiff(ls(), c("ADS","ADS_Treated_Regressor","vlookup","timestamp_log","user_id", "password","forecast_week","current_week","week_start_date","div_nm","base_data_file_name","DCT_raw1_file_name","DCT_raw2_file_name")))
}
#####################################################################################################

{
  #Get outlier treated data
  ads <- ADS_Treated_Regressor
  ads$date_st <- as.Date(ads$CLNDR_WK_STRT_DT, "%d-%b-%y")
  ads$CASES <- NULL
  ads$CASES <- ads$Treated_cases
  ads$Treated_cases <- NULL

  #Sorting columns
  ads = ads[,c(1,86:89,2:85)]

  #Removing the variables that are highly corelated
  {
    xreg_all <- ads[,which(colnames(ads) == 'newyear'):which(colnames(ads) == 'mk')]
    cc <- cor(xreg_all)
    diag(cc) <- 0
    cc[upper.tri(cc)] <- 0
    transpose = as.data.frame(as.table(cc))
    remove_col <- subset(transpose, Freq >= 0.7)
    ads <- ads[-which(colnames(ads) %in% remove_col$Var1)]
  }

  #Remove unwanted data frames
  {
    rm(list = setdiff(ls(),c("ads","timestamp_log","ADS_Treated_Regressor","vlookup", "user_id", "password","forecast_week","current_week","week_start_date","div_nm","base_data_file_name","DCT_raw1_file_name","DCT_raw2_file_name")))
  }

  #Initializing variables and data frames
  i <- 1
  log_transformed <- 0
  only_arima <- 0
  model_type <- ""
  cnt_insample_na <- 0
  a <- 0
  b <- 0

  # Creating the all combinations file
  {
    all_combinations <- data.frame(distinct(ads[c("DIV_NM","SEGMENT","CLASS")]))
    all_combinations$key <- 1:nrow(all_combinations)

    #Getting key to keep track of the models
    all_combinations_file_name <- paste0(div_nm,"_combination_key_",current_week,".csv")
    write.csv(all_combinations,all_combinations_file_name, row.names = F)
  }
}

# Initialization of the parallel (Rmpi/doParallel backend)
{
  Sys.setenv(PATH = paste0(Sys.getenv('PATH'), ':/usr/lib/rstudio-server/bin/postback'))
  # 
  # system('ssh-keyscan  -t rsa ip-172-31-0-18 >> ~/.ssh/known_hosts')
  # system('ssh-keyscan  -t rsa ip-172-31-10-6 >> ~/.ssh/known_hosts')
  # system('ssh-keyscan  -t rsa ip-172-31-5-205 >> ~/.ssh/known_hosts')
  # 
  #system('ssh -oStrictHostKeyChecking=no spark@172.31.1.66')
  
  cluster_nodes = c(rep("localhost", 10), rep("slave1", 32), rep("slave2", 32), rep("slave3", 32))
  cl = makeCluster(cluster_nodes, type = 'PSOCK')
  cluster_spec = clusterCall(cl, function() Sys.info()[c("nodename","machine")])
  print(cluster_spec)
  # cl = makeCluster(5, type = "MPI")
  registerDoParallel(cl)
}

Start.time = Sys.time()
saveRDS(Start.time,'Start_time.rds')

# Initialization of the parallel (doParallel backend) (for a single instance)
 # {
 #   cl = makeCluster(detectCores() - 1)
 #   registerDoParallel(cl)
 # }

packages_list = c("forecast","plyr","dplyr","tseries","MASS","data.table")

ads_list = split(ads, with(ads, interaction(DIV_NM,SEGMENT,CLASS), drop = T))

source('arima_function.R')
max_iterator = 720

# Model snippet
{
  #Running codes for the division
  foreach(i = 1:max_iterator, .packages = packages_list, .verbose = T, .export = c("ads_list", "arima_function", "current_week", "all_combinations"), .inorder = F) %dopar%
  {
    gc()
    arima_function(ads = data.frame(ads_list[[i]]), i = i)
  }
}

stopCluster(cl)

forecast_allwk_final_base_comb <- data.frame()
forecast_pre_52 <- data.frame()
all_combinations <- data.frame()
pdq <- data.frame()
loop_time = data.frame()

# Binding all the individual files created in the loop
{
for (i in 1:max_iterator) {
  load(paste0("saved_variables_",i,".rda"), envir = .GlobalEnv)
  assign(paste0("loop_time_", i), readRDS(paste0("loop_time_",i,".rds")), envir = .GlobalEnv)

  all_combinations = rbind(all_combinations, get(paste0("all_combinations_", i))[i,])
  forecast_allwk_final_base_comb = rbind(forecast_allwk_final_base_comb, get(paste0("forecast_final_base_comb_", i)))
  forecast_pre_52 = rbind(forecast_pre_52, get(paste0("forecast_pre_52_", i)))
  pdq = rbind(pdq, get(paste0("pdq_", i)))
}

for (i in 1:max_iterator) {
  loop_time = rbind(loop_time, get(paste0("loop_time_", i)))
}
}

rm(list = ls(pattern = "all_combinations_"))
rm(list = ls(pattern = "forecast_final_base_comb_"))
rm(list = ls(pattern = "loop_time_"))
rm(list = ls(pattern = "pdq_"))
rm(list = ls(pattern = "forecast_pre_52_"))

# writing the outputs
{
  #Export pdq values
  all_combinations_pdq <- paste0(div_nm,"_all_combinations_pdq_", current_week,".csv")
  write.csv(all_combinations, all_combinations_pdq, row.names = F)

  #Export data for weekly forecast
  file_name_allwk <- paste0(div_nm,"_forecast_allwk_", current_week,".csv")
  write.csv(forecast_allwk_final_base_comb, file_name_allwk, row.names = F)

  #Export data 52 week validation
  file_name_52wk_pre <- paste0(div_nm,"_forecast_val_pre_52wk_", current_week,".csv")
  write.csv(forecast_pre_52,file_name_52wk_pre, row.names = F)

  # Weighted accuracy
  rolledup_accuracy <- ddply(forecast_pre_52,c('DIV_NBR','DIV_NM','SEGMENT',"CLASS"),summarise,mean_acc = mean(accuracy),weighted_acc = sum(CASES*accuracy)/sum(CASES))

  # Printing rolledup accuracy values to veiw model health
  pre_52_wk_rolledup_accuracy_file_name <- paste0(div_nm,"_pre_52_wk_rolled_up_accuracy_",current_week,".csv")
write.csv(rolledup_accuracy,pre_52_wk_rolledup_accuracy_file_name, row.names = F)
print("Rolledup Accuracy file written")
}

End.time = Sys.time()
saveRDS(End.time,'End_time.rds')

Total_time = End.time - Start.time
saveRDS(Total_time,'Total_time.rds')

sink(Total_time)
