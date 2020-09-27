# Just Profiling for all BCs
# creating 3 profiling sets for (low, medium and high)

# clear environment, console, garbage collection
rm(list = setdiff(ls(), c()))
cat("\014")
gc()

# set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# install and load packages
{
  packages <- function(x) {
    x <- as.character(match.call()[[2]])
    if (!require(x,character.only = TRUE)) {
      install.packages(pkgs = x, repos = "http://cran.r-project.org", dependencies = T, verbose = T)
      require(x,character.only = TRUE)
    }
  }
  # Fill below snippet with required packages
  suppressMessages({
    packages('data.table')
    packages('dummies')
    packages('xgboost')
    #packages('mlbench')
    #packages('caret')
    #packages('randomForest')
    #packages('party')
    #packages('ranger')
    #packages('mlr')
    packages('dplyr')
    packages('rattle')
    packages('rpart.plot')
    packages('RColorBrewer')
    packages('rpart')
  })
}

# read and save all files
output_files = list.files(pattern = "NPD_")
size_files = list.files(pattern = "Size_")
BC = sub(pattern = ".txt", x = sub(pattern = "Size_.*_", x = size_files, replacement = ""), replacement = "")

for (iter_output in 1:length(output_files))
{
  final_out = read.csv(file = output_files[iter_output], header = T, as.is = T) %>% mutate(X = NULL)

  final_out_Y = final_out %>%
    group_by(Base_Product_Number, Year_Week_Number) %>%
    mutate(vol_sum = sum(vol)) %>%
    ungroup() %>%
    group_by(Base_Product_Number) %>%
    summarise(vol = mean(vol_sum))

  assign("BC_buckets", quantile(final_out_Y$vol, probs = c(0.33, 0.67), type = 8, names = F), envir = .GlobalEnv)

  final_out_Y = final_out_Y %>%
    mutate(vol_bucket = if_else(vol < BC_buckets[1], "Low",
                                if_else(vol < BC_buckets[2], "Medium", "High")))

  {
    final_out_attributes = final_out %>%
      group_by(Base_Product_Number) %>%
      summarise(#BrandName = unique(Brand_Name),
        BrandInd = unique(Brand_ind),
        PSGCode = unique(Product_Sub_Group_Code),
        Package = unique(Package_Type),
        #TillRoll = unique(Till_Roll_Description),
        #MerchGrp = unique(MERCHANDISE_GROUP_CODE),
        Buyer = unique(BUYER),
        Supplier = unique(parent_supplier),
        ASP = mean(as.numeric(asp), na.rm = T),
        ACP = mean(as.numeric(acp), na.rm = T),
        NumSubsSameBrand = mean(as.numeric(no_of_subs_same_brand), na.rm = T),
        NumSubsDiffBrand = mean(as.numeric(no_of_subs_diff_brand), na.rm = T),
        ProSamePriceBand = mean(as.numeric(price_band_prod_count), na.rm = T),
        ProSamePSG = mean(as.numeric(PSG_prod_count), na.rm = T),
        WeeksSold = n_distinct(Year_Week_Number)) %>%
      data.frame()

    No_of_Stores_data <- final_out %>% group_by(Base_Product_Number, Year_Week_Number) %>% summarize(No_of_Stores = sum(NO_stores)) %>% ungroup() %>% group_by(Base_Product_Number) %>% summarise(No_of_Stores = mean(No_of_Stores))

    final_out_attributes <- merge(final_out_attributes, No_of_Stores_data, by = c("Base_Product_Number"), all.x = T)
    }

  final_out = inner_join(final_out, final_out_Y[, c(1, 3)])

  final_out = final_out %>%
    group_by(Base_Product_Number) %>%
    filter(length(unique(Year_Week_Number)) >= 5) %>%
    ungroup() %>%
    arrange(Base_Product_Number, Area_Price_Code, Year_Week_Number)

  #save.image("profiling_backup.RData")
  #load("profiling_backup.RData")

  # creating the 3 main datasets
  final_out_low = final_out %>% filter(vol_bucket == "Low")
  final_out_medium = final_out %>% filter(vol_bucket == "Medium")
  final_out_high = final_out %>% filter(vol_bucket == "High")

  # now the final changes
  {
    # Low
    final_out_low_rolled <- final_out_low %>%
      group_by(Base_Product_Number) %>%
      summarise(Actual = sum(vol, na.rm = T),
                Predicted = sum(out, na.rm = T)) %>%
      data.frame()
    final_out_medium_rolled <- final_out_medium %>%
      group_by(Base_Product_Number) %>%
      summarise(Actual = sum(vol, na.rm = T),
                Predicted = sum(out, na.rm = T)) %>%
      data.frame()
    final_out_high_rolled <- final_out_high %>%
      group_by(Base_Product_Number) %>%
      summarise(Actual = sum(vol, na.rm = T),
                Predicted = sum(out, na.rm = T)) %>%
      data.frame()


    {
      final_out_low <- merge(final_out_low_rolled, final_out_attributes, by = c("Base_Product_Number"), all.x = TRUE)
      final_out_medium <- merge(final_out_medium_rolled, final_out_attributes, by = c("Base_Product_Number"), all.x = TRUE)
      final_out_high <- merge(final_out_high_rolled, final_out_attributes, by = c("Base_Product_Number"), all.x = TRUE)
      }


    size <- read.csv(file = size_files[iter_output], sep = ";", header = T, stringsAsFactors = F)
    colnames(size) <- c("Base_Product_Number", "Size", "Measure_Type")

    {
      final_out_low <- merge(final_out_low, size, by = c("Base_Product_Number"), all.x = T)
      final_out_medium <- merge(final_out_medium, size, by = c("Base_Product_Number"), all.x = T)
      final_out_high <- merge(final_out_high, size, by = c("Base_Product_Number"), all.x = T)
    }

    {
      final_out_low$APE <- abs(final_out_low$Actual - final_out_low$Predicted)/
        abs(final_out_low$Actual)
      final_out_medium$APE <- abs(final_out_medium$Actual - final_out_medium$Predicted)/
        abs(final_out_medium$Actual)
      final_out_high$APE <- abs(final_out_high$Actual - final_out_high$Predicted)/
        abs(final_out_high$Actual)
    }
  }


  final_out_low = final_out_low[final_out_low$Actual > 50, ]
  final_out_medium = final_out_medium[final_out_medium$Actual > 50, ]
  final_out_high = final_out_high[final_out_high$Actual > 50, ]

  {
    # LOW
    is.na(final_out_low$NumSubsDiffBrand) = 0
    is.na(final_out_low$NumSubsSameBrand) = 0
    final_out_low$ACPASPPERC = (abs(final_out_low$ASP - final_out_low$ACP)/final_out_low$ACP)*100

    # MEDIUM
    is.na(final_out_medium$NumSubsDiffBrand) = 0
    is.na(final_out_medium$NumSubsSameBrand) = 0
    final_out_medium$ACPASPPERC = (abs(final_out_medium$ASP - final_out_medium$ACP)/final_out_medium$ACP)*100

    # HIGH
    is.na(final_out_high$NumSubsDiffBrand) = 0
    is.na(final_out_high$NumSubsSameBrand) = 0
    final_out_high$ACPASPPERC = (abs(final_out_high$ASP - final_out_high$ACP)/final_out_high$ACP)*100
  }

  # Creating the rpart fit models
  {
    fit_low = rpart(APE ~ BrandInd+ACPASPPERC+Package+Buyer+ASP+ACP+NumSubsSameBrand+NumSubsDiffBrand+ProSamePriceBand+ProSamePSG+WeeksSold+Size+Measure_Type, data = final_out_low, control = rpart.control(minbucket = 4, maxdepth = 8))

    fit_medium = rpart(APE ~ BrandInd+ACPASPPERC+Package+Buyer+ASP+ACP+NumSubsSameBrand+NumSubsDiffBrand+ProSamePriceBand+ProSamePSG+WeeksSold+Size+Measure_Type, data = final_out_medium, control = rpart.control(minbucket = 4, maxdepth = 8))

    fit_high = rpart(APE ~ BrandInd+ACPASPPERC+Package+Buyer+ASP+ACP+NumSubsSameBrand+NumSubsDiffBrand+ProSamePriceBand+ProSamePSG+WeeksSold+Size+Measure_Type, data = final_out_high, control = rpart.control(minbucket = 4, maxdepth = 8))
  }

  #fancyRpartPlot(fit11)
  #asRules(fit11)

  iteration = c("low", "medium", "high")
  final_out_iterations = c("final_out_low", "final_out_medium", "final_out_high")

  for (i in 1:length(iteration))
  {
    fit = get(paste0("fit_", iteration[i]))
    prp(fit, type = 2, fallen.leaves = T, digits = 4, nn = T, ni = T)
    png(paste0(BC[iter_output], "_", i, ".png"), width = 1000, height = 1000,units = "px", pointsize = 12, bg = "white", res = NA, restoreConsole = TRUE)
    rpart.plot(fit, digits = 4)
    dev.off()
    rm(fit)
  }

  Actual_Good_APE_Threshold = matrix(nrow = 3, ncol = 1)

  Actual_Good_APE_Threshold[1,1] = as.numeric(mean(final_out_low$APE) + 0.1*mean(final_out_low$APE))
  Actual_Good_APE_Threshold[2,1] = as.numeric(mean(final_out_medium$APE) + 0.1*mean(final_out_medium$APE))
  Actual_Good_APE_Threshold[3,1] = as.numeric(mean(final_out_high$APE) + 0.1*mean(final_out_high$APE))

  # to save the node ids for each tree, and their corresponding APE values
  node_ids = list()
  node_id_flag_table = list()

  for (i in 1:length(iteration))
  {
    fit = get(paste0("fit_", iteration[i]))
    node_ids[[i]] = as.vector(dimnames(table(fit$where)))
    node_id = as.numeric(as.vector(unlist(node_ids[[i]])))
    fit_frame = data.frame(fit$frame, row.names = NULL) %>%
      mutate(node_ids = row_number(var)) %>%
      filter(row_number(var) %in% node_id) %>%
      mutate(profile = paste0("Profile_set_", i)) %>%
      rowwise() %>%
      mutate(profile_AGAT = Actual_Good_APE_Threshold[i,1]) %>%
      select(profile, profile_AGAT, node_ids, yval) %>%
      mutate(Good_or_Bad = if_else(yval < profile_AGAT, "GOOD", "BAD"))

    node_id_flag_table[[i]] = fit_frame
  }

  confusion_matrix = list()

  for (i in 1:length(iteration)) {
    x = cbind(get(final_out_iterations[i])[,c("Base_Product_Number", "APE")],
              data.frame(node_ids = data.frame(get(paste0("fit_", iteration[i]))$where)[,1]))
    y = inner_join(x, node_id_flag_table[[i]][,c(2,3,4,5)]) %>%
      mutate(True_or_False = if_else(APE < profile_AGAT, "True", "False")) %>%
      rename(Positive_or_Negative = Good_or_Bad) %>%
      mutate(cm = paste0(True_or_False,"_",Positive_or_Negative)) %>%
      group_by(cm) %>%
      summarise(cm_table = n()) %>%
      mutate(profile_set = i,
             BC = BC[iter_output])
    confusion_matrix[[i]] = y
    rm(x,y)
  }

  confusion_matrix_df = bind_rows(confusion_matrix)
  saveRDS(confusion_matrix_df, paste0("confusion_matrix_", BC[iter_output]))
}

confusion_matrix_all = list()
for (i in 1:length(output_files))
{
  confusion_matrix_all[[i]] = readRDS(paste0("confusion_matrix_", BC[i]))
}

confusion_matrix_all = bind_rows(confusion_matrix_all)
saveRDS(confusion_matrix_all, "confusion_matrix_all.rds")
View(confusion_matrix_all)
