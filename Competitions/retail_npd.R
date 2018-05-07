# without parallelization and loading provisions
# for now the BC part is not dynamic, change names of Main_file and Seasonality index accordingly

# NPD - MONTHLY - RANGER & XGBOOST - RANGED STORES - ME/EXP LEVEL


# The Beginning ####
{
  # Pre-Requisites
  {
    # TESCO STRATEGY
    # Predicting sales for new products using xgboost
    # clear environment, console, garbage collection
    rm(list = setdiff(ls(), c()))
    cat("\014")
    gc()
    
    # set working directory
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
    
    # setting libraries
    .libPaths(c("/home/vishal.raju/R/packages", "/usr/lib64/R/library", "/usr/share/R/library", "/home/vishal.raju/R/x86_64-redhat-linux-gnu-library/3.3"))
    
    
    # install and load packages
    {
      packages <- function(x) {
        x <- as.character(match.call()[[2]])
        if (!require(x,character.only = TRUE)) {
          install.packages(pkgs = x, repos = "http://cran.r-project.org", dependencies = T, verbose = T)
          require(x, character.only = TRUE)
        }
      }
      # Fill below snippet with required packages
      suppressMessages({
        packages('dplyr')
        packages('data.table')
        packages('dummies')
        packages('xgboost')
        packages('caret')
        #packages('party')
        packages('ranger')
        packages('magrittr')
        packages('stringdist')
        packages('rpart')
        packages("matrixStats")
      })
    }
  }
  
  # Iterator segment
  Time_start = 201101
  Time_end = 201601
  Time_end_actual_validation = 201701
  
  BC <- c("Meat Fish and Veg")
  
  cat("\n\n")
  print("The Beginning - Completed")
  cat("\n\n")
}


# ADS ####
{
  # Snippet specific to a BC (initialization with respect to the BC)
  {
    Main_file <- paste0(BC,'_MONTHLY_ADS_05142017_ST_MONTH_RANGED', '.txt')
    SI_file <- paste0(BC, "_Seasonality_Index", '.csv')
    Size_file <- paste0('Size_', BC, '.txt')
  }
  
  # ADS reading and filtering segment
  {
    # Reading the input raw ADS from its rds file
    ADS_6YR = fread(input = Main_file, sep = ";", header = T, na.strings = c('?', '#N/A'), stringsAsFactors = F, fill = T, quote = "")
    #saveRDS(ADS_6YR, 'ADS.rds')
    #ADS_6YR = readRDS(paste0(BC, "_ADS.rds"))
    
    # some temporary name changes
    ADS_6YR = ADS_6YR %>%
      rename(ME_EXP = store_type)
    
    convert_single_to_double = function(x) {
      if_else(nchar(x) < 2,
              paste0("0", x),
              as.character(x))
    }
    
    # this is the place where the loop for time frame iteration needs to be made
    ADS_6YR_iter2 = data.frame(ADS_6YR) %>%
      mutate(Period_Number = as.character(Period_Number)) %>%
      mutate(Period_Number = if_else(nchar(Period_Number) < 2,
                                     paste0("0", Period_Number),
                                     Period_Number),
             Year_Month = as.numeric(paste0(Year_Number, Period_Number))) %>%
      filter(Year_Month >= Time_start & Year_Month < Time_end_actual_validation) %>%
      mutate(Period_Number = as.numeric(Period_Number),
             Launch_Year_Month = as.numeric(paste0(substr(Launch_Tesco_Week, 1, 4), if_else(nchar(Launch_Month) < 2, paste0("0", Launch_Month), as.character(Launch_Month)))),
             Year_Quarter = as.numeric(paste0(Year_Number, convert_single_to_double(Quarter_Number))))
    
    # the tesco launch week/month snippet (probably not required)
    {
      ADS_TLW = ADS_6YR_iter2 %>%
        select(Base_Product_Number, Year_Month, Launch_Year_Month) %>%
        group_by(Base_Product_Number) %>%
        mutate(Launch_Year_Month = if_else(min(Year_Month) < Launch_Year_Month, min(Year_Month), Launch_Year_Month)) %>%
        ungroup() %>%
        arrange(Base_Product_Number, Year_Month) %>%
        mutate(Months_Since_Launch = 12*(as.numeric(substr(Year_Month, 1, 4)) - as.numeric(substr(Launch_Year_Month, 1, 4))) +
                 as.numeric(substr(Year_Month, 5, 6)) - as.numeric(substr(Launch_Year_Month, 5, 6))) %>%
        distinct()
      
      ADS_6YR_iter2 = ADS_6YR_iter2 %>% select(-Launch_Year_Month)
      ADS_6YR_iter2 = left_join(ADS_6YR_iter2, ADS_TLW) #%>% filter(0 <= Months_Since_Launch & Months_Since_Launch <= 100)
      rm(ADS_TLW)
      }
    
    # adding the launch year quarter
    ADS_6YR_iter2 = ADS_6YR_iter2 %>%
      mutate(Launch_Year_Quarter = as.numeric(paste0(substr(as.character(Launch_Year_Month), 1, 4),
                                                     convert_single_to_double(as.character(ceiling(as.numeric(Launch_Month) / 3))))))
  }
  
  # Treating negative sales
  {
    # To simply remove all such weeks
    # ADS_6YR <- ADS_6YR[as.numeric(ADS_6YR$vol) > 0 ,]
    # Results are better with inclusion of the negative sales as well
    # ADS_6YR = ADS_6YR
    # Treating all such weeks to 0
    ADS_6YR_iter2$vol = if_else(ADS_6YR_iter2$vol < 0, 0, as.numeric(ADS_6YR_iter2$vol))
  }
  
  # size, holidays, SI and calender tables
  {
    Size = read.table(Size_file, sep = ";", header = T, as.is = T)
    colnames(Size)[1] <- "Base_Product_Number"
    colnames(Size)[2] <- "Size"
    colnames(Size)[3] <- "measure_type"
    
    Holidays <- read.csv('UK_Holidays.csv')
    
    # Seasonality index
    SI_PSG <- read.csv(SI_file, header = T, as.is = T) %>%
      mutate(X = NULL)
    
    ADS_6YR_iter2 = left_join(ADS_6YR_iter2, SI_PSG[, c('months','psg', 'adjusted_index')], by = c("Period_Number" = "months", "Product_Sub_Group_Code" = "psg")) %>%
      rename(SI = adjusted_index)
    
    
    calendar = read.table("Calendar_Table.txt", sep = ";", header = T) %>% filter(Year_Week_Number > 201101) %>% select(Period_Number, Week_Number) %>% distinct()
    #calendar2 = read.table("Calendar_Table.txt", sep = ";", header = T) %>% filter(Year_Week_Number > 201101) %>% select(-Week_Number)
  }
  
  # The entire ranged stores replacing the sold stores values segment
  {
    #   Store_Count1 = read.table("Store_Count_BPN_APC_YWN.txt", sep = ";", header = T, stringsAsFactors = F, quote = "") %>%
    #     dplyr::rename_(.dots = setNames(names(.), c("Base_Product_Number", "Year_Week_Number", "Area_Price_Code", "NO_stores1", "NO_PFS_stores1", "NO_5k_stores1", "NO_20k_stores1", "NO_50k_stores1", "NO_100k_stores1", "NO_100kplus_stores1"))) %>%
    #     inner_join(., calendar2) %>%
    #     group_by(Base_Product_Number, Area_Price_Code, Year_Number, Period_Number) %>%
    #     summarise_all(funs(max)) %>%
    #     select(-Year_Week_Number) %>%
    #     mutate(ME_EXP = if_else(Area_Price_Code %in% c(2, 3, 4, 6), "ME", "EXP"),
    #            Period_Number = as.character(Period_Number),
    #            Period_Number = if_else(nchar(Period_Number) < 2,
    #                                    paste0("0", Period_Number),
    #                                    Period_Number),
    #            Year_Month = as.numeric(paste0(Year_Number, Period_Number))) %>%
    #     ungroup() %>%
    #     select(-Period_Number, -Area_Price_Code, -Year_Number) %>%
    #     group_by(Base_Product_Number, ME_EXP, Year_Month) %>%
    #     summarise_all(funs(sum))
    #
    #   ADS_6YR_iter2 = left_join(ADS_6YR_iter2, Store_Count1)
    #
    #   Store_Count2 = read.table("Store_Count_BPN_APC_YWN_new.txt", sep = ";", header = T, stringsAsFactors = F, quote = "") %>%
    #     dplyr::rename_(.dots = setNames(names(.), c("Base_Product_Number", "Year_Week_Number", "Area_Price_Code", "NO_stores2", "NO_PFS_stores2", "NO_5k_stores2", "NO_20k_stores2", "NO_50k_stores2", "NO_100k_stores2", "NO_100kplus_stores2"))) %>%
    #     inner_join(., calendar2) %>%
    #     group_by(Base_Product_Number, Area_Price_Code, Year_Number, Period_Number) %>%
    #     summarise_all(funs(max)) %>%
    #     select(-Year_Week_Number) %>%
    #     mutate(ME_EXP = if_else(Area_Price_Code %in% c(2, 3, 4, 6), "ME", "EXP"),
    #            Period_Number = as.character(Period_Number),
    #            Period_Number = if_else(nchar(Period_Number) < 2,
    #                                    paste0("0", Period_Number),
    #                                    Period_Number),
    #            Year_Month = as.numeric(paste0(Year_Number, Period_Number))) %>%
    #     ungroup() %>%
    #     select(-Period_Number, -Area_Price_Code, -Year_Number) %>%
    #     group_by(Base_Product_Number, ME_EXP, Year_Month) %>%
    #     summarise_all(funs(sum))
    #
    #   ADS_6YR_iter2 = left_join(ADS_6YR_iter2, Store_Count2)
    #
    #   drop_new_columns = c("NO_stores1","NO_PFS_stores1","NO_5k_stores1","NO_20k_stores1","NO_50k_stores1","NO_100k_stores1","NO_100kplus_stores1","NO_stores2","NO_PFS_stores2","NO_5k_stores2","NO_20k_stores2","NO_50k_stores2","NO_100k_stores2","NO_100kplus_stores2")
    #   drop_old_columns = c("NO_stores","NO_PFS_Stores","NO_5K_Stores","NO_20K_Stores","NO_50K_Stores","NO_100K_Stores","NO_100Kplus_Stores")
    #
    #   ADS_6YR_iter2 = ADS_6YR_iter2 %>%
    #     mutate(NO_stores_ranged = if_else(is.na(if_else(!(is.na(NO_stores1)) & !(is.na(NO_stores2)) & (NO_stores1 < NO_stores2), NO_stores2, NO_stores1)) & !(is.na(NO_stores2)), NO_stores2, NO_stores1),
    #            NO_PFS_stores_ranged = if_else(is.na(if_else(!(is.na(NO_stores1)) & !(is.na(NO_stores2)) & (NO_stores1 < NO_stores2), NO_PFS_stores2, NO_PFS_stores1)) & !(is.na(NO_PFS_stores2)), NO_PFS_stores2, NO_PFS_stores1),
    #            NO_5k_stores_ranged = if_else(is.na(if_else(!(is.na(NO_stores1)) & !(is.na(NO_stores2)) & (NO_stores1 < NO_stores2), NO_5k_stores2, NO_5k_stores1)) & !(is.na(NO_5k_stores2)), NO_5k_stores2, NO_5k_stores1),
    #            NO_20k_stores_ranged = if_else(is.na(if_else(!(is.na(NO_stores1)) & !(is.na(NO_stores2)) & (NO_stores1 < NO_stores2), NO_20k_stores2, NO_20k_stores1)) & !(is.na(NO_20k_stores2)), NO_20k_stores2, NO_20k_stores1),
    #            NO_50k_stores_ranged = if_else(is.na(if_else(!(is.na(NO_stores1)) & !(is.na(NO_stores2)) & (NO_stores1 < NO_stores2), NO_50k_stores2, NO_50k_stores1)) & !(is.na(NO_50k_stores2)), NO_50k_stores2, NO_50k_stores1),
    #            NO_100k_stores_ranged = if_else(is.na(if_else(!(is.na(NO_stores1)) & !(is.na(NO_stores2)) & (NO_stores1 < NO_stores2), NO_100k_stores2, NO_100k_stores1)) & !(is.na(NO_100k_stores2)), NO_100k_stores2, NO_100k_stores1),
    #            NO_100kplus_stores_ranged = if_else(is.na(if_else(!(is.na(NO_stores1)) & !(is.na(NO_stores2)) & (NO_stores1 < NO_stores2), NO_100kplus_stores2, NO_100kplus_stores1)) & !(is.na(NO_100kplus_stores2)), NO_100kplus_stores2, NO_100kplus_stores1)) %>%
    #     select_(.dots = paste0("-", drop_new_columns)) %>%
    #     group_by(Base_Product_Number, ME_EXP, Year_Number) %>%
    #     mutate(NO_stores_ranged = if_else(is.na(NO_stores_ranged), max(NO_stores, na.rm = T), NO_stores_ranged),
    #            NO_PFS_stores_ranged = if_else(is.na(NO_PFS_stores_ranged), max(NO_PFS_Stores, na.rm = T), NO_PFS_stores_ranged),
    #            NO_5k_stores_ranged = if_else(is.na(NO_5k_stores_ranged), max(NO_5K_Stores, na.rm = T), NO_5k_stores_ranged),
    #            NO_20k_stores_ranged = if_else(is.na(NO_20k_stores_ranged), max(NO_20K_Stores, na.rm = T), NO_20k_stores_ranged),
    #            NO_50k_stores_ranged = if_else(is.na(NO_50k_stores_ranged), max(NO_50K_Stores, na.rm = T), NO_50k_stores_ranged),
    #            NO_100k_stores_ranged = if_else(is.na(NO_100k_stores_ranged), max(NO_100K_Stores, na.rm = T), NO_100k_stores_ranged),
    #            NO_100kplus_stores_ranged = if_else(is.na(NO_100kplus_stores_ranged), max(NO_100Kplus_Stores, na.rm = T), NO_100kplus_stores_ranged)) %>%
    #     ungroup() %>%
    #     select_(.dots = paste0("-", drop_old_columns)) %>%
    #     data.frame() %>%
    #     mutate(NO_stores_ranged = (NO_PFS_stores_ranged + NO_5k_stores_ranged + NO_20k_stores_ranged + NO_50k_stores_ranged + NO_100k_stores_ranged + NO_100kplus_stores_ranged))
  }
  
  # Capping the no of stores features to reduce correlation with vol
  {
    ADS_6YR_iter2 = ADS_6YR_iter2 %>%
      group_by(Base_Product_Number, ME_EXP, Year_Quarter) %>%
      select(-starts_with("NO_Range")) %>%
      mutate_at(.cols = vars(starts_with("NO_")),
                .funs = funs(max(.))) %>%
      mutate(NO_stores = NO_PFS_Stores + NO_5K_Stores + NO_20K_Stores + NO_50K_Stores + NO_100K_Stores + NO_100Kplus_Stores) %>%
      rename(NO_Stores_ranged = NO_stores,
             NO_PFS_Stores_ranged = NO_PFS_Stores,
             NO_5K_Stores_ranged = NO_5K_Stores,
             NO_20K_Stores_ranged = NO_20K_Stores,
             NO_50K_Stores_ranged = NO_50K_Stores,
             NO_100K_Stores_ranged = NO_100K_Stores,
             NO_100Kplus_Stores_ranged = NO_100Kplus_Stores) %>%
      ungroup()
  }
  
  # use actual sold stores
  {
    # ADS_6YR_iter2 = ADS_6YR_iter2 %>%
    #   rename(NO_Stores_ranged = NO_stores,
    #          NO_PFS_Stores_ranged = NO_PFS_Stores,
    #          NO_5K_Stores_ranged = NO_5K_Stores,
    #          NO_20K_Stores_ranged = NO_20K_Stores,
    #          NO_50K_Stores_ranged = NO_50K_Stores,
    #          NO_100K_Stores_ranged = NO_100K_Stores,
    #          NO_100Kplus_Stores_ranged = NO_100Kplus_Stores)
  }
  
  # Promotions and Features initialization
  {
    all_promo <- c("num_days_MPP", "num_days_MDZ", "num_days_LSS", "num_days_LPV", "num_days_SPP", "num_days_MSC",
                   "num_days_NIO", "num_days_LSM", "num_days_SLZ", "num_days_NEW", "num_days_MSZ", "num_days_NIP",
                   "num_days_SMZ", "num_days_SOM", "num_days_LSP", "num_days_MSP", "num_days_MPV", "num_days_MSM",
                   "num_days_SMP", "num_days_MSS", "num_days_SOE", "num_days_MSV")
    
    
    features <- c("Brand_Ind", "BUYER", "Package_Type" , "Product_Sub_Group_Code", "NO_Stores_ranged", "NO_PFS_Stores_ranged", "NO_5K_Stores_ranged", "NO_20K_Stores_ranged", "NO_50K_Stores_ranged", "NO_100K_Stores_ranged", "NO_100Kplus_Stores_ranged", "asp",  "price_band_prod_count", "PSG_prod_count",
                  "Quarter_Number", "Period_Number", "Launch_Month", "holidays_sum", "Size", "measure_type", "Year_Number", "Launch_Year_Month", "psg_price_band_prod_count",
                  "no_of_subs_same_brand", "no_of_subs_diff_brand","price_band", "acp", "parent_supplier","ACPASPPERC", "MERCHANDISE_GROUP_CODE", "Till_Roll_Description", "Brand_Name", "JUNIOR_BUYER", "Launch_Week_Number", "Year_Quarter", "Launch_Year_Quarter", "ME_EXP")
    
    # features for exporting ADS alone
    # features <- c("Brand_Ind", "BUYER", "Package_Type" , "Product_Sub_Group_Code", "NO_Stores_ranged", "NO_PFS_Stores_ranged", "NO_5K_Stores_ranged", "NO_20K_Stores_ranged", "NO_50K_Stores_ranged", "NO_100K_Stores_ranged", "NO_100Kplus_Stores_ranged", "asp",  "price_band_prod_count", "PSG_prod_count", "Quarter_Number", "Period_Number", "Launch_Month", "holidays_sum", "Size", "measure_type", "Year_Number", "Launch_Year_Month", "psg_price_band_prod_count", "no_of_subs_same_brand", "no_of_subs_diff_brand","price_band", "acp", "parent_supplier","ACPASPPERC","VARIANTGRPID","BPBVGRPID","GBBGRPID", "MERCHANDISE_GROUP_CODE", "Till_Roll_Description", "Brand_Name", "JUNIOR_BUYER", "Product_Sub_Group_Description", "Long_Description", "MERCHANDISE_GROUP_DESCRIPTION", "Launch_Week_Number", "Year_Quarter", "Launch_Year_Quarter", "ME_EXP")
    
    
    
    for (i in all_promo) {
      if (sum(ADS_6YR_iter2[, i], na.rm = T) != 0) {
        print(i)
        features <- append(features, i)
      }
    }
  }
  
  # Add size features
  ADS_6YR_iter2 = merge(ADS_6YR_iter2, Size, by = 'Base_Product_Number', all.x = T)
  
  # Aggregating Holidays and creating a holiday table as well
  {
    Holidays1 = Holidays %>%
      select(Area_Price_Code, Year_Week_Number, holiday_flag, Year_Number = Holiday_year) %>%
      distinct() %>%
      mutate(week = as.numeric(substr(Year_Week_Number, 5, 6))) %>%
      inner_join(., calendar, by = c("week" = "Week_Number")) %>%
      group_by(Area_Price_Code, Year_Number, Period_Number) %>%
      summarise(holidays_sum = sum(holiday_flag)) %>%
      ungroup() %>%
      mutate(ME_EXP = if_else(Area_Price_Code %in% c(2, 3, 4, 6), "ME", "EXP"),
             Area_Price_Code = NULL) %>%
      group_by(ME_EXP, Year_Number, Period_Number) %>%
      summarise(holidays_sum = max(holidays_sum))
    
    # Holidays2 = Holidays %>%
    #   select(Area_Price_Code, Holiday, Year_Week_Number, Year_Number = Holiday_year)
    #
    # Holidays2 = dummies::dummy.data.frame(data = Holidays2, names = c("Holiday"), sep = "_") %>%
    #   mutate(Week_Number = as.numeric(substr(Year_Week_Number, 5, 6))) %>%
    #   inner_join(., calendar) %>%
    #   dplyr::select(-Year_Week_Number, -Week_Number) %>%
    #   group_by(Area_Price_Code, Year_Number, Period_Number) %>%
    #   summarise_all(funs(max)) %>%
    #   ungroup() %>%
    #   mutate(ME_EXP = if_else(Area_Price_Code %in% c(2, 3, 4, 6), "ME", "EXP"),
    #          Area_Price_Code = NULL) %>%
    #   group_by(ME_EXP, Year_Number, Period_Number) %>%
    #   summarise_all(funs(max))
    
    
    # Get count of holidays in each store type and year month
    ADS_6YR_iter2 = left_join(ADS_6YR_iter2, Holidays1)
    
    # Adding the holidays features as well
    # ADS_6YR_iter2 = full_join(ADS_6YR_iter2, Holidays2) %>%
    #   mutate_at(.cols = vars(starts_with(("Holiday_"))), funs(replace(., is.na(.), 0)))
    
    #holidays_list = ADS_6YR_iter2 %>% select(starts_with("Holiday_")) %>% colnames()
    #features = append(features, holidays_list)
  }
  
  # Treating the other features
  {
    ADS_6YR_iter2$no_of_subs_same_brand[is.na(ADS_6YR_iter2$no_of_subs_same_brand)] <- 0
    ADS_6YR_iter2$no_of_subs_diff_brand[is.na(ADS_6YR_iter2$no_of_subs_diff_brand)] <- 0
    ADS_6YR_iter2$acp[is.na(ADS_6YR_iter2$acp)] <- 0
    ADS_6YR_iter2$parent_supplier[is.na(ADS_6YR_iter2$parent_supplier)] <- "Not Available"
    ADS_6YR_iter2$ACPASPPERC <- if_else(is.infinite((ADS_6YR_iter2$asp - ADS_6YR_iter2$acp)/ADS_6YR_iter2$acp),
                                        0,
                                        if_else(is.na((ADS_6YR_iter2$asp - ADS_6YR_iter2$acp)/ADS_6YR_iter2$acp),
                                                0,
                                                (ADS_6YR_iter2$asp - ADS_6YR_iter2$acp)/ADS_6YR_iter2$acp))
    ADS_6YR_iter2$holidays_sum[is.na(ADS_6YR_iter2$holidays_sum)] = 0
    
    rm(Holidays, Holidays1, calendar)
    
    ADS_6YR_iter2 = ADS_6YR_iter2[complete.cases(ADS_6YR_iter2[, c("Base_Product_Number", features, "vol", "Year_Month")]), c("Base_Product_Number", features, "SI", "vol", "Year_Month")]
  }
  
  # removing all unnecessary files
  rm(SI_file, SI_PSG, Size, all_promo, i, Size_file, ADS_6YR)
  
  cat("\n\n")
  print("ADS - Completed")
  cat("\n\n")
}


# fuzzy matching segment
{
  # the various categorical variables to be cleaned (strict algorithm allowing only one character discrepancy)
  {
    # brands
    {
      all_brands = ADS_6YR_iter2 %>% select(Brand_Name) %>% distinct() %>% arrange(Brand_Name)
      
      y = data.frame(brand = 0, dist = 0)
      
      for (i in 1:nrow(all_brands))
      {
        all_brands_to_match = all_brands
        all_brands_to_match[i, 1] = ""
        x = stringdist::amatch(x = all_brands[i,], table = all_brands_to_match[, 1], maxDist = 1)
        
        y[i, 1] = all_brands[i,]
        y[i, 2] = x
      }
      
      all_brands %<>% mutate(dist = row_number(.))
      
      y = y %>%
        mutate(dist = if_else(!is.na(dist),
                              if_else(dist < row_number(), as.integer(dist), row_number()),
                              as.integer(dist)),
               dist = if_else(is.na(dist), row_number(), as.integer(dist))) %>%
        inner_join(all_brands) %>%
        select(-dist) %>%
        rename_(.dots = setNames(names(.), c("Brand_Name", "New_Brand_Name")))
      
      ADS_6YR_iter2 %<>% left_join(y) %>% select(-Brand_Name) %>% rename(Brand_Name = New_Brand_Name)
    }
    
    # tillroll
    {
      all_tillroll = ADS_6YR_iter2 %>% select(Till_Roll_Description) %>% distinct() %>% arrange(Till_Roll_Description)
      
      y = data.frame(tillroll = 0, dist = 0)
      
      for (i in 1:nrow(all_tillroll))
      {
        all_tillroll_to_match = all_tillroll
        all_tillroll_to_match[i, 1] = ""
        x = stringdist::amatch(x = all_tillroll[i,], table = all_tillroll_to_match[, 1], maxDist = 2)
        
        y[i, 1] = all_tillroll[i,]
        y[i, 2] = x
      }
      
      all_tillroll %<>% mutate(dist = row_number(.))
      
      y = y %>%
        mutate(dist = if_else(!is.na(dist),
                              if_else(dist < row_number(), as.integer(dist), row_number()),
                              as.integer(dist)),
               dist = if_else(is.na(dist), row_number(), as.integer(dist))) %>%
        inner_join(all_tillroll) %>%
        select(-dist) %>%
        rename_(.dots = setNames(names(.), c("Till_Roll_Description", "New_Till_Roll_Description")))
      
      ADS_6YR_iter2 %<>% left_join(y) %>% select(-Till_Roll_Description) %>% rename(Till_Roll_Description = New_Till_Roll_Description)
    }
    
    # remove the extra dfs
    rm(all_brands, all_brands_to_match, all_tillroll, all_tillroll_to_match, y, x)
  }
  
  # use the jaro-winkler results from python
  {
    # tillroll = read.csv('Till Roll Description _ String match.csv', header  =T, as.is = T)
    #
    # brandname = read.csv('Brand Name _ String match.csv', header = T, as.is = T)
  }
  
  # to impute missing parent suppliers
  {
    ## imputing the missing values with a very crude method
    ADS_parent_suppliers1 = ADS_6YR_iter2 %>%
      group_by(Product_Sub_Group_Code, Brand_Name, parent_supplier) %>%
      summarise(sum = sum(vol),
                count = length(unique(Year_Month))) %>%
      group_by(Product_Sub_Group_Code, Brand_Name) %>%
      filter(parent_supplier != "Not Available") %>%
      mutate(rank = dense_rank(-sum)) %>%
      filter(rank == 1) %>%
      select(Product_Sub_Group_Code, Brand_Name, parent_supplier) %>%
      rename(parent_supplier2 = parent_supplier)
    
    ADS_parent_suppliers2 = ADS_6YR_iter2 %>%
      group_by(Product_Sub_Group_Code, parent_supplier) %>%
      summarise(sum = sum(vol),
                count = length(unique(Year_Month))) %>%
      group_by(Product_Sub_Group_Code) %>%
      filter(parent_supplier != "Not Available") %>%
      mutate(rank = dense_rank(-sum)) %>%
      filter(rank == 1) %>%
      select(Product_Sub_Group_Code, parent_supplier) %>%
      rename(parent_supplier3 = parent_supplier)
    
    ADS_6YR_iter2 %<>%
      left_join(ADS_parent_suppliers1) %>%
      left_join(ADS_parent_suppliers2) %>%
      mutate(parent_supplier = if_else(parent_supplier == "Not Available", parent_supplier2, parent_supplier),
             parent_supplier = if_else(parent_supplier == "Not Available", parent_supplier3, parent_supplier)) %>%
      select(-parent_supplier2, -parent_supplier3)
    
    rm(ADS_parent_suppliers1, ADS_parent_suppliers2)
  }
  
  cat("\n\n")
  print("fuzzy wuzzy - Completed")
  cat("\n\n")
}


# Conversion to ME/EXP level (commented since it wont be required again. Kept for QC purposes only)
{
  #
  # ADS_6YR_iter2 %<>%
  #   # removing columns not needed
  #   select(-MAXWNINDEX, -LTWINDEX) %>%
  #   mutate(ME_EXP = if_else(Area_Price_Code %in% c(2, 3, 4, 6), "ME", "EXP")) %>%
  #   select(-Area_Price_Code)
  #
  # ADS_1 = ADS_6YR_iter2 %>%
  #   group_by(Base_Product_Number, ME_EXP, Year_Month) %>%
  #   summarise_at(.cols = vars(Brand_Grp20, Brand_Grp10, Brand_Ind, BUYER, Package_Type, Product_Sub_Group_Code, measure_type, Quarter_Number, Launch_Month, Size, Year_Number, Launch_Year_Month, parent_supplier, VARIANTGRPID, BPBVGRPID, GBBGRPID, MERCHANDISE_GROUP_CODE, Till_Roll_Description, Brand_Name, Launch_Year_Quarter, Year_Month, Launch_Year_Quarter, Year_Quarter, Months_Since_Launch, Period_Number),       .funs = c("unique")) %>%
  #   ungroup()
  #
  # ADS_2 = ADS_6YR_iter2 %>%
  #   group_by(Base_Product_Number, ME_EXP, Year_Month) %>%
  #   summarise_at(.cols = vars(matches("no_of_subs|Holiday_"), holidays_sum, psg_price_band_prod_count),
  #                .funs = c("max")) %>%
  #   ungroup()
  #
  # ADS_3 = ADS_6YR_iter2 %>%
  #   group_by(Base_Product_Number, ME_EXP, Year_Month) %>%
  #   summarise_at(.cols = vars(asp, acp, ACPASPPERC, SI,  starts_with("num_days"), price_band_prod_count, PSG_prod_count),
  #                .funs = c("mean")) %>%
  #   mutate_all(.funs = "round(., digits = 2)") %>%
  #   ungroup()
  #
  # ADS_4 = ADS_6YR_iter2 %>%
  #   group_by(Base_Product_Number, ME_EXP, Year_Month) %>%
  #   summarise_at(.cols = vars(matches("NO_", ignore.case = F), vol),
  #                .funs = c("sum")) %>%
  #   ungroup()
  #
  # Mode <- function(x) {
  #   ux <- unique(x)
  #   ux[which.max(tabulate(match(x, ux)))]
  # }
  # ADS_price_band = ADS_6YR_iter2 %>%
  #   group_by(Base_Product_Number, ME_EXP, Year_Month) %>%
  #   summarise(price_band = Mode(price_band)) %>%
  #   ungroup()
  #
  # ADS_6YR_iter2 = bind_cols(ADS_1, ADS_2, ADS_3, ADS_4, ADS_price_band)
  # ADS_6YR_iter2 = ADS_6YR_iter2[, !duplicated(colnames(ADS_6YR_iter2))] %>% data.frame()
  #
  # rm(ADS_1, ADS_2, ADS_3, ADS_4, ADS_price_band)
}


# Convertion to Quarterly level (commented for now)
{
  #   convert_single_to_double = function(x) {
  #     if_else(nchar(x) < 2,
  #             paste0("0", x),
  #             as.character(x))
  #   }
  #
  #   # one long dplyr pipe, find comments for each segment in it
  #   ADS_6YR_iter2 %<>%
  #     # removing columns not needed at quarter level
  #     select(-Weeks_Since_Launch, -Period_Number, -MAXWNINDEX, -LTWINDEX, -Long_Description, -Year_Month) %>%
  #     mutate(Year_Quarter = as.numeric(paste0(Year_Number, convert_single_to_double(Quarter_Number))),
  #            Launch_Year_Quarter = as.numeric(paste0(substr(as.character(Launch_Year_Month), 1, 4),
  #                                                    convert_single_to_double(as.character(ceiling(as.numeric(Launch_Month) / 3)))))) %>% data.frame()
  #
  #   ADS_quarter1 = ADS_6YR_iter2 %>%
  #     group_by(Base_Product_Number, Area_Price_Code, Year_Quarter) %>%
  #     summarise_at(.cols = vars(Brand_Grp20, Brand_Grp10, Brand_Ind, BUYER, Package_Type, Product_Sub_Group_Code, measure_type, Quarter_Number, Launch_Month, Size, Year_Number, Launch_Year_Month, parent_supplier, VARIANTGRPID, BPBVGRPID, GBBGRPID, MERCHANDISE_GROUP_CODE, Till_Roll_Description, Brand_Name, Launch_Year_Quarter),       .funs = c("unique")) %>%
  #     ungroup()
  #
  #   ADS_quarter2 = ADS_6YR_iter2 %>%
  #     group_by(Base_Product_Number, Area_Price_Code, Year_Quarter) %>%
  #     summarise_at(.cols = vars(matches("NO_|Holiday_"), price_band_prod_count, PSG_prod_count, psg_price_band_prod_count),
  #                  .funs = c("max")) %>%
  #     ungroup()
  #
  #   ADS_quarter3 = ADS_6YR_iter2 %>%
  #     group_by(Base_Product_Number, Area_Price_Code, Year_Quarter) %>%
  #     summarise_at(.cols = vars(asp, acp, ACPASPPERC, SI),
  #                  .funs = c("mean")) %>%
  #     ungroup()
  #
  #   ADS_quarter4 = ADS_6YR_iter2 %>%
  #     group_by(Base_Product_Number, Area_Price_Code, Year_Quarter) %>%
  #     summarise_at(.cols = vars(holidays_sum, starts_with("num_days"), vol),
  #                  .funs = c("sum")) %>%
  #     ungroup()
  #
  #   Mode <- function(x) {
  #     ux <- unique(x)
  #     ux[which.max(tabulate(match(x, ux)))]
  #   }
  #   ADS_price_band = ADS_6YR_iter2 %>%
  #     group_by(Base_Product_Number, Area_Price_Code, Year_Quarter) %>%
  #     summarise(price_band = Mode(price_band)) %>%
  #     ungroup()
  #
  #   ADS_6YR_iter2 = bind_cols(ADS_quarter1, ADS_quarter2, ADS_quarter3, ADS_quarter4, ADS_price_band)
  #   ADS_6YR_iter2 = ADS_6YR_iter2[, !duplicated(colnames(ADS_6YR_iter2))] %>% data.frame()
  #
  #   rm(ADS_quarter1, ADS_quarter2, ADS_quarter3, ADS_quarter4, ADS_price_band)
  #
  #   # defining new time start at a quarter level
  #   Time_start = 201101
  #   Time_end = 201504
}


# Additional elements ####
{
  # Getting the Till Roll features
  {
    # source('Till_Roll_wordcloud.R')
    # ADS_6YR_iter2_inter = left_join(ADS_6YR_iter2, new_till)
    # rm(new_till)
  }
  
  # Getting the similar product last year sales
  {
    # sim_prod_last_year_vol_df = read.table(file = "BPN_15SUB_VOL_MON.txt", sep = ";", header = T, as.is = T, quote = "", na.strings = "?") %>%
    #   select(-BASE_PROD_VOL, -BASE_PROD_SALES, -Base_prod_launch_wk) %>%
    #   rename(ME_EXP = STORE_TYPE)
    #
    # ADS_6YR_iter2_inter = left_join(ADS_6YR_iter2_inter, sim_prod_last_year_vol_df) %>%
    #   filter(is.na(SUB_PROD_VOL) == FALSE)
    # rm(sim_prod_last_year_vol_df)
  }
  
  # Getting the volume features at multiple levels
  {
    # Creating the other volume features (PSG, MerchGrp, BUYER, JUNIOR BUYER, parent supplier, Brand Name, Brand ind level, price band, package type, measure type, till roll)
    {
      ADS_6YR_iter2_vol_features = ADS_6YR_iter2 %>%
        select(Base_Product_Number,
               Product_Sub_Group_Code, MERCHANDISE_GROUP_CODE, BUYER,
               JUNIOR_BUYER, parent_supplier, Brand_Ind, Brand_Name,
               price_band, measure_type, Package_Type, Till_Roll_Description,
               ME_EXP,
               vol,
               Year_Month, Year_Quarter, Year_Number) %>%
        
        # the PSG features
        group_by(Product_Sub_Group_Code, ME_EXP, Year_Month) %>%
        mutate(PSG_month_vol_sum = sum(vol),
               PSG_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Product_Sub_Group_Code, ME_EXP, Year_Quarter) %>%
        mutate(PSG_quarter_vol_sum = sum(vol),
               PSG_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Product_Sub_Group_Code, ME_EXP, Year_Number) %>%
        mutate(PSG_year_vol_sum = sum(vol),
               PSG_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the MerchGrp features
        group_by(MERCHANDISE_GROUP_CODE, ME_EXP, Year_Month) %>%
        mutate(MerchGrp_month_vol_sum = sum(vol),
               MerchGrp_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(MERCHANDISE_GROUP_CODE, ME_EXP, Year_Quarter) %>%
        mutate(MerchGrp_quarter_vol_sum = sum(vol),
               MerchGrp_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(MERCHANDISE_GROUP_CODE, ME_EXP, Year_Number) %>%
        mutate(MerchGrp_year_vol_sum = sum(vol),
               MerchGrp_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the BUYER features
        group_by(BUYER, ME_EXP, Year_Month) %>%
        mutate(BUYER_month_vol_sum = sum(vol),
               BUYER_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(BUYER, ME_EXP, Year_Quarter) %>%
        mutate(BUYER_quarter_vol_sum = sum(vol),
               BUYER_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(BUYER, ME_EXP, Year_Number) %>%
        mutate(BUYER_year_vol_sum = sum(vol),
               BUYER_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the JUNIOR BUYER features
        group_by(JUNIOR_BUYER, ME_EXP, Year_Month) %>%
        mutate(JUNIOR_BUYER_month_vol_sum = sum(vol),
               JUNIOR_BUYER_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(JUNIOR_BUYER, ME_EXP, Year_Quarter) %>%
        mutate(JUNIOR_BUYER_quarter_vol_sum = sum(vol),
               JUNIOR_BUYER_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(JUNIOR_BUYER, ME_EXP, Year_Number) %>%
        mutate(JUNIOR_BUYER_year_vol_sum = sum(vol),
               JUNIOR_BUYER_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the parent supplier features
        group_by(parent_supplier, ME_EXP, Year_Month) %>%
        mutate(parent_supplier_month_vol_sum = sum(vol),
               parent_supplier_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(parent_supplier, ME_EXP, Year_Quarter) %>%
        mutate(parent_supplier_quarter_vol_sum = sum(vol),
               parent_supplier_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(parent_supplier, ME_EXP, Year_Number) %>%
        mutate(parent_supplier_year_vol_sum = sum(vol),
               parent_supplier_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the Brand ind features
        group_by(Brand_Ind, ME_EXP, Year_Month) %>%
        mutate(Brand_Ind_month_vol_sum = sum(vol),
               Brand_Ind_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Brand_Ind, ME_EXP, Year_Quarter) %>%
        mutate(Brand_Ind_quarter_vol_sum = sum(vol),
               Brand_Ind_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Brand_Ind, ME_EXP, Year_Number) %>%
        mutate(Brand_Ind_year_vol_sum = sum(vol),
               Brand_Ind_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the Brand name features
        group_by(Brand_Name, ME_EXP, Year_Month) %>%
        mutate(Brand_Name_month_vol_sum = sum(vol),
               Brand_Name_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Brand_Name, ME_EXP, Year_Quarter) %>%
        mutate(Brand_Name_quarter_vol_sum = sum(vol),
               Brand_Name_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Brand_Name, ME_EXP, Year_Number) %>%
        mutate(Brand_Name_year_vol_sum = sum(vol),
               Brand_Name_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the price band features
        group_by(price_band, ME_EXP, Year_Month) %>%
        mutate(price_band_month_vol_sum = sum(vol),
               price_band_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(price_band, ME_EXP, Year_Quarter) %>%
        mutate(price_band_quarter_vol_sum = sum(vol),
               price_band_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(price_band, ME_EXP, Year_Number) %>%
        mutate(price_band_year_vol_sum = sum(vol),
               price_band_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the package type features
        group_by(Package_Type, ME_EXP, Year_Month) %>%
        mutate(Package_Type_month_vol_sum = sum(vol),
               Package_Type_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Package_Type, ME_EXP, Year_Quarter) %>%
        mutate(Package_Type_quarter_vol_sum = sum(vol),
               Package_Type_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Package_Type, ME_EXP, Year_Number) %>%
        mutate(Package_Type_year_vol_sum = sum(vol),
               Package_Type_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the measure type features
        group_by(measure_type, ME_EXP, Year_Month) %>%
        mutate(measure_type_month_vol_sum = sum(vol),
               measure_type_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(measure_type, ME_EXP, Year_Quarter) %>%
        mutate(measure_type_quarter_vol_sum = sum(vol),
               measure_type_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(measure_type, ME_EXP, Year_Number) %>%
        mutate(measure_type_year_vol_sum = sum(vol),
               measure_type_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        # the till roll features
        group_by(Till_Roll_Description, ME_EXP, Year_Month) %>%
        mutate(Till_Roll_Description_month_vol_sum = sum(vol),
               Till_Roll_Description_month_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Till_Roll_Description, ME_EXP, Year_Quarter) %>%
        mutate(Till_Roll_Description_quarter_vol_sum = sum(vol),
               Till_Roll_Description_quarter_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        group_by(Till_Roll_Description, ME_EXP, Year_Number) %>%
        mutate(Till_Roll_Description_year_vol_sum = sum(vol),
               Till_Roll_Description_year_vol_avg = sum(vol)/length(unique(Base_Product_Number))) %>%
        ungroup() %>%
        
        mutate(Year_Month = Year_Month + 100,
               Year_Quarter = Year_Quarter + 100,
               Year_Number = Year_Number + 1) %>%
        select(-vol, -Base_Product_Number) %>%
        distinct()
    }
    
    # Creating the individual sets for joining with the main ADS
    {
      ADS_6YR_iter2_vol_features_PSG = ADS_6YR_iter2_vol_features %>%
        select(Product_Sub_Group_Code, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("PSG_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_MerchGrp = ADS_6YR_iter2_vol_features %>%
        select(MERCHANDISE_GROUP_CODE, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("MerchGrp_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_BUYER = ADS_6YR_iter2_vol_features %>%
        select(BUYER, ME_EXP, Year_Month, Year_Quarter, Year_Number, starts_with("BUYER_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_JUNIOR_BUYER = ADS_6YR_iter2_vol_features %>%
        select(JUNIOR_BUYER, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("JUNIOR_BUYER_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_parent_supplier = ADS_6YR_iter2_vol_features %>%
        select(parent_supplier, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("parent_supplier_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_Brand_Ind = ADS_6YR_iter2_vol_features %>%
        select(Brand_Ind, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("Brand_Ind_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_Brand_Name = ADS_6YR_iter2_vol_features %>%
        select(Brand_Name, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("Brand_Name_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_price_band = ADS_6YR_iter2_vol_features %>%
        select(price_band, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("price_band_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_Package_Type = ADS_6YR_iter2_vol_features %>%
        select(Package_Type, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("Package_Type_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_measure_type = ADS_6YR_iter2_vol_features %>%
        select(measure_type, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("measure_type_")) %>%
        distinct()
      
      ADS_6YR_iter2_vol_features_Till_Roll_Description = ADS_6YR_iter2_vol_features %>%
        select(Till_Roll_Description, ME_EXP, Year_Month, Year_Quarter, Year_Number, matches("Till_Roll_Description_")) %>%
        distinct()
    }
    
    base_columns = colnames(ADS_6YR_iter2)
    
    # snippet to join all of them finally
    {
      # store all the df names in a vector
      vol_feature_dfs = apropos("ADS_6YR_iter2_vol_features_")
      
      for (i in 1:length(vol_feature_dfs)) {
        ADS_6YR_iter2 = left_join(ADS_6YR_iter2, get(vol_feature_dfs[i])) %>% distinct()
      }
    }
    
    # removing all extra dataframes
    rm(list = apropos("ADS_6YR_iter2_vol_features"))
    
    # final treatment of nulls in final dataset (converting to zero and then removing 2012 from set)
    ADS_6YR_iter2[is.na(ADS_6YR_iter2)] = 0
    ADS_6YR_iter2 = ADS_6YR_iter2 %>% filter(Year_Number > 2012)
    
    # Removing redundant features
    {
      # ensure the results are repeatable
      set.seed(121)
      
      # remove all the categorical variables
      ADS_cor = ADS_6YR_iter2
      
      categ_variables = c("acp", "asp", "vol", "Year_Month", "Year_Number", "Year_Quarter", "Launch_Year_Quarter", "Launch_Month", "Launch_Year_Month", "Launch_Week_Number", "Period_Number", "QUarter_Number")
      for (i in 1:ncol(ADS_cor)) {
        if ((class(ADS_cor[, i]) == "character")) {
          categ_variables = append(categ_variables, colnames(ADS_cor[, i, drop = F]))
        }
      }
      
      ADS_cor = ADS_cor[, !colnames(ADS_cor) %in% categ_variables]
      
      # calculate correlation matrix
      correlationMatrix = cor(ADS_cor)
      correlationMatrix[is.na(correlationMatrix)] <- 0
      
      # find attributes that are highly corrected (ideally > 0.9, change if necessary)
      highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.9)
      
      # print indexes of highly correlated attributes
      print("Correlated features")
      print(colnames(ADS_cor[ , highlyCorrelated, drop = F]))
      
      highlyCorrelated = colnames(ADS_cor[ , highlyCorrelated, drop = F])
      
      # subset dataset without these features
      ADS_6YR_iter2 = ADS_6YR_iter2[, !colnames(ADS_6YR_iter2) %in% highlyCorrelated]
      
      rm(ADS_cor, correlationMatrix, categ_variables, highlyCorrelated, i, base_columns, vol_feature_dfs, features)
    }
  }
  
  cat("\n\n")
  print("Additional elements - Completed")
  cat("\n\n")
}


# Getting the bucket features for important categorical variables ####
{
  train = ADS_6YR_iter2 %>%
    filter(Year_Number < 2016 & Launch_Year_Month < 201601)
  
  test = ADS_6YR_iter2 %>%
    filter(Launch_Year_Month >= 201601)
  
  # creating the bucket features for the various categorical variables
  {
    bucket_numbers1 = ceiling(length(unique(train$Brand_Name)) /
                                floor(sqrt(length(unique(train$Brand_Name)))))
    bucket_numbers2 = ceiling(length(unique(train$Till_Roll_Description)) /
                                floor(sqrt(length(unique(train$Till_Roll_Description)))))
    bucket_numbers3 = ceiling(length(unique(train$parent_supplier)) /
                                floor(sqrt(length(unique(train$parent_supplier)))))
    bucket_numbers4 = ceiling(length(unique(train$Product_Sub_Group_Code)) /
                                floor(sqrt(length(unique(train$Product_Sub_Group_Code)))))
    bucket_numbers5 = ceiling(length(unique(train$JUNIOR_BUYER)) / floor(sqrt(length(unique(train$JUNIOR_BUYER)))))
    bucket_numbers6 = ceiling(length(unique(train$Package_Type)) / floor(sqrt(length(unique(train$Package_Type)))))
    bucket_numbers7 = ceiling(length(unique(train$Size)) / floor(sqrt(length(unique(train$Size)))))
    bucket_numbers8 = ceiling(length(unique(train$price_band)) / floor(sqrt(length(unique(train$price_band)))))
    
    brands = train %>%
      group_by(Brand_Name, ME_EXP) %>%
      summarise(sum_vol = sum(vol),
                avg_vol_sku_ym = sum(vol) / (length(unique(Base_Product_Number)) * length(unique(Year_Month)))) %>%
      group_by(ME_EXP) %>%
      mutate(brnd_buck1 = ntile(sum_vol, n = bucket_numbers1),
             brnd_buck2 = ntile(avg_vol_sku_ym, n = bucket_numbers1)) %>%
      select(Brand_Name, brnd_buck1, brnd_buck2, ME_EXP)
    
    tillrolldescs = train %>%
      group_by(Till_Roll_Description, ME_EXP) %>%
      summarise(sum_vol = sum(vol),
                avg_vol_sku_ym = sum(vol) / (length(unique(Base_Product_Number)) * length(unique(Year_Month)))) %>%
      group_by(ME_EXP) %>%
      mutate(trd_buck1 = ntile(sum_vol, n = bucket_numbers2),
             trd_buck2 = ntile(avg_vol_sku_ym, n = bucket_numbers2)) %>%
      select(Till_Roll_Description, trd_buck1, trd_buck2, ME_EXP)
    
    suppliers = train %>%
      group_by(parent_supplier, ME_EXP) %>%
      summarise(sum_vol = sum(vol),
                avg_vol_sku_ym = sum(vol) / (length(unique(Base_Product_Number)) * length(unique(Year_Month)))) %>%
      group_by(ME_EXP) %>%
      mutate(supplier_buck1 = ntile(sum_vol, n = bucket_numbers3),
             supplier_buck2 = ntile(avg_vol_sku_ym, n = bucket_numbers3)) %>%
      select(parent_supplier, supplier_buck1, supplier_buck2, ME_EXP)
    
    psgs = train %>%
      group_by(Product_Sub_Group_Code, ME_EXP) %>%
      summarise(sum_vol = sum(vol),
                avg_vol_sku_ym = sum(vol) / (length(unique(Base_Product_Number)) * length(unique(Year_Month)))) %>%
      group_by(ME_EXP) %>%
      mutate(psg_buck1 = ntile(sum_vol, n = bucket_numbers4),
             psg_buck2 = ntile(avg_vol_sku_ym, n = bucket_numbers4)) %>%
      select(Product_Sub_Group_Code, psg_buck1, psg_buck2, ME_EXP)
    
    juniorbuyer = train %>%
      group_by(JUNIOR_BUYER, ME_EXP) %>%
      summarise(sum_vol = sum(vol),
                avg_vol_sku_ym = sum(vol) / (length(unique(Base_Product_Number)) * length(unique(Year_Month)))) %>%
      group_by(ME_EXP) %>%
      mutate(jbuyer_buck1 = ntile(sum_vol, n = bucket_numbers5),
             jbuyer_buck2 = ntile(avg_vol_sku_ym, n = bucket_numbers5)) %>%
      select(JUNIOR_BUYER, jbuyer_buck1, jbuyer_buck2, ME_EXP)
    
    packagetype = train %>%
      group_by(Package_Type, ME_EXP) %>%
      summarise(sum_vol = sum(vol),
                avg_vol_sku_ym = sum(vol) / (length(unique(Base_Product_Number)) * length(unique(Year_Month)))) %>%
      group_by(ME_EXP) %>%
      mutate(pkgtype_buck1 = ntile(sum_vol, n = bucket_numbers6),
             pkgtype_buck2 = ntile(avg_vol_sku_ym, n = bucket_numbers6)) %>%
      select(Package_Type, pkgtype_buck1, pkgtype_buck2, ME_EXP)
    
    sizes = train %>%
      group_by(Size, ME_EXP) %>%
      summarise(sum_vol = sum(vol),
                avg_vol_sku_ym = sum(vol) / (length(unique(Base_Product_Number)) * length(unique(Year_Month)))) %>%
      group_by(ME_EXP) %>%
      mutate(size_buck1 = ntile(sum_vol, n = bucket_numbers7),
             size_buck2 = ntile(avg_vol_sku_ym, n = bucket_numbers7)) %>%
      select(Size, size_buck1, size_buck2, ME_EXP)
    
    priceband = train %>%
      group_by(price_band, ME_EXP) %>%
      summarise(sum_vol = sum(vol),
                avg_vol_sku_ym = sum(vol) / (length(unique(Base_Product_Number)) * length(unique(Year_Month)))) %>%
      group_by(ME_EXP) %>%
      mutate(priceband_buck1 = ntile(sum_vol, n = bucket_numbers8),
             priceband_buck2 = ntile(avg_vol_sku_ym, n = bucket_numbers8)) %>%
      select(price_band, priceband_buck1, priceband_buck2, ME_EXP)
    }
  
  base_columns = colnames(train)
  
  # combine all of them with the train set
  train %<>%
    left_join(brands) %>%
    left_join(tillrolldescs) %>%
    left_join(suppliers) %>%
    left_join(psgs) %>%
    left_join(juniorbuyer) %>%
    left_join(packagetype) %>%
    left_join(sizes) %>%
    left_join(priceband)
  
  # snippet to run an xgboost model to select only required features
  {
    # creating a xgboost model with these features alone
    train_buckets = train %>%
      select(contains("_buck"), vol)
    
    dim_cols = c("vol")
    X_cols = !colnames(train_buckets) %in% dim_cols
    gc()
    train_X = as.matrix(sapply(train_buckets[,X_cols], as.numeric))
    train_Y = as.matrix(as.numeric(train_buckets[,'vol']))
    
    XGB_Model = xgboost(data = train_X, label = train_Y, nrounds = 100, verbose = 1, booster = "gbtree", eta = 0.1, max_depth = 7, objective = "reg:linear", print_every_n = 20)
    
    # get the important features alone
    xgb_importance = data.table(xgboost::xgb.importance(feature_names = colnames(train_X), model = XGB_Model))
    
    Importance_table = data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
      mutate(Rank = dense_rank(desc(Importance))) %>%
      filter(Rank <= 10)
    colnames_features_additional_features = as.vector(Importance_table$Feature)
    
    base_columns = append(base_columns, colnames_features_additional_features) %>% unique()
    
    train = train[, base_columns]
    rm(Importance_table, train_X, train_Y, xgb_importance, XGB_Model, train_buckets)
    rm(colnames_features_additional_features, dim_cols, X_cols)
    }
  
  # snippet to use jaro winkler distance algo to map new test values to those in train
  {
    # custom function to calculate mode of a group
    Mode <- function(x) {
      ux <- unique(x)
      ux[which.max(tabulate(match(x, ux)))]
    }
    
    # for brands ####
    train_brands = train %>% select(Brand_Name) %>% distinct()
    test_brands = test %>% select(Brand_Name) %>% distinct() %>% setdiff(train_brands)
    
    all_brands_grid = expand.grid(brand1 = test_brands$Brand_Name, brand2 = train_brands$Brand_Name, stringsAsFactors = F) %>%
      mutate(jw_score = 1 - stringdist(a = brand1, b = brand2, method = "jw")) %>%
      filter(jw_score != 1 & jw_score > 0.7) %>%
      select(-jw_score) %>%
      rename(test_Brand_Name = brand1, Brand_Name = brand2)
    
    new_brands = left_join(brands, all_brands_grid) %>%
      mutate(Brand_Name = if_else(is.na(test_Brand_Name), Brand_Name, test_Brand_Name)) %>%
      group_by(Brand_Name, ME_EXP) %>%
      summarise(brnd_buck1 = median(brnd_buck1),
                brnd_buck2 = median(brnd_buck2)) %>%
      rbind(brands) %>%
      ungroup() %>%
      distinct()
    
    # for tillrolls ####
    train_tillroll = train %>% select(Till_Roll_Description) %>% distinct()
    test_tillroll = test %>% select(Till_Roll_Description) %>% distinct() %>% setdiff(train_tillroll)
    
    all_tillroll_grid = expand.grid(tillroll1 = test_tillroll$Till_Roll_Description, tillroll2 = train_tillroll$Till_Roll_Description, stringsAsFactors = F) %>%
      mutate(jw_score = 1 - stringdist(a = tillroll1, b = tillroll2, method = "jw")) %>%
      filter(jw_score != 1 & jw_score > 0.6) %>%
      select(-jw_score) %>%
      rename(test_Till_Roll_Description = tillroll1, Till_Roll_Description = tillroll2)
    
    new_tillrolldescs = left_join(tillrolldescs, all_tillroll_grid) %>%
      mutate(Till_Roll_Description = if_else(is.na(test_Till_Roll_Description), Till_Roll_Description, test_Till_Roll_Description)) %>%
      group_by(Till_Roll_Description, ME_EXP) %>%
      summarise(trd_buck1 = median(trd_buck1),
                trd_buck2 = median(trd_buck2)) %>%
      rbind(tillrolldescs) %>%
      ungroup() %>%
      distinct()
  }
  
  # mapping the values computed in train to the test observations
  {
    # combine all of them with the test set
    test %<>%
      left_join(new_brands) %>%
      left_join(new_tillrolldescs) %>%
      left_join(suppliers) %>%
      left_join(psgs) %>%
      left_join(juniorbuyer) %>%
      left_join(packagetype) %>%
      left_join(priceband) %>%
      left_join(sizes)
    
    test = test[, base_columns]
    
    #colSums(is.na(test))
    
    # snippet to convert all NAs to median of the column (except till roll)
    {
      test %<>% mutate(index = row_number(Base_Product_Number))
      
      test_nas_to_median_ME = test %>%
        filter(ME_EXP == "ME") %>%
        select(contains("buck"), index) %>%
        as.matrix()
      
      test_nas_to_median_EXP = test %>%
        filter(ME_EXP == "EXP") %>%
        select(contains("buck"), index) %>%
        as.matrix()
      
      indx_ME = which(is.na(test_nas_to_median_ME), arr.ind = TRUE)
      indx_EXP = which(is.na(test_nas_to_median_EXP), arr.ind = TRUE)
      
      test_nas_to_median_ME[indx_ME] = matrixStats::colMedians(test_nas_to_median_ME, na.rm = TRUE)[indx_ME[, 2]]
      test_nas_to_median_EXP[indx_EXP] = matrixStats::colMedians(test_nas_to_median_EXP, na.rm = TRUE)[indx_EXP[, 2]]
      
      test_nas_to_median = rbind(test_nas_to_median_EXP, test_nas_to_median_ME) %>% data.frame()
      
      test %<>% select(-contains("buck")) %>% left_join(test_nas_to_median) %>% select(-index)
    }
  }
  
  # remove all the unnecessary objects
  rm(brands, juniorbuyer, psgs, packagetype, sizes, suppliers, tillrolldescs, priceband, bucket_numbers1, bucket_numbers2, bucket_numbers3, bucket_numbers4, bucket_numbers5, bucket_numbers6, bucket_numbers7, bucket_numbers8)
  rm(new_brands, new_tillrolldescs, all_brands_grid, all_tillroll_grid, test_brands, test_tillroll, train_brands, train_tillroll, test_nas_to_median, test_nas_to_median_EXP, test_nas_to_median_ME, indx_EXP, indx_ME)
}


# Some more feature engineering (open and use with care) ####
{
  # function to compute the deviation encoded features
  categtoDeviationenc = function(char_data,
                                 num_data,
                                 funcs = funs(mean(., na.rm=T),
                                              sd(., na.rm=T),
                                              'median' = median(., na.rm=T))) {
    
    train_char_data = char_data %>% data.frame()
    train_num_data = num_data %>% data.frame()
    
    res = list()
    
    for(x in names(train_char_data)){
      res[[x]] = train_num_data %>% group_by(.dots=train_char_data[,x]) %>% summarise_each(funcs) # calculate mean/sd/median encodings
      res[[x]][,-1] = apply(res[[x]][,-1], 2, scale, scale = FALSE, center = TRUE) # calculate deviances of mean/sd/median encodings
      # rename columns
      colnames(res[[x]])[1] = x
      if (ncol(train_num_data) == 1)
        colnames(res[[x]])[-1] = paste0(names(train_num_data),'_',names(res[[x]])[-1])
      res[[x]] <- merge(char_data[, x, drop = F], res[[x]], all.x = T, by = x)[,-1] # apply encodings to all data
    }
    res = data.frame(do.call(cbind, res))
    return(res)
  }
  
  # converting whatever character columns we can into factors
  for (i in 1:ncol(train)) {
    if ((class(train[, i]) == "character")) {
      train[, i] = as.factor(train[, i])
    }
  }
  
  # the categorical features to make deviation features of
  categ_variables = c("Brand_Name",
                      "Till_Roll_Description",
                      "parent_supplier",
                      "Product_Sub_Group_Code",
                      "BUYER",
                      "Package_Type",
                      "JUNIOR_BUYER",
                      "Size",
                      "MERCHANDISE_GROUP_CODE")
  
  base_columns = colnames(train)
  
  # snippet to add the features to the train dataset
  {
    # selecting and subsetting the train set with the categorical features to make the deviation features of
    train_ME = train %>% filter(ME_EXP == "ME")
    
    train_dev_ME = train %>%
      filter(ME_EXP == "ME") %>%
      select(vol,
             Brand_Name,
             Till_Roll_Description,
             parent_supplier,
             Product_Sub_Group_Code,
             BUYER,
             Package_Type,
             JUNIOR_BUYER,
             Size,
             MERCHANDISE_GROUP_CODE)
    
    
    train_EXP = train %>% filter(ME_EXP == "EXP")
    
    train_dev_EXP = train %>%
      filter(ME_EXP == "EXP") %>%
      select(vol,
             Brand_Name,
             Till_Roll_Description,
             parent_supplier,
             Product_Sub_Group_Code,
             BUYER,
             Package_Type,
             JUNIOR_BUYER,
             Size,
             MERCHANDISE_GROUP_CODE)
    
    ADS_dev_features_ME = list()
    ADS_dev_features_EXP = list()
    
    # adding it to the ME train set
    for (i in 1:length(categ_variables))
    {
      # creating the deviation features
      char_data = data.frame(train_dev_ME[,c(i+1)])
      names(char_data) = categ_variables[i]
      
      ADS_dev_features_ME[[i]] = categtoDeviationenc(char_data = char_data, num_data = train_dev_ME[,1,drop=F])
      
      char_data = char_data %>% arrange(.[[1]])
      
      ADS_dev_features_ME[[i]] = cbind(char_data, ADS_dev_features_ME[[i]]) %>% distinct()
      
      # adding it to the train dataset and assigning it to train
      train_ME = left_join(train_ME, ADS_dev_features_ME[[i]])
    }
    
    # adding it to the EXP train set
    for (i in 1:length(categ_variables))
    {
      # creating the deviation features
      char_data = data.frame(train_dev_EXP[,c(i+1)])
      names(char_data) = categ_variables[i]
      
      ADS_dev_features_EXP[[i]] = categtoDeviationenc(char_data = char_data, num_data = train_dev_EXP[,1,drop=F])
      
      char_data = char_data %>% arrange(.[[1]])
      
      ADS_dev_features_EXP[[i]] = cbind(char_data, ADS_dev_features_EXP[[i]]) %>% distinct()
      
      # adding it to the train dataset and assigning it to train
      train_EXP = left_join(train_EXP, ADS_dev_features_EXP[[i]])
    }
  }
  
  train = rbind(train_EXP, train_ME)
  rm(train_dev_EXP, train_dev_ME)
  
  # snippet to run an xgboost model to select only required features
  {
    # creating a xgboost model with these features alone
    train_dev_imp = train %>%
      select(contains(".vol_"), vol)
    
    dim_cols = c("vol")
    X_cols = !colnames(train_dev_imp) %in% dim_cols
    gc()
    train_X = as.matrix(sapply(train_dev_imp[,X_cols], as.numeric))
    train_Y = as.matrix(as.numeric(train_dev_imp[,'vol']))
    
    XGB_Model = xgboost(data = train_X, label = train_Y, nrounds = 100, verbose = 1, booster = "gbtree", eta = 0.01, max_depth = 7, objective = "reg:linear", print_every_n = 20)
    
    # get the important features alone
    xgb_importance = data.table(xgboost::xgb.importance(feature_names = colnames(train_X), model = XGB_Model))
    
    Importance_table = data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
      mutate(Rank = dense_rank(desc(Importance))) %>%
      filter(Rank <= 10)
    colnames_features_additional_features = as.vector(Importance_table$Feature)
    
    base_columns = append(base_columns, colnames_features_additional_features) %>% unique()
    
    train = train[, base_columns]
    rm(Importance_table, train_X, train_Y, xgb_importance, XGB_Model, train_dev_imp)
    rm(colnames_features_additional_features, dim_cols, X_cols)
  }
  
  # snippet to map new test values to its closest train values
  {
    # for brands ####
    train_brands_ME = train %>% filter(ME_EXP == "ME") %>% select(Brand_Name) %>% distinct()
    test_brands_ME = test %>% filter(ME_EXP == "ME") %>% select(Brand_Name) %>% distinct() %>% setdiff(train_brands_ME)
    
    train_brands_EXP = train %>% filter(ME_EXP == "EXP") %>% select(Brand_Name) %>% distinct()
    test_brands_EXP = test %>% filter(ME_EXP == "EXP") %>% select(Brand_Name) %>% distinct() %>% setdiff(train_brands_EXP)
    
    all_brands_grid_ME = expand.grid(brand1 = test_brands_ME$Brand_Name,
                                     brand2 = train_brands_ME$Brand_Name,
                                     stringsAsFactors = F) %>%
      mutate(jw_score = 1 - stringdist(a = brand1,
                                       b = brand2,
                                       method = "jw")) %>%
      filter(jw_score != 1 & jw_score > 0.7) %>%
      select(-jw_score) %>%
      rename(test_Brand_Name = brand1, Brand_Name = brand2)
    
    all_brands_grid_EXP = expand.grid(brand1 = test_brands_EXP$Brand_Name,
                                      brand2 = train_brands_EXP$Brand_Name,
                                      stringsAsFactors = F) %>%
      mutate(jw_score = 1 - stringdist(a = brand1,
                                       b = brand2,
                                       method = "jw")) %>%
      filter(jw_score != 1 & jw_score > 0.7) %>%
      select(-jw_score) %>%
      rename(test_Brand_Name = brand1, Brand_Name = brand2)
    
    new_brands_ME = left_join(ADS_dev_features_ME[[1]], all_brands_grid_ME) %>%
      mutate(Brand_Name = if_else(is.na(test_Brand_Name), as.character(Brand_Name), test_Brand_Name)) %>%
      group_by(Brand_Name) %>%
      summarise_at(.cols = vars(starts_with("Brand_Name.vol_")),
                   .funs = funs(mean(.))) %>%
      rbind(ADS_dev_features_ME[[1]]) %>%
      distinct()
    new_brands_EXP = left_join(ADS_dev_features_EXP[[1]], all_brands_grid_EXP) %>%
      mutate(Brand_Name = if_else(is.na(test_Brand_Name), as.character(Brand_Name), test_Brand_Name)) %>%
      group_by(Brand_Name) %>%
      summarise_at(.cols = vars(starts_with("Brand_Name.vol_")),
                   .funs = funs(mean(.))) %>%
      rbind(ADS_dev_features_EXP[[1]]) %>%
      distinct()
    
    ADS_dev_features_ME[[1]] = new_brands_ME
    ADS_dev_features_EXP[[1]] = new_brands_EXP
    
    # for tillrolls ####
    train_tillroll_ME = train %>% filter(ME_EXP == "ME") %>% select(Till_Roll_Description) %>% distinct()
    test_tillroll_ME = test %>% filter(ME_EXP == "ME") %>% select(Till_Roll_Description) %>% distinct() %>% setdiff(train_tillroll_ME)
    
    train_tillroll_EXP = train %>% filter(ME_EXP == "EXP") %>% select(Till_Roll_Description) %>% distinct()
    test_tillroll_EXP = test %>% filter(ME_EXP == "EXP") %>% select(Till_Roll_Description) %>% distinct() %>% setdiff(train_tillroll_EXP)
    
    all_tillroll_grid_ME = expand.grid(tillroll1 = test_tillroll_ME$Till_Roll_Description,
                                       tillroll2 = train_tillroll_ME$Till_Roll_Description,
                                       stringsAsFactors = F) %>%
      mutate(jw_score = 1 - stringdist(a = tillroll1, b = tillroll2, method = "jw")) %>%
      filter(jw_score != 1 & jw_score > 0.7) %>%
      select(-jw_score) %>%
      rename(test_Till_Roll_Description = tillroll1, Till_Roll_Description = tillroll2)
    
    all_tillroll_grid_EXP = expand.grid(tillroll1 = test_tillroll_EXP$Till_Roll_Description,
                                        tillroll2 = train_tillroll_EXP$Till_Roll_Description,
                                        stringsAsFactors = F) %>%
      mutate(jw_score = 1 - stringdist(a = tillroll1, b = tillroll2, method = "jw")) %>%
      filter(jw_score != 1 & jw_score > 0.7) %>%
      select(-jw_score) %>%
      rename(test_Till_Roll_Description = tillroll1, Till_Roll_Description = tillroll2)
    
    new_tillrolldescs_ME = left_join(ADS_dev_features_ME[[2]], all_tillroll_grid_ME) %>%
      mutate(Till_Roll_Description = if_else(is.na(test_Till_Roll_Description), as.character(Till_Roll_Description), test_Till_Roll_Description)) %>%
      group_by(Till_Roll_Description) %>%
      summarise_at(.cols = vars(starts_with("Till_Roll_Description.vol_")),
                   .funs = funs(mean(.))) %>%
      rbind(ADS_dev_features_ME[[2]]) %>%
      distinct()
    
    new_tillrolldescs_EXP = left_join(ADS_dev_features_EXP[[2]], all_tillroll_grid_EXP) %>%
      mutate(Till_Roll_Description = if_else(is.na(test_Till_Roll_Description), as.character(Till_Roll_Description), test_Till_Roll_Description)) %>%
      group_by(Till_Roll_Description) %>%
      summarise_at(.cols = vars(starts_with("Till_Roll_Description.vol_")),
                   .funs = funs(mean(.))) %>%
      rbind(ADS_dev_features_EXP[[2]]) %>%
      distinct()
    
    ADS_dev_features_ME[[2]] = new_tillrolldescs_ME
    ADS_dev_features_EXP[[2]] = new_tillrolldescs_EXP
  }
  
  # snippet to add the features to the test dataset as well
  {
    test_ME = test %>% filter(ME_EXP == "ME")
    test_EXP = test %>% filter(ME_EXP == "EXP")
    
    for (i in 1:length(categ_variables))
    {
      x_ME = ADS_dev_features_ME[[i]]
      x_EXP = ADS_dev_features_EXP[[i]]
      
      #x[, 1] = as.character(x[, 1])
      test_ME = left_join(test_ME, x_ME)
      test_EXP = left_join(test_EXP, x_EXP)
    }
    
    test = rbind(test_ME, test_EXP)
    test = test[, base_columns]
    
    # snippet to convert all NAs to median of the column (for test)
    {
      test %<>% mutate(index = row_number(Base_Product_Number))
      
      test_nas_to_median_ME = test %>%
        filter(ME_EXP == "ME") %>%
        select(contains(".vol_"), index) %>%
        as.matrix()
      
      test_nas_to_median_EXP = test %>%
        filter(ME_EXP == "EXP") %>%
        select(contains(".vol_"), index) %>%
        as.matrix()
      
      indx_ME = which(is.na(test_nas_to_median_ME), arr.ind = TRUE)
      indx_EXP = which(is.na(test_nas_to_median_EXP), arr.ind = TRUE)
      
      test_nas_to_median_ME[indx_ME] = matrixStats::colMedians(test_nas_to_median_ME, na.rm = TRUE)[indx_ME[, 2]]
      test_nas_to_median_EXP[indx_EXP] = matrixStats::colMedians(test_nas_to_median_EXP, na.rm = TRUE)[indx_EXP[, 2]]
      
      test_nas_to_median = rbind(test_nas_to_median_EXP, test_nas_to_median_ME) %>% data.frame()
      
      test %<>% select(-contains(".vol_")) %>% left_join(test_nas_to_median) %>% select(-index)
    }
    
    # snippet to convert all NAs to median of the column (for train)
    {
      train %<>% mutate(index = row_number(Base_Product_Number))
      
      train_nas_to_median_ME = train %>%
        filter(ME_EXP == "ME") %>%
        select(contains(".vol_"), index) %>%
        as.matrix()
      
      train_nas_to_median_EXP = train %>%
        filter(ME_EXP == "EXP") %>%
        select(contains(".vol_"), index) %>%
        as.matrix()
      
      indx_ME = which(is.na(train_nas_to_median_ME), arr.ind = TRUE)
      indx_EXP = which(is.na(train_nas_to_median_EXP), arr.ind = TRUE)
      
      train_nas_to_median_ME[indx_ME] = matrixStats::colMedians(train_nas_to_median_ME, na.rm = TRUE)[indx_ME[, 2]]
      train_nas_to_median_EXP[indx_EXP] = matrixStats::colMedians(train_nas_to_median_EXP, na.rm = TRUE)[indx_EXP[, 2]]
      
      train_nas_to_median = rbind(train_nas_to_median_EXP, train_nas_to_median_ME) %>% data.frame()
      
      train %<>% select(-contains(".vol_")) %>% left_join(train_nas_to_median) %>% select(-index)
    }
  }
}


# removing some dfs
{
  rm(ADS_dev_features_ME,ADS_dev_features_EXP, char_data, categ_variables, base_columns)
  rm(new_brands_EXP, new_brands_ME, new_tillrolldescs_EXP, new_tillrolldescs_ME, test_brands_EXP, test_brands_ME, test_tillroll_EXP, test_tillroll_ME, train_brands_EXP, train_brands_ME, train_tillroll_EXP, train_tillroll_ME, all_brands_grid_EXP, all_brands_grid_ME, all_tillroll_grid_EXP, all_tillroll_grid_ME)
  rm(test_EXP, test_ME, train_EXP, train_ME, x_EXP, x_ME)
  rm(indx_EXP, indx_ME, test_nas_to_median, test_nas_to_median_EXP, test_nas_to_median_ME, train_nas_to_median, train_nas_to_median_EXP, train_nas_to_median_ME)
}


# Creating the H-M-L buckets for PSG
{
  ADS_6YR_iter2 = rbind(train, test)
  
  ADS_PSG_BPN_YEAR_AVG = ADS_6YR_iter2 %>%
    group_by(Product_Sub_Group_Code) %>%
    summarise(vol_sum = sum(vol),
              vol_avg_prod = sum(vol)/length(unique(Base_Product_Number)),
              count_prod = length(unique(Base_Product_Number)),
              vol_avg_prod_month = vol_avg_prod/length(unique(Year_Month))) %>%
    ungroup() %>%
    mutate(vol_sum_norm = (vol_sum - min(vol_sum)) / (max(vol_sum) - min(vol_sum)),
           vol_avg_prod_norm = (vol_avg_prod - min(vol_avg_prod)) / (max(vol_avg_prod) - min(vol_avg_prod)),
           count_prod_norm = (count_prod - min(count_prod)) / (max(count_prod) - min(count_prod)),
           vol_avg_prod_month_norm = (vol_avg_prod_month - min(vol_avg_prod_month)) / (max(vol_avg_prod_month) - min(vol_avg_prod_month)),
           final_score = 0*vol_sum_norm + 0*vol_avg_prod_norm + 0*count_prod_norm + 1*vol_avg_prod_month_norm,
           bucket = if_else(final_score < quantile(final_score, probs = 0.4),
                            "LOW", if_else(final_score < quantile(final_score, probs = 0.75),
                                           "MEDIUM", "HIGH"))) %>%
    select(Product_Sub_Group_Code, bucket)
  
  ADS_6YR_iter2 = left_join(ADS_6YR_iter2, ADS_PSG_BPN_YEAR_AVG) %>%
    rename(PSG_bucket = bucket)
  
  rm(ADS_PSG_BPN_YEAR_AVG)
}


# RPart SEGMENT ####
{
  ADS_rpart = ADS_6YR_iter2 %>%
    select(Product_Sub_Group_Code, Brand_Name, parent_supplier, Brand_Ind, BUYER, JUNIOR_BUYER, MERCHANDISE_GROUP_CODE, Package_Type, measure_type, Size, SI, asp, acp, price_band, ME_EXP, PSG_bucket, Till_Roll_Description, vol,
           Year_Month, Launch_Month, Launch_Week_Number, Quarter_Number, Period_Number, holidays_sum, Launch_Year_Month,
           NO_Stores_ranged, NO_5K_Stores_ranged, price_band_prod_count) %>%
    distinct()
  
  ADS_rpart = ADS_6YR_iter2
  
  for (i in 1:ncol(ADS_rpart)) {
    if ((class(ADS_rpart[, i]) == "character")) {
      ADS_rpart[, i] = as.factor(ADS_rpart[, i])
    }
  }
  
  ADS_rpart_train = ADS_rpart %>%
    filter(Year_Month < 201601 & Launch_Year_Month < 201601) %>%
    select(-Launch_Year_Month, -Year_Month, -Year_Number, -Year_Quarter, -Launch_Year_Quarter)
  
  ADS_rpart_test = ADS_rpart %>%
    filter(Year_Month >= 201601 & Launch_Year_Month >= 201601) %>%
    select(-Launch_Year_Month, -Year_Month, -Year_Number, -Year_Quarter, -Launch_Year_Quarter)
  
  rpart_model = rpart::rpart(formula = vol ~ ., data = ADS_rpart_train, control = rpart.control(minbucket = 2, maxdepth = 30, minsplit = 5))
  
  output = predict(object = rpart_model, newdata = ADS_rpart_test) %>% data.frame()
  x = data.frame(actuals = ADS_rpart_test$vol) %>%
    cbind(output) %>%
    dplyr::rename_(.dots = setNames(names(.), c("actuals", "pred"))) %>%
    mutate(APE = (actuals - pred)/pred)
}


# generic train/test split to be used for intermediate purposes only - leave commented if not sure why we have this
{
  train = ADS_6YR_iter2 %>%
    filter(Year_Month < 201601 & Launch_Year_Month < 201601)
  
  test = ADS_6YR_iter2 %>%
    filter(Year_Month >= 201601 & Launch_Year_Month >= 201601)
}


save.image(paste0(BC, "_MEEXP_Monthly_backup.RData"))
load(paste0(BC, "_MEEXP_Monthly_backup.RData"), envir = .GlobalEnv)


# RANGER SEGMENT ####
{
  ADS_6YR_iter2_inter = ADS_6YR_iter2
  
  ADS_6YR_iter2 = rbind(train, test)
  
  for (i in 1:ncol(ADS_6YR_iter2)) {
    if ((class(ADS_6YR_iter2[, i]) == "factor")) {
      ADS_6YR_iter2[, i] = as.character(ADS_6YR_iter2[, i])
    }
  }
  
  # converting whatever character columns we can into factors
  for (i in 1:ncol(ADS_6YR_iter2)) {
    if ((class(ADS_6YR_iter2[, i]) == "character") & (length(unique(ADS_6YR_iter2[, i])) < 53)) {
      ADS_6YR_iter2[, i] = as.factor(ADS_6YR_iter2[, i])
    }
  }
  
  # need to make dummies for the other categorical features
  dummy_columns_to_encode = c()
  for (i in 1:ncol(ADS_6YR_iter2)) {
    if ((class(ADS_6YR_iter2[, i]) == "character") & (length(unique(ADS_6YR_iter2[, i])) >= 53)) {
      dummy_columns_to_encode = append(dummy_columns_to_encode, names(ADS_6YR_iter2)[i])
      print(names(ADS_6YR_iter2)[i])
    }
  }
  
  dummy_columns_to_encode = setdiff(dummy_columns_to_encode, "Till_Roll_Description")
  
  # creating the test set for checking the accuracy after prediction
  test_before = test
  
  ADS_6YR_iter2 = dummy.data.frame(data = ADS_6YR_iter2, names = dummy_columns_to_encode, sep = "_", fun = as.numeric)
  colnames(ADS_6YR_iter2) = make.names(colnames(ADS_6YR_iter2), unique = T)
  
  
  #creating the test and train datasets
  #timeframe used is Time_start to Time_end
  train <- ADS_6YR_iter2 %>% filter(Year_Number < 2016)
  
  test <- ADS_6YR_iter2 %>% filter(Launch_Year_Month >= 201601)
  
  X_cols = !colnames(train) %in% c("Base_Product_Number", "Year_Month", "Year_Quarter", "Year_Number", "Launch_Year_Month", "Launch_Year_Quarter")
  train_X <- data.frame(train[,X_cols])
  
  Y_cols = !colnames(test) %in% c("Base_Product_Number", "Year_Month", "Year_Quarter", "Year_Number", "Launch_Year_Month", "Launch_Year_Quarter", "vol")
  test_X <- data.frame(test[,Y_cols])
  
  # Modelling
  {
    set.seed(121)
    parameter_grid = expand.grid(num.trees = c(50, 500),
                                 mtry = c(ceiling(ncol(train_X)/2), floor(ncol(train_X) - 5))) %>%
      as.data.frame()
    accuracy = matrix(nrow = nrow(parameter_grid), ncol = 7)
    
    predictions_ranger = test_before
    
    for (i in 1:nrow(parameter_grid))
    {
      i = i
      
      #
      gc()
      
      #creating models
      fitrandom = ranger(vol ~ ., data = train_X, num.trees = parameter_grid[i, 1], mtry = parameter_grid[i, 2], importance = "none", verbose = T, respect.unordered.factors = TRUE)
      
      #prediction
      out = predict(object = fitrandom, data = test_X, verbose = T)
      pred_vol = out$predictions
      pred_vol[is.nan(pred_vol)] = 0
      final_out = cbind(test_before, pred_vol)
      
      pred_vol = data.frame(pred_vol)
      colnames(pred_vol) = paste0("pred_ranger_", i)
      
      assign(x = paste0("pred_ranger_", i), value = data.frame(pred_vol), envir = .GlobalEnv)
      
      product_level_out <- final_out %>% group_by(Base_Product_Number) %>% summarize(actual = sum(as.numeric(vol)), pred = sum(as.numeric(pred_vol))) %>% data.frame()
      
      # product level out
      product_level_out$APE <- (abs((product_level_out$pred - product_level_out$actual)/product_level_out$actual))*100
      product_level_out$actual = as.numeric(product_level_out$actual)
      
      # Accuracy segment
      colnames(accuracy) = c("Model","RMSE", "MAE", "MAPE", "WAPE", "products_>75%_acc", "vol_contrib")
      accuracy[i,1] = i
      
      # Metrics for measuring accuracy
      {
        # RMSE
        accuracy[i,2] = sqrt(mean((product_level_out$pred - product_level_out$actual)^2))
        
        # MAE
        accuracy[i,3] = mean(abs(product_level_out$pred - product_level_out$actual))
        prod_ap <- (abs((product_level_out$pred - product_level_out$actual)/product_level_out$actual))*100
        prod_ap[is.infinite(prod_ap)] <- 0
        
        # MAPE
        accuracy[i,4] = mean(prod_ap)
        
        # WAPE
        accuracy[i,5] = 100*sum(abs(product_level_out$pred - product_level_out$actual))/sum(product_level_out$actual)
        
        # % of products greater than 75% accuracy
        accuracy[i,6] = (sum(product_level_out$APE <= 25)/nrow(product_level_out))*100
        
        # Volume contribution by these products
        accuracy[i,7] = (sum(product_level_out[product_level_out$APE <= 25,"actual"])/sum(product_level_out$actual))*100
      }
      
      # save the predictions
      predictions_ranger = cbind(predictions_ranger, get(paste0("pred_ranger_", i)))
    }
    
    # Final output
    Ranger_Final_results = cbind(accuracy, parameter_grid)
    saveRDS(Ranger_Final_results, paste0("Ranger_Final_Results_MEEXP_Monthly_", BC, ".rds"))
    
    write.csv(predictions_ranger, "RANGER_predictions_MFV.csv", row.names = F)
    saveRDS(predictions_ranger, "predictions_ranger.rds")
    
    # importance and summary of model (only works if you have explicitly mentioned importance in the ranger model. it has been put to none for speed for now)
    {
      # ranger::importance(fitrandom)
      # fitrandom
      #
      # z = data.frame(attributes(fitrandom$variable.importance))
      #
      # Imp = data.frame(Feature = z, Imp = importance(fitrandom)) %>%
      #   mutate(Rank = dense_rank(desc(Imp)))
    }
  }
}

# system('rm -f ranger_product_level_out_*')
# system('rm -f ranger_accuracy_*')


# XGBOOST SEGMENT ####
{
  ADS_6YR_iter2 = ADS_6YR_iter2_inter
  
  train = ADS_6YR_iter2 %>%
    filter(Year_Month < 201601 & Launch_Year_Month < 201601)
  
  test = ADS_6YR_iter2 %>%
    filter(Year_Month >= 201601 & Launch_Year_Month >= 201601)
  
  # getting the important parent suppliers
  {
    ADS_6YR_iter2_X = train %>% group_by(Base_Product_Number, Year_Month, parent_supplier) %>% summarise(vol = sum(vol)) %>% ungroup() %>% data.frame()
    
    ADS_6YR_iter2_X = ADS_6YR_iter2_X %>% dummy.data.frame(., names = c("parent_supplier"), sep = "_", fun = as.numeric) %>% data.frame()
    
    dim_cols = c("Base_Product_Number", "Year_Month", "vol")
    X_cols = !colnames(ADS_6YR_iter2_X) %in% dim_cols
    train_X = as.matrix(ADS_6YR_iter2_X[,X_cols])
    train_Y = as.matrix(as.numeric(ADS_6YR_iter2_X[,'vol']))
    
    XGB_Model = xgboost(data = train_X, label = train_Y, nrounds = 100, verbose = 1, booster = "gbtree", eta = 0.01, max_depth = 7, objective = "reg:linear", print_every_n = 25)
    
    xgb_importance = data.table(xgboost::xgb.importance(feature_names = colnames(train_X), model = XGB_Model))
    
    Importance_table = data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
      mutate(Rank = dense_rank(desc(Importance))) %>%
      filter(Rank <= 10)
    colnames_features_parent_suppliers = as.vector(Importance_table$Feature) %>% setdiff(c("parent_supplier_Not.Available"))
    }
  
  # getting the important till roll description
  {
    ADS_6YR_iter2_X = train %>% group_by(Base_Product_Number, Year_Month, Till_Roll_Description) %>% summarise(vol = sum(vol)) %>% ungroup() %>% data.frame()
    
    top_till_roll = ADS_6YR_iter2_X %>%
      group_by(Till_Roll_Description) %>%
      summarise(vol = sum(as.numeric(vol))) %>%
      mutate(rank = dense_rank(-vol)) %>%
      filter(rank < 50) %>%
      mutate(Till_Roll_Description = make.names(paste0("Till_Roll_Description_", Till_Roll_Description), unique = T)) %>%
      select(Till_Roll_Description)
    
    ADS_6YR_iter2_X = ADS_6YR_iter2_X %>% dummy.data.frame(., names = c("Till_Roll_Description"), sep = "_", fun = as.numeric) %>% data.frame()
    
    X_cols = colnames(ADS_6YR_iter2_X) %in% top_till_roll$Till_Roll_Description
    
    train_X = as.matrix(ADS_6YR_iter2_X[,X_cols])
    train_Y = as.matrix(as.numeric(ADS_6YR_iter2_X[,'vol']))
    
    XGB_Model = xgboost(data = train_X, label = train_Y, nrounds = 100, verbose = 1, booster = "gbtree", eta = 0.01, max_depth = 7, objective = "reg:linear", print_every_n = 25)
    
    xgb_importance = data.table(xgboost::xgb.importance(feature_names = colnames(train_X), model = XGB_Model))
    
    Importance_table = data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
      mutate(Rank = dense_rank(desc(Importance))) %>%
      filter(Rank <= 25)
    colnames_features_till_roll = as.vector(Importance_table$Feature)
  }
  
  # getting the important Brand names
  {
    ADS_6YR_iter2_X = train %>% group_by(Base_Product_Number, Year_Month, Brand_Name) %>% summarise(vol = sum(vol)) %>% ungroup() %>% data.frame()
    
    ADS_6YR_iter2_X = ADS_6YR_iter2_X %>% dummy.data.frame(., names = c("Brand_Name"), sep = "_", fun = as.numeric) %>% data.frame()
    
    dim_cols = c("Base_Product_Number", "Year_Month", "vol")
    X_cols = !colnames(ADS_6YR_iter2_X) %in% dim_cols
    train_X = as.matrix(ADS_6YR_iter2_X[,X_cols])
    train_Y = as.matrix(as.numeric(ADS_6YR_iter2_X[,'vol']))
    
    XGB_Model = xgboost(data = train_X, label = train_Y, nrounds = 100, verbose = 1, booster = "gbtree", eta = 0.01, max_depth = 7, objective = "reg:linear", print_every_n = 25)
    
    xgb_importance = data.table(xgboost::xgb.importance(feature_names = colnames(train_X), model = XGB_Model))
    
    Importance_table = data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
      mutate(Rank = dense_rank(desc(Importance))) %>%
      filter(Rank <= 20)
    colnames_features_brands = as.vector(Importance_table$Feature)
  }
  
  # adding the relevant features from above to the final list of features
  {
    additional_features = append(colnames_features_brands, colnames_features_parent_suppliers) %>%
      append(., colnames_features_till_roll)
    dummy_names = c("ME_EXP", "Product_Sub_Group_Code", "Package_Type", "price_band", "measure_type", "BUYER", "MERCHANDISE_GROUP_CODE", "Brand_Ind", "Brand_Name", "Till_Roll_Description", "parent_supplier", "JUNIOR_BUYER")
  }
  
  rm(train_X, train_Y, Importance_table, top_till_roll, xgb_importance, XGB_Model,
     colnames_features_brands, colnames_features_parent_suppliers, colnames_features_till_roll,
     dim_cols)
  
  # combine test and train for below exercise alone
  test_before = test
  ADS_6YR_iter2 = rbind(train, test)
  
  # doing one hot encoding for some features for xgboost
  ADS_6YR_iter2_X <- dummy.data.frame(data = ADS_6YR_iter2, names = dummy_names, sep = "_", fun = as.numeric)
  colnames(ADS_6YR_iter2_X) = make.names(colnames(ADS_6YR_iter2_X), unique = T)
  
  x1 = ADS_6YR_iter2_X %>% select(starts_with("Brand_Name_")) %>% colnames() %>% make.names(., unique = T)
  x2 = ADS_6YR_iter2_X %>% select(starts_with("parent_supplier_")) %>% colnames() %>% make.names(., unique = T)
  x3 = ADS_6YR_iter2_X %>% select(starts_with("Till_Roll_Description_")) %>% colnames() %>% make.names(., unique = T)
  x = append(x1, x2) %>% append(., x3) %>% setdiff(., additional_features)
  
  
  ## some steps to remove the features not needed as evaluated above
  ADS_6YR_iter2_X_colnames = !colnames(ADS_6YR_iter2_X) %in% x
  ADS_6YR_iter2_X = ADS_6YR_iter2_X[, ADS_6YR_iter2_X_colnames]
  ADS_6YR_iter2 = ADS_6YR_iter2_X
  
  # removing a lot of unnecessary files
  rm(ADS_6YR_iter2_X, ADS_6YR_iter2_X_colnames, dummy_names, additional_features,
     x, x1, x2, x3, X_cols)
  
  #creating the test and train datasets
  #timeframe used is Time_start to Time_end
  train = ADS_6YR_iter2 %>% filter(Year_Number < 2016)
  test <- ADS_6YR_iter2 %>% filter(Launch_Year_Month >= 201601)
  
  # Initialization
  {
    dim_cols = c("Base_Product_Number", "Year_Month" ,"vol", "Year_Quarter", "Year_Number", "Launch_Year_Month", "Launch_Year_Quarter")
    X_cols = !colnames(train) %in% dim_cols
    gc()
    #train_X <- xgb.DMatrix(data = data.matrix(lapply(train[,X_cols], as.numeric)), label = data.matrix(as.numeric(train[,'vol'])))
    train_X = as.matrix(sapply(train[,X_cols], as.numeric))
    train_Y = as.matrix(as.numeric(train[,'vol']))
    test_X = data.matrix(sapply(test[,X_cols], as.numeric))
  }
  
  # Modelling-1
  {
    XGB_Model = xgboost(data = train_X, label = train_Y, nrounds = 200, verbose = 1, booster = "gbtree", eta = 0.01, max_depth = 7, objective = "reg:linear", print_every_n = 20)
    
    #if model already present
    xgb_importance = data.table(xgboost::xgb.importance(feature_names = colnames(train_X), model = XGB_Model))
    #xgboost::xgb.plot.importance(xgb_importance)
    
    Importance_table = data.frame(Feature = xgb_importance$Feature, Importance = xgb_importance$Gain) %>%
      mutate(Rank = dense_rank(desc(Importance))) %>%
      filter(Rank <= 150)
    colnames_features = as.vector(Importance_table$Feature)
    
    
    XX_cols = colnames(train) %in% colnames_features
    XXX_cols = X_cols & XX_cols
    
    train_X = as.matrix(sapply(train[, XXX_cols], as.numeric))
    train_Y = as.matrix(as.numeric(train[, 'vol']))
    test_X = as.matrix(sapply(test[, XXX_cols], as.numeric))
  }
  
  # Modelling-2
  {
    set.seed(121)
    parameter_grid = expand.grid(eta = c(0.01, 0.1),
                                 nrounds = c(10, 200),
                                 max_depth = c(10, 15)) %>%
      #rowwise() %>%
      #mutate(rand = rbinom(1, size = 1, prob = 0.35)) %>%
      #filter(rand == 1) %>%
      as.data.frame()
    parameter_grid = parameter_grid[!(parameter_grid$eta == 0.001 & parameter_grid$nrounds == 10), ]
    
    accuracy = matrix(nrow = nrow(parameter_grid), ncol = 7)
    
    predictions_xgboost = test_before
    
    for (i in 1:nrow(parameter_grid))
    {
      i = i
      #seed
      set.seed(i)
      gc()
      
      #print current model
      print(paste0("XGB_Model_", i))
      
      #creating the model
      XGB_Model = xgboost(data = train_X, label = train_Y, booster = "gbtree", nrounds = parameter_grid[i, 2], eta = parameter_grid[i, 1], max_depth = parameter_grid[i, 3], verbose = 1, print_every_n = 25, early_stopping_rounds = 2, objective = "reg:linear")
      
      #prediction
      out <- predict(XGB_Model, test_X)
      final_out <- cbind(test_before, out)
      product_level_out <- final_out %>% dplyr::group_by(Base_Product_Number) %>% dplyr::summarize(actual = sum(vol), pred = sum(out)) %>% data.frame()
      
      out = data.frame(out)
      colnames(out) = paste0("pred_xgboost_", i)
      
      assign(x = paste0("pred_xgboost_", i), value = data.frame(out), envir = .GlobalEnv)
      
      #accuracy
      product_level_out$APE <- (abs((product_level_out$pred - product_level_out$actual)/product_level_out$actual))*100
      product_level_out$actual = as.numeric(product_level_out$actual)
      
      # Accuracy segment
      colnames(accuracy) = c("Model","RMSE", "MAE", "MAPE", "WAPE", "products_>75%_acc", "vol_contrib")
      accuracy[i,1] = i
      # Metrics for measuring accuracy
      {
        # RMSE
        accuracy[i,2] = sqrt(mean((product_level_out$pred - product_level_out$actual)^2))
        
        # MAE
        accuracy[i,3] = mean(abs(product_level_out$pred - product_level_out$actual))
        prod_ap <- (abs((product_level_out$pred - product_level_out$actual)/product_level_out$actual))*100
        prod_ap[is.infinite(prod_ap)] <- 0
        
        # MAPE
        accuracy[i,4] = mean(prod_ap)
        
        # WAPE
        accuracy[i,5] = 100*sum(abs(product_level_out$pred - product_level_out$actual))/sum(product_level_out$actual)
        
        # % of products greater than 75% accuracy
        accuracy[i,6] = (sum(product_level_out$APE <= 25)/nrow(product_level_out))*100
        
        # Volume contribution by these products
        accuracy[i,7] = (sum(product_level_out[product_level_out$APE <= 25,"actual"])/sum(product_level_out$actual))*100
      }
      
      # save the predictions
      predictions_xgboost = cbind(predictions_xgboost, get(paste0("pred_xgboost_", i)))
    }
    
    # Final output
    Xgboost_Final_results = cbind(accuracy, parameter_grid)
    saveRDS(Xgboost_Final_results, paste0("XGBOOST_Final_Results_MEEXP_Monthly_", BC, ".rds"))
    
    write.csv(predictions_xgboost, "XGBOOST_predictions_MFV.csv", row.names = F)
    saveRDS(predictions_xgboost, "predictions_xgboost.rds")
  }
}

# system('rm -f xgboost_product*')
# system('rm -f xgb_model*')


# ENSEMBLE ####
{
  # write the final predictions set for ensemble modelling
  combined_predictions = inner_join(predictions_ranger, predictions_xgboost)
  write.csv(combined_predictions, "combined_predictions.csv", row.names = F)
  
  combined_predictions = predictions_xgboost
  
  # XGBOOST
  {
    # read the predictions file and create the train/test sets
    {
      combined_predictions = read.csv("combined_predictions.csv", header = T, as.is = T) %>%
        select(starts_with("pred_"), vol, Base_Product_Number)
      
      sample_pop = floor(0.8 * nrow(combined_predictions))
      
      ## set the seed to make the partition reproductible
      set.seed(123)
      train_ind = sample(seq_len(nrow(combined_predictions)), size = sample_pop)
      
      trainX = combined_predictions[train_ind, ] %>% select(-vol, -Base_Product_Number) %>% data.matrix()
      trainY = combined_predictions[train_ind, "vol"] %>% data.matrix()
      test = combined_predictions[-train_ind, ] %>% select(-vol, -Base_Product_Number) %>% data.matrix()
      test_validate = combined_predictions[-train_ind, ] %>% data.frame()
    }
    
    # the mother of all loops
    {
      parameter_grid = expand.grid(eta = c(0.001, 0.01),
                                   nrounds = c(10, 100, 200, 1000),
                                   max_depth = c(8, 10))
      
      # Accuracy segment
      accuracy_xgboost = data.frame("Model" = 0,
                                    "RMSE" = 0,
                                    "MAE" = 0,
                                    "MAPE" = 0,
                                    "WAPE" = 0,
                                    "products_>75%_acc" = 0,
                                    "vol_contrib" = 0,
                                    "eta" = 0,
                                    "nrounds" = 0,
                                    "max_depth" = 0)
      
      
      for (i in 1:nrow(parameter_grid))
      {
        # create the model and importance table
        {
          print(paste0("XGB_MODEL_", i))
          XGB_Model = xgboost(data = trainX, label = trainY, nrounds = parameter_grid[i, 2], verbose = 1, booster = "gbtree", eta = parameter_grid[i, 1], max_depth = parameter_grid[i, 3], objective = "reg:linear", print_every_n = 25, early_stopping_rounds = 2)
        }
        
        # do the prediction and the validation sets
        out = predict(XGB_Model, test)
        final_out = cbind(test_validate, out)
        product_level_out = final_out %>% dplyr::group_by(Base_Product_Number) %>% dplyr::summarize(actual = sum(vol), pred = sum(out)) %>% data.frame()
        
        
        #accuracy_xgboost
        product_level_out$APE <- (abs((product_level_out$pred - product_level_out$actual)/product_level_out$actual))*100
        product_level_out$actual = as.numeric(product_level_out$actual)
        
        
        accuracy_xgboost[i,1] = paste("Ensemble_", i)
        
        # Metrics for measuring accuracy_xgboost
        {
          # RMSE
          accuracy_xgboost[i,2] = sqrt(mean((product_level_out$pred - product_level_out$actual)^2))
          
          # MAE
          accuracy_xgboost[i,3] = mean(abs(product_level_out$pred - product_level_out$actual))
          prod_ap <- (abs((product_level_out$pred - product_level_out$actual)/product_level_out$actual))*100
          prod_ap[is.infinite(prod_ap)] <- 0
          
          # MAPE
          accuracy_xgboost[i,4] = mean(prod_ap)
          
          # WAPE
          accuracy_xgboost[i,5] = 100*sum(abs(product_level_out$pred - product_level_out$actual))/sum((product_level_out$actual))
          
          # % of products greater than 75% accuracy
          accuracy_xgboost[i,6] = (sum(product_level_out$APE <= 25)/nrow(product_level_out))*100
          
          # Volume contribution by these products
          accuracy_xgboost[i,7] = (sum(product_level_out[product_level_out$APE <= 25,"actual"])/sum(product_level_out$actual))*100
        }
        
        # add the parameters of the model
        {
          accuracy_xgboost[i, 8] = parameter_grid[i, 1]
          accuracy_xgboost[i, 9] = parameter_grid[i, 2]
          accuracy_xgboost[i, 10] = parameter_grid[i, 3]
          accuracy_xgboost[i, 11] = parameter_grid[i, 4]
        }
      }
    }
  }
  
  # RANDOMFOREST
  {
    # read the predictions file and create the train/test sets
    {
      combined_predictions = read.csv("combined_predictions.csv", header = T, as.is = T) %>%
        select(starts_with("pred_"), vol, Base_Product_Number, Brand_Ind, BUYER, Package_Type, Product_Sub_Group_Code, NO_Stores_ranged, Period_Number, Size, measure_type, MERCHANDISE_GROUP_CODE, parent_supplier, Brand_Name, JUNIOR_BUYER, ME_EXP, SI, price_band, Launch_Month, NO_Stores_ranged, acp, asp, Quarter_Number) %>%
        mutate(pred_ranger_avg = (pred_ranger_1 + pred_ranger_2)/2,
               pred_xgboost_avg = (pred_xgboost_1 + pred_xgboost_2 + pred_xgboost_3 + pred_xgboost_4 + pred_xgboost_5 + pred_xgboost_6)/6,
               pred_all_avg = (pred_xgboost_1 + pred_xgboost_2 + pred_xgboost_3 + pred_xgboost_4 + pred_xgboost_5 + pred_xgboost_6 + pred_ranger_1 + pred_ranger_2)/8,
               pred_all_min = pmin(pred_xgboost_1, pred_xgboost_2, pred_xgboost_3, pred_xgboost_4, pred_xgboost_5, pred_xgboost_6, pred_ranger_1, pred_ranger_2),
               pred_all_max = pmax(pred_xgboost_1, pred_xgboost_2, pred_xgboost_3, pred_xgboost_4, pred_xgboost_5, pred_xgboost_6, pred_ranger_1, pred_ranger_2))
      
      
      # converting whatever character columns we can into factors
      for (i in 1:ncol(combined_predictions)) {
        if ((class(combined_predictions[, i]) == "character") && (length(unique(combined_predictions[, i])) < 53)) {
          combined_predictions[, i] = as.factor(combined_predictions[, i])
        }
      }
      
      # need to make dummies for the other categorical features
      dummy_columns_to_encode = c()
      for (i in 1:ncol(combined_predictions)) {
        if ((class(combined_predictions[, i]) == "character") && (length(unique(combined_predictions[, i])) >= 53)) {
          dummy_columns_to_encode = append(dummy_columns_to_encode, names(combined_predictions)[i])
          print(names(combined_predictions)[i])
        }
      }
    }
    
    # the mother of all loops
    {
      unique_bpns = unique(combined_predictions$Base_Product_Number) %>% data.frame()
      
      #initialise the iterator for adding the results later
      j = 1
      
      parameter_grid = expand.grid(num.trees = c(10, 500),
                                   mtry = c(floor(ncol(combined_predictions)/2)),
                                   splitrule = c("variance", "extratrees")) %>%
        as.data.frame()
      
      # Accuracy segment
      accuracy_rf = data.frame("Model" = 0,
                               "RMSE" = 0,
                               "MAE" = 0,
                               "MAPE" = 0,
                               "WAPE" = 0,
                               "products_>75%_acc" = 0,
                               "vol_contrib" = 0,
                               "numtrees" = 0,
                               "mtry" = 0,
                               "splitrule" = 0,
                               "bpn" = 0,
                               "pred" = 0,
                               "actual" = 0,
                               "BPN" = 0)
      
      for (k in 1:nrow(unique_bpns)) {
        print(k)
        cat("\n\n")
        
        test_validate = combined_predictions[combined_predictions$Base_Product_Number == unique_bpns[k, 1], ]
        
        if (!is.null(dummy_columns_to_encode)) {
          combined_predictions = dummy.data.frame(data = combined_predictions, names = dummy_columns_to_encode, sep = "_", fun = as.numeric)
          colnames(combined_predictions) = make.names(colnames(combined_predictions), unique = T)
        }
        
        trainX = combined_predictions[combined_predictions$Base_Product_Number != unique_bpns[k, 1], ] %>% select(-Base_Product_Number) %>% data.frame()
        test = combined_predictions[combined_predictions$Base_Product_Number == unique_bpns[k, 1], ] %>% select(-vol, -Base_Product_Number) %>% data.frame()
        
        #always_variables = trainX %>% select(starts_with("pred_")) %>% colnames(.)
        
        for (i in 1:nrow(parameter_grid))
        {
          #creating models
          if (parameter_grid[i, 3] == "extratrees")
          {
            fitrandom = ranger(vol ~ ., data = trainX, num.trees = parameter_grid[i, 1], mtry = parameter_grid[i, 2], importance = "impurity", verbose = T, respect.unordered.factors = TRUE, splitrule = parameter_grid[i, 3], num.random.splits = 10)
          } else {
            fitrandom = ranger(vol ~ ., data = trainX, num.trees = parameter_grid[i, 1], mtry = parameter_grid[i, 2], importance = "none", verbose = T, respect.unordered.factors = TRUE, splitrule = parameter_grid[i, 3])
          }
          
          print(paste0("Ranger_Ensemble_", i, "_", k))
          cat("\n")
          
          #prediction
          out = predict(object = fitrandom, data = test, verbose = T)
          pred_vol = out$predictions
          pred_vol[is.nan(pred_vol)] = 0
          final_out = cbind(test_validate, pred_vol)
          
          product_level_out <- final_out %>% group_by(Base_Product_Number) %>% summarize(actual = sum(as.numeric(vol)), pred = sum(as.numeric(pred_vol))) %>% data.frame()
          
          #accuracy_rf
          product_level_out$APE <- (abs((product_level_out$pred - product_level_out$actual)/product_level_out$actual))*100
          product_level_out$actual = as.numeric(product_level_out$actual)
          
          
          accuracy_rf[i + j -1, 1] = paste0("Ranger_Ensemble_", i, "_", j)
          
          # Metrics for measuring accuracy_rf
          {
            # RMSE
            accuracy_rf[i + j -1, 2] = sqrt(mean((product_level_out$pred - product_level_out$actual)^2))
            
            # MAE
            accuracy_rf[i + j -1, 3] = mean(abs(product_level_out$pred - product_level_out$actual))
            prod_ap <- (abs((product_level_out$pred - product_level_out$actual)/product_level_out$actual))*100
            prod_ap[is.infinite(prod_ap)] <- 0
            
            # MAPE
            accuracy_rf[i + j -1, 4] = mean(prod_ap)
            
            # WAPE
            accuracy_rf[i + j -1, 5] = 100*sum(abs(product_level_out$pred - product_level_out$actual))/sum((product_level_out$actual))
            
            # % of products greater than 75% accuracy
            accuracy_rf[i + j -1, 6] = (sum(product_level_out$APE <= 25)/nrow(product_level_out))*100
            
            # Volume contribution by these products
            accuracy_rf[i + j -1, 7] = (sum(product_level_out[product_level_out$APE <= 25,"actual"])/sum(product_level_out$actual))*100
          }
          
          # add the parameters of the model
          {
            accuracy_rf[i + j -1, 8] = parameter_grid[i, 1]
            accuracy_rf[i + j -1, 9] = parameter_grid[i, 2]
            accuracy_rf[i + j -1, 10] = parameter_grid[i, 3]
            accuracy_rf[i + j -1, 11] = k
            accuracy_rf[i + j -1, 12] = product_level_out$actual
            accuracy_rf[i + j -1, 13] = product_level_out$pred
            accuracy_rf[i + j -1, 14] = product_level_out$Base_Product_Number
          }
        }
        
        j = j + 4
      }
    }
  }
  
  Accuracy_BPN = accuracy_rf %>%
    group_by(bpn) %>%
    mutate(best = if_else(WAPE == min(WAPE), 1, 0)) %>%
    filter(best == 1) %>%
    mutate(WAPE = round(WAPE, digits = 2),
           actual = round(actual, digits = 2))
  
  Final_accuracy = 100*sum(abs(Accuracy_BPN$actual - Accuracy_BPN$pred))/sum(abs(Accuracy_BPN$actual))
  
  {
    ranger::importance(fitrandom)
    fitrandom
    
    z = data.frame(attributes(fitrandom$variable.importance))
    
    Imp = data.frame(Feature = z, Imp = importance(fitrandom)) %>%
      mutate(Rank = dense_rank(desc(Imp)))
  }
}
