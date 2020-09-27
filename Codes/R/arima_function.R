# Function for arima

arima_function = function(ads,i)
{
  # Storing start time of loop for a combination and loading package
  start_loop_time = Sys.time()
  library(forecast)

  # Initialization of variables and identifiers
  {
    div_nm <- as.character(ads[1, c("DIV_NM")])
    seg <- as.character(ads[1, c("SEGMENT")])
    class <- as.character(ads[1, c("CLASS")])

    log_transformed <- 0
    only_arima <- 0

    model_type <- ""
    cnt_insample_na <- 0
    a <- 0
    b <- 0
  }

  # Creating the training/testing dataset, xreg file and basic data treatment of regressors before modelling
  {
    #Getting Pre period data
    filtered_data <- subset(ads,DIV_NM == div_nm & as.character(SEGMENT) == seg
                            & as.character(CLASS) == class & as.integer(FISC_YR_WK) <= current_week)

    #Sorting data based on dates
    filtered_data_sorted <- filtered_data[order(filtered_data$date_st),]

    #Getting time series data
    time_series_data_1comb <- data.frame(filtered_data_sorted$CASES)

    #Adding external regressors
    xreg <- filtered_data_sorted[which(colnames(filtered_data_sorted) == 'newyear'):which(colnames(filtered_data_sorted) == 'mk')]

    #Removing columns that have only 0s
    col_sum0 <- data.frame(colSums(xreg) == 0)
    colnames(col_sum0) <- c("t1")
    remove_col_sum0 <- subset(col_sum0,t1 == TRUE )

    if (nrow(remove_col_sum0) != 0) {
      xreg1 <- xreg[,-which(colSums(xreg) == 0)]
    } else {xreg1 <- xreg}

    #Getting Post period data
    filtered_data_post <- subset(ads,DIV_NM == div_nm & as.character(SEGMENT) == seg
                                 & as.character(CLASS) == class & as.integer(FISC_YR_WK) > current_week)

    #Sorting post period data based on dates
    filtered_data_sorted_post <- filtered_data_post[order(filtered_data_post$date_st),]


    #Adding external regressors
    xreg_post <- filtered_data_sorted_post[which(colnames(filtered_data_sorted) == 'newyear'):which(colnames(filtered_data_sorted) == 'mk')]

    #Removing the extra regressors which were not present in train data
    if (nrow(remove_col_sum0) != 0) { xreg_post <- xreg_post[,-which(colnames(xreg_post) %in% rownames(remove_col_sum0))]
    }
  }

  #Creating time series model and finding lambda for box cox
  seasonal_ts <- ts(time_series_data_1comb,frequency = 52)
  lambda <- BoxCox.lambda(seasonal_ts)

  ##################################### Creating models from scratch #############
  if (recreate_models  == 1) {
    #Running auto arima to get non-seasonal pdq and seasonal PDQ. If Avg cases < 100 then Running without external regressors to save time
    if (mean(time_series_data_1comb$filtered_data_sorted.CASES) <= 100) {
      a <- 1
      fit_new <- try(auto.arima(seasonal_ts,lambda = lambda, stepwise = TRUE, seasonal = F))
      only_arima <- 1
      model_type <- "ARIMA <100 Cases"
    } else if (lambda > -0.1 & lambda < 0.1) {
      b <- 1
      fit_new <- try(auto.arima(log(seasonal_ts), xreg = xreg1, stepwise = TRUE, seasonal = F))
      log_transformed <- 1
      model_type <- "Log transformed >-0.1 & <0.1 ARIMAX"
    } else {
      fit_new <- try(auto.arima(seasonal_ts,lambda = lambda,xreg = xreg1 , stepwise = TRUE, seasonal = F))
      model_type <- "ARIMAX - MLE"
    }

    #Log transform if error is thrown
    if (inherits(fit_new, "try-error") & a != 1 & b != 1 ) {
      fit_new <- try(auto.arima(log(seasonal_ts), xreg = xreg1, stepwise = TRUE, seasonal = F))
      log_transformed <- 1
      only_arima <- 0
      model_type <- "Log transformed ARIMAX"
    }

    #Running only Arima
    if (inherits(fit_new, "try-error") & a != 1 & b != 1) {
      fit_new <- try(auto.arima(seasonal_ts,lambda = lambda,stepwise = TRUE, seasonal = F))
      log_transformed <- 0
      only_arima <- 1
      model_type <- "ARIMA"
    }

    #Simply Running AR model if nothing works out
    if (inherits(fit_new, "try-error")) {
      seasonal_ts <- ts(time_series_data_1comb,frequency = 52)
      fit_new <- try(Arima(seasonal_ts, order = c(1,0,0),seasonal = c(0,0,0)))
      only_arima <- 1
      log_transformed <- 0
      model_type <- "AR 1-0-0"
    }
  }
  ################################################################################

  ###################################### Re run With Re estimation ###############
  if (recreate_models  == 0) {

    order <- as.matrix(data.frame(all_combinations[i,c(5:10)]))

    #Seasonal PDQ to 0 if NA
    order[is.na(order) == T] <- 0

    # Re-estimation of model coefficients
    if (mean(time_series_data_1comb$filtered_data_sorted.CASES) <= 100) {
      a <- 1
      fit_new <- try(Arima(seasonal_ts, order = order[1:3], seasonal = order[4:6],lambda = lambda))
      only_arima <- 1
      model_type <- "ARIMA <100 Cases"
    } else if (lambda > -0.1 & lambda < 0.1) {
      b <- 1
      fit_new <- try(Arima(log(seasonal_ts), order = order[1:3], seasonal = order[4:6], xreg = xreg1) )
      log_transformed <- 1
      model_type <- "Log transformed >-0.1 & <0.1 ARIMAX"
    } else {
      fit_new <- try(Arima(seasonal_ts, order = order[1:3], seasonal = order[4:6],xreg = xreg1, lambda = lambda) )
      model_type <- "ARIMAX - MLE"
    }

    # Re running the models obtained with re-estimation using different methods if the Maximum likely hood method is not able to converge
    if (inherits(fit_new, "try-error") & a != 1 & b != 1 ) {fit_new <- try(Arima(seasonal_ts,method = "CSS", order = order[1:3], seasonal = order[4:6], xreg = xreg1, lambda = lambda))
    model_type <- "ARIMAX - CSS"
    only_arima <- 0
    log_transformed <- 0
    }

    #Trying log transform
    if (inherits(fit_new, "try-error") & a != 1 & b != 1) {
      fit_new <- try(Arima(log(seasonal_ts), method = "CSS", order = order[1:3], seasonal = order[4:6], xreg = xreg1))
      model_type <- "Log transformed ARIMAX"
      log_transformed <- 1
      only_arima <- 0
    }

    #Running only Arima
    if (inherits(fit_new, "try-error") & a != 1) {
      seasonal_ts <- ts(time_series_data_1comb,frequency = 52)
      fit_new <- try(Arima(seasonal_ts, order = order[1:3],seasonal = order[4:6]))
      only_arima <- 1
      log_transformed <- 0
      model_type <- "ARIMA"
    }

    #Simply Running AR model if nothing works out
    if (inherits(fit_new, "try-error")) {
      seasonal_ts <- ts(time_series_data_1comb,frequency = 52)
      fit_new <- try(Arima(seasonal_ts, order = c(1,0,0),seasonal = c(0,0,0)))
      only_arima <- 1
      model_type <- "AR 1-0-0"
    }
  }
  ################################################################################

  # store pdq values from fit_new
  {
    pdq = arimaorder(fit_new)
  }


  #Forecast results
  if (log_transformed == 1) {
    forecast_op <- data.frame(forecast(fit_new,xreg = xreg_post, row.names = F))
    forecast_op <- exp(forecast_op)
    log_transformed <- 0
  } else if (only_arima == 1) {
    forecast_op <- data.frame(forecast(fit_new,h = nrow(xreg_post), row.names = F))
    only_arima <- 0
  } else {forecast_op <- data.frame(forecast(fit_new,xreg = xreg_post, row.names = F))}


  #Getting the records
  forecast_final_base_comb <- subset(filtered_data_sorted_post, select = c(DIV_NBR,DIV_NM,CLASS,SEGMENT,date_st,CLNDR_WK_STRT_DT,YEAR,FISC_YR_WK,CASES))

  #Adding forecast to respective levels
  assign(paste0("forecast_final_base_comb_", i), cbind(forecast_final_base_comb, forecast_op))

  #################### Model Fit #################################33
  #Getting actual for previous 52 weeks
  actual_52 <- subset(filtered_data_sorted, select = c(DIV_NBR,DIV_NM,CLASS,SEGMENT,YEAR,CASES, date_st))

  #Getting residuals for Arima
  fitted <- data.frame(fitt = fitted(fit_new))
  colnames(fitted) <- c('fitt')

  actual_52 <- cbind(actual_52,fitted)
  actual_52$accuracy <- (1 - abs((actual_52$CASES - actual_52$fitt)/actual_52$CASES))
  actual_52$mape <- abs((actual_52$CASES - actual_52$fitt)/actual_52$CASES)
  actual_52$key <- 1:nrow(actual_52)

  #Filtering for most resent 52 weeks
  actual_52_fin <- subset(actual_52, actual_52$key >= max(actual_52$key) - 51)

  #Treating exceptionally high and NA values as dashboard outputs are biased
  cnt_insample_na <- nrow(actual_52_fin[which(is.na(actual_52_fin$fitt)),])
  actual_52_fin$fitt[is.na(actual_52_fin$fitt)] <- 0
  actual_52_fin$accuracy[is.na(actual_52_fin$accuracy)] <- 0
  actual_52_fin$mape[is.na(actual_52_fin$mape)] <- 0

  actual_52_fin$fitt[!is.finite(actual_52_fin$fitt)] <- 0
  actual_52_fin$accuracy[!is.finite(actual_52_fin$accuracy)] <- 0


  #Capping low accuracies to 0 as this means that the forecast is coming thrice the actual meaning 0 accuracy
  actual_52_fin$accuracy <- ifelse(actual_52_fin$accuracy > 1 | actual_52_fin$accuracy < -1, 0,
                                   actual_52_fin$accuracy)

  #Appending in sample values
  assign(paste0("forecast_pre_52_", i), actual_52_fin)

  #Getting the models for future runs
  all_combinations[i,"p"] <- pdq[1]
  all_combinations[i,"d"] <- pdq[2]
  all_combinations[i,"q"] <- pdq[3]
  all_combinations[i,"p_s"] <- pdq[4]
  all_combinations[i,"d_s"] <- pdq[5]
  all_combinations[i,"q_s"] <- pdq[6]
  all_combinations[i,"aic"] <- fit_new$aic
  all_combinations[i,"aicc"] <- fit_new$aicc
  all_combinations[i,"bic"] <- fit_new$bic
  all_combinations[i,"lambda"] <- lambda
  all_combinations[i,"model_type"] <- model_type
  all_combinations[i,"NA_counts"] <- cnt_insample_na
  all_combinations[i,"average_cases"] <- mean(time_series_data_1comb$filtered_data_sorted.CASES)
  all_combinations[i,"exec_time"] <- difftime(Sys.time(), start_loop_time, units = 'mins')

  a <- 0
  b <- 0
  log_transformed <- 0
  only_arima <- 0

  assign(paste0("all_combinations_", i), all_combinations)

  #################################### Saving the outputs for parallel ###########
  {
    assign(paste0("pdq_",i), arimaorder(fit_new))
    #assign(paste0("model_fits_new_",i), fit_new)

    file_names = c(paste0("pdq_", i), paste0("forecast_final_base_comb_", i), paste0("forecast_pre_52_", i), paste0("all_combinations_", i))
    save(list = file_names, file = paste0("saved_variables_", i, ".rda"))
    #save(list = file_names, file = paste0("/home/ubuntu/progs/shared_folder/USF_SF/saved_variables_", i, ".rda"))
  }

  end_loop_time = Sys.time()
  loop_time = end_loop_time - start_loop_time
  saveRDS(loop_time, paste0("loop_time_", i, ".rds"))
  #saveRDS(loop_time, paste0("/home/ubuntu/progs/shared_folder/USF_SF/loop_time_", i, ".rds"))

  cat(loop_time)
}
