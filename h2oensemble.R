
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

h2o.init(nthreads = 2, max_mem_size = "4G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

output = fread('output.csv')

train_full.hex = h2o.importFile(path = 'final_train.csv', destination_frame = 'final_train')
test_full.hex = h2o.importFile(path = 'final_test.csv', destination_frame = 'final_test')
names(train_full.hex)

train_full.hex[, 301] = as.factor(train_full.hex[, 301])

# splits = h2o.splitFrame(train_full.hex, c(0.6,0.2), seed = 1234)
# train  = h2o.assign(splits[[1]], "train.hex") # 60%
# valid  = h2o.assign(splits[[2]], "valid.hex") # 20%
# test   = h2o.assign(splits[[3]], "test.hex")  # 20%

response = "project_is_approved"
predictors = setdiff(names(train_full.hex), response)

# define the splits
h2o_splits = h2o.splitFrame(train_full.hex, 0.7, seed = 1234)
h2o_DstTrain  = h2o.assign(h2o_splits[[1]], "train.hex") # 70%
h2o_DstTest  = h2o.assign(h2o_splits[[2]], "test.hex") # 30%

# Number of CV folds (to generate level-one data for stacking)
cvfolds <- 5

get_auc <- function(x) h2o.auc(h2o.performance(h2o.getModel(x), newdata = h2o_DstTest))

# Train & Cross-validate a GBM
{
my_gbm <- h2o.gbm(x = predictors,
                  y = response,
                  training_frame = h2o_DstTrain,
                  distribution = "bernoulli",
                  max_depth = 5,
                  min_rows = 3,
                  learn_rate = 0.15,
                  ntrees = 100,
                  nfolds = cvfolds,
                  fold_assignment = "Modulo",
                  keep_cross_validation_predictions = TRUE,
                  seed = 1,
                  stopping_metric = "AUC",
                  balance_classes = T,
                  max_after_balance_size = 2,
                  sample_rate = 0.7,
                  col_sample_rate = 0.7,
                  calibrate_model = T,
                  verbose = F,
                  calibration_frame = h2o_DstTest,
                  stopping_rounds = 10)
# Measure auc
get_auc(my_gbm@model_id)

h2o.saveModel(my_gbm, path = "./", force = TRUE)
my_gbm <- h2o.loadModel('GBM_model_R_1524547384365_194')
}

# Train & Cross-validate a RF
{
my_rf <- h2o.randomForest(x = predictors,
                          y = response,
                          training_frame = h2o_DstTrain,
                          nfolds = cvfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1,
                          #balance_classes = T,
                          #max_after_balance_size = 2,
                          ntrees = 200,
                          max_depth = 5,
                          stopping_rounds = 10,
                          stopping_metric = "AUC",
                          verbose = F,
                          calibrate_model = T,
                          calibration_frame = h2o_DstTest)
# Measure auc
get_auc(my_rf@model_id)

h2o.saveModel(my_rf, path = "./", force = TRUE)
my_rf <- h2o.loadModel('DRF_model_R_1524547384365_350')
}

# Train & Cross-validate a DNN
{
my_dl <- h2o.deeplearning(x = predictors,
                          y = response,
                          training_frame = h2o_DstTrain,
                          l1 = 0.001,
                          l2 = 0.001,
                          hidden = c(50, 50, 50),
                          nfolds = cvfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1,
                          activation = "MaxoutWithDropout",
                          stopping_rounds = 10,
                          stopping_metric = "AUC",
                          variable_importances = F,
                          balance_classes = T,
                          max_after_balance_size = 2,
                          input_dropout_ratio = 0.5,
                          epochs = 1)
# Measure auc
get_auc(my_dl@model_id)

h2o.saveModel(my_dl, path = "./", force = TRUE)
my_dl <- h2o.loadModel('DeepLearning_model_R_1524547384365_723')
}

# Create a Stacked Ensemble
# To maximize predictive power, will create an H2O Stacked Ensemble from the models we created above and print the performance gain the ensemble has over the best base model.
# Train a stacked ensemble using the H2O and XGBoost models from above
base_models <- list(my_gbm@model_id, my_rf@model_id, my_dl@model_id)

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

# convert to h2o frame
#h2o_FinalTest = as.h2o(DstTest[,ToDropTest])
h2o_FinalTest = as.h2o(toNumeric(DstTest[,ToDropTest]))

# predict with the model
predictFinal <- h2o.predict(ensemble, h2o_FinalTest)

# convert H2O format into data frame and save as csv
predictFinal.df <- as.data.frame(predictFinal)

# create a csv file for submittion
Result <- data.frame(id = DstTest$id, project_is_approved = predictFinal.df$X0)
head(Result,n=5L)
# write the submition file
write.csv(Result,file = "Result.csv",row.names = FALSE)

# shut down virtual H2O cluster
h2o.shutdown(prompt = FALSE)
