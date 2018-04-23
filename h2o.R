
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
#Start up a 1-node H2O server on your local machine, and allow it to use all CPU cores and up to 2GB of memory:

h2o.init(nthreads = 2, max_mem_size = "4G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

output = fread('output.csv')

train_full.hex = h2o.importFile(path = 'final_train.csv', destination_frame = 'final_train')
test_full.hex = h2o.importFile(path = 'final_test.csv', destination_frame = 'final_test')
names(train_full.hex)

train_full.hex[, 301] = as.factor(train_full.hex[, 301])

splits = h2o.splitFrame(train_full.hex, c(0.6,0.2), seed = 1234)
train  = h2o.assign(splits[[1]], "train.hex") # 60%
valid  = h2o.assign(splits[[2]], "valid.hex") # 20%
test   = h2o.assign(splits[[3]], "test.hex")  # 20%

response = "project_is_approved"
predictors <- setdiff(names(train), response)

hyper_params <- list(
  activation = c("RectifierWithDropout","MaxoutWithDropout"),
  hidden = list(c(20,20),c(30,30,30), c(50, 50)),
  input_dropout_ratio = c(0,0.05),
  l1 = seq(0,1e-4,1e-5),
  l2 = seq(0,1e-4,1e-5)
)
hyper_params
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 360, max_models = 100, seed = 123, stopping_rounds = 5, stopping_tolerance = 1e-2)
dl_random_grid <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = "dl_grid_random",
  training_frame = train,
  validation_frame = valid,
  x = predictors,
  y = response,
  epochs = 1,
  stopping_metric = "logloss",
  stopping_tolerance = 1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds = 2,
  score_validation_samples = 10000, ## downsample validation set for faster scoring
  score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
  max_w2 = 10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria)
grid <- h2o.getGrid("dl_grid_random",sort_by = "auc",decreasing = TRUE)
grid

grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]])
best_model

best_params <- best_model@allparameters
best_params$activation
best_params$hidden
best_params$input_dropout_ratio
best_params$l1
best_params$l2

path <- h2o.saveModel(m_cont,
                      path = "./", force = TRUE)

m_loaded <- h2o.loadModel(path)

dlmodel <- h2o.deeplearning(
  x=predictors,
  y=response,
  training_frame=train,
  hidden=c(10,10),
  epochs=1,
  nfolds=5,
  fold_assignment="Modulo" # can be "AUTO", "Modulo", "Random" or "Stratified"
)

plot(h2o.performance(dlmodel)) ## display ROC curve
#### All done, shutdown H2O
h2o.shutdown(prompt=FALSE)
