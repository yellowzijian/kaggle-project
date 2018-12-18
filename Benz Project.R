#test#

library(caret)
library(lattice)
library(readr)
library(tidyverse)
library(h2o)
h2o.init(nthreads=-1)
#load Data
train <- fread("~/Desktop/Benz Project/train.csv", stringsAsFactors = T)
test <- fread("~/Desktop/Benz Project/test.csv", stringsAsFactors = T) 

train <- train %>% mutate(set = "train")

# Remove outlier
train <- train %>% filter(y < 220)

test<- test %>% mutate(set = "test")

# Concatenate
final <- train %>% bind_rows(test)

# Identify constant cols
constant_cols_train <- train %>% summarise_all(n_distinct) %>% gather(key, value) %>% filter(value == 1) %>% select(key) %>% unlist()
constant_cols_test <- test %>% summarise_all(n_distinct) %>% gather(key, value) %>% filter(value == 1) %>% select(key) %>% unlist()
keep_cols <- setdiff(names(train), c(constant_cols_train, constant_cols_test))

# Remove them
final <- final %>% select(one_of(keep_cols), set, ID)

# Convert to factor
data <- final
variables = colnames(data)

for (f in variables){
  if( (class(data[[f]]) == "character") || (class(data[[f]]) == "factor"))
  {
    levels = unique(data[[f]])
    data[[f]] = factor(data[[f]], level = levels)
  }
}

# split groups of categorical and binary features
categorical_vars = paste0("X", c(0,1,2,3,4,5,6,8))
categorical_df <- data %>% select(one_of(categorical_vars))

library(caret)

# perform one-hot encoding 
dmy <- dummyVars(~., data = categorical_df)
ohe_features <- data.frame(predict(dmy, newdata = categorical_df))

df_all <- cbind(data, ohe_features)

train <- df_all[df_all$set=="train",]
test <- df_all[df_all$set=="test",]

train$set<-NULL
test$set<-NULL
test$y<-NULL

trainHex <- as.h2o(train)
testHex <- as.h2o(test)

# define the features
features<-colnames(trainHex)[!(colnames(trainHex) %in% c("ID","y"))]

hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(50,50),c(75,75),c(100,100),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.03,0.05),
  rate=c(0.01,0.02,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 360, max_models = 100, seed=1234567, stopping_rounds=5, stopping_tolerance=1e-2)
dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=trainHex,
  x=features, 
  y="y",
  epochs=1,
  nfolds=5,
  stopping_metric="RMSE",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)                                
grid <- h2o.getGrid("dl_grid_random",sort_by="mse",decreasing=FALSE)
grid

grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest logloss
best_model

b=as.matrix(h2o.predict(best_model,testHex))
submission = data.table(ID=test$ID, y=b)
colnames(submission)<-c("ID", "y")
write.table(submission, "mercedes_submission_dl_v3.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)


# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

# XGBoost Hyperparamters - see list of parameters to tune here:
xgb_grid <- h2o.grid(algorithm = "xgboost",
                     grid_id = "xgb_grid_gaussian",
                     x = features,
                     y = "y",
                     training_frame = trainHex,
                     ntrees = 250, #change, or add to grid search list
                     seed = 1,
                     nfolds = 5,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

sortedGrid <- h2o.getGrid("xgb_grid_gaussian", sort_by = "mse", decreasing = FALSE)  
xgb_model <- h2o.getModel(sortedGrid@model_ids[[1]])
# Train a stacked ensemble of the 3 XGB grid models
ensemble <- h2o.stackedEnsemble(x = features,
                                y = "y",
                                training_frame = trainHex,
                                model_id = "ensemble_xgb_grid_gaussian",
                                base_models = xgb_grid@model_ids)
gbm <- h2o.gbm( y = "y", x=features, training_frame = trainHex,nfolds = 5, model_id = "gbm", ntrees = 65, max_depth = 2, learn_rate = 0.2, learn_rate_annealing = 0.99)
kable(gbm@model$cross_validation_metrics_summary %>% as.tibble() %>% rownames_to_column())




#Magic Feature
#We use the feature X0:
#First, we need the average of "y" for each "X0":

meanX0<-aggregate(train_prep$y, by=list(train_prep$X0), mean)
colnames(meanX0)<-c("X0", "meanX0")
train_final<-merge(train_prep, meanX0, by="X0")

#Same merge for test:
testmean<-test_prep[, c("ID", "X0")]
test_final<-merge(test_prep, meanX0, by="X0", all.x=T)
#Replace each "NA" by the average:
test_final[is.na(test_final)]<-100.97

#DNN
library(readr)
library(caret)
library(Metrics) # for mae() function
library(data.table)

# initialize h2o
trainHex = as.h2o(train_final)
testHex = as.h2o(test_final)

# define the features
features<-colnames(trainHex)[!(colnames(trainHex) %in% c("ID","y"))]

### Train & Cross-validate a GBM
TARGET<-"y"
SEED <- 12345

# Split data for machine learning
splits<-h2o.splitFrame(trainHex, 0.9, destination_frames = c("trainSplit","validSplit"), seed = SEED)
trainSplit <- splits[[1]]
validSplit <- splits[[2]]

## Hyper-Parameter Search

## Construct a large Cartesian hyper-parameter space
ntrees_opts <- c(10000) ## early stopping will stop earlier
max_depth_opts <- seq(1,20)
min_rows_opts <- c(1,5,10,20,50,100)
learn_rate_opts <- seq(0.001,0.01,0.001)
sample_rate_opts <- seq(0.3,1,0.05)
col_sample_rate_opts <- seq(0.3,1,0.05)
col_sample_rate_per_tree_opts = seq(0.3,1,0.05)
nbins_cats_opts = seq(100,10000,100) ## no categorical features in this dataset

hyper_params = list( ntrees = ntrees_opts,
                     max_depth = max_depth_opts,
                     min_rows = min_rows_opts,
                     learn_rate = learn_rate_opts,
                     sample_rate = sample_rate_opts,
                     col_sample_rate = col_sample_rate_opts,
                     col_sample_rate_per_tree = col_sample_rate_per_tree_opts,
                     nbins_cats = nbins_cats_opts
)

## Search a random subset of these hyper-parmameters (max runtime and max models are enforced, and the search will stop after we don't improve much over the best 5 random models)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 600, max_models = 100, stopping_metric = "AUTO", stopping_tolerance = 0.00001, stopping_rounds = 5, seed = SEED)

gbm.grid <- h2o.grid("gbm",
                     grid_id = "mygrid1",
                     x = features,
                     y = TARGET,
                     
                     # faster to use a 80/20 split
                     training_frame = trainHex,
                     nfolds = 5,
                     
                     # alternatively, use N-fold cross-validation
                     #training_frame = train,
                     #nfolds = 5,
                     
                     distribution="gaussian", ## best for MSE loss, but can try other distributions ("laplace", "quantile")
                     
                     ## stop as soon as mse doesn't improve by more than 0.1% on the validation set,
                     ## for 2 consecutive scoring events
                     stopping_rounds = 2,
                     stopping_tolerance = 1e-3,
                     stopping_metric = "MSE",
                     
                     score_tree_interval = 100, ## how often to score (affects early stopping)
                     seed = SEED, ## seed to control the sampling of the Cartesian hyper-parameter space
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

gbm.sorted.grid <- h2o.getGrid(grid_id = "mygrid1", sort_by = "r2", decreasing = TRUE)
print(gbm.sorted.grid)

best_model_gbm <- h2o.getModel(gbm.sorted.grid@model_ids[[1]])
summary(best_model_gbm)

scoring_history <- as.data.frame(best_model_gbm@model$scoring_history)
plot(scoring_history$number_of_trees, scoring_history$training_MSE, type="p") #training mse
points(scoring_history$number_of_trees, scoring_history$validation_MSE, type="l") #validation mse

## get the actual number of trees
ntrees <- best_model_gbm@model$model_summary$number_of_trees
print(ntrees)

# Train & Cross-validate a (shallow) XGB-GBM
my_xgb1 <- h2o.xgboost(x = features,
                       y = TARGET,
                       training_frame = trainHex,
                       ntrees = 50,
                       max_depth = 8,
                       min_rows = 1,
                       learn_rate = 0.2,
                       nfolds = 5,
                       fold_assignment = "Modulo",
                       keep_cross_validation_predictions = TRUE,
                       seed = SEED)
my_xgb2 <- h2o.xgboost(x = features,
                       y = TARGET,
                       training_frame = trainHex,
                       ntrees = 100,
                       max_depth = 8,
                       min_rows = 1,
                       learn_rate = 0.1,
                       sample_rate = 0.7,
                       col_sample_rate = 0.9,
                       nfolds = 5,
                       fold_assignment = "Modulo",
                       keep_cross_validation_predictions = TRUE,
                       seed = SEED)










dnn <- h2o.deeplearning(x=features, y=TARGET, training_frame = train)

# define the features
features<-colnames(trainHex)[!(colnames(trainHex) %in% c("ID","y"))]

### Train & Cross-validate a GBM
TARGET<-"y"
SEED <- 1

#### Random Hyper-Parameter Search
hyper_params <- list(
  hidden=list(c(32,32,32),c(64,64)),
  input_dropout_ratio=c(0,0.05),
  rate=c(0.01,0.02),
  rate_annealing=c(1e-8,1e-7,1e-6)
)
hyper_params
grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id="dl_grid", 
  training_frame=train,
  validation_frame=valid, 
  x=features, 
  y=TARGET,
  epochs=10,
  stopping_metric="misclassification",
  stopping_tolerance=1e-2,        ## stop when misclassification does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  momentum_start=0.5,             ## manually tuned momentum
  momentum_stable=0.9, 
  momentum_ramp=1e7, 
  l1=1e-5,
  l2=1e-5,
  activation=c("Rectifier"),
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params=hyper_params
)
grid
## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 360, max_models = 100, seed=SEED, stopping_rounds=5, stopping_tolerance=1e-2)
dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=train,

  x=features, 
  y=TARGET,
  epochs=1,
  stopping_metric="RMSE",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)                                
grid <- h2o.getGrid("dl_grid_random",sort_by="mae",decreasing=FALSE)
grid

grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest logloss
best_model
dl <- h2o.deeplearning(x = features
                       ,y = "y" 
                       ,seed = SEED
                       #,reproducible = TRUE
                       ,hidden = c(100,100,100)
                       ,epochs = 2
                       #,variable_importances = TRUE
                       ,training_frame=trainHex
                       , variable_importances=T
                       #,nfolds = 10
)
summary(dl)
#Warning message:
#  In .h2o.startModelJob(algo, params, h2oRestApiVersion) :
#  Dropping bad and constant columns: [X107, X297, X330, X233, X2ab, X235, X0ae, X2ad, X0ag, X0, X1, X290, X0p, X2, X2u, X3, X4, X293, X2w, X5, X6, X8, X5b, X0av, X5a, X2aj, X0an, X0bb, X93, X289, X268, X347, X5t, X11, X2ax, X5z].

m3 <- h2o.deeplearning(
  model_id="dl_model_tuned", 
  training_frame=trainHex, 
  x=features, 
  y=TARGET, 
  overwrite_with_best_model=F,    ## Return the final model after 10 epochs, even if not the best
  hidden=c(128,128,128),          ## more hidden layers -> more complex interactions
  epochs=10,                      ## to keep it short enough
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  rate=0.01, 
  rate_annealing=2e-6,            
  momentum_start=0.2,             ## manually tuned momentum
  momentum_stable=0.4, 
  momentum_ramp=1e7, 
  l1=1e-5,                        ## add some L1/L2 regularization
  l2=1e-5,
  max_w2=10                       ## helps stability for Rectifier
) 
summary(m3)

# Tuned
hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(50,50),c(75,75),c(100,100),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.03,0.05),
  #rate=c(0.01,0.02,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 360, max_models = 100, seed=1234567, stopping_rounds=5, stopping_tolerance=1e-2)
dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=trainHex,
  x=features, 
  y=TARGET,
  nfolds=5,
  epochs=1,
  stopping_metric="RMSE",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)                                
grid <- h2o.getGrid("dl_grid_random",sort_by="r2",decreasing=TRUE)
grid

grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest logloss
best_model
best_params <- best_model@allparameters
best_params$activation
best_params$hidden
best_params$input_dropout_ratio
best_params$l1
best_params$l2

a=as.matrix(predict(best_model,testHex))
submission = data.table(ID=test_id, y=a)
colnames(submission)<-c("ID", "y")
write.table(submission, "mercedes_submission_dnn_v02.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)



# isolate id and target variables
train_id <- data.frame(ID=as.character(train$ID))
train_labels <- data.frame(y=train$y)
test_id <- data.frame(ID=as.character(test$ID))
train$ID <- NULL
train$y <- NULL
test$ID <- NULL

# combine features from train and test set 
df_all <- rbind(train, test)

# split groups of categorical and binary features
categorical_vars = paste0("X", c(0,1,2,3,4,5,6,8))

categorical_df <- df_all %>% select(one_of(categorical_vars))
binary_df <- df_all %>% select(-one_of(categorical_vars))

library(caret)
library(lattice)
# perform one-hot encoding 
dmy <- dummyVars(~., data = categorical_df)
ohe_features <- data.frame(predict(dmy, newdata = categorical_df))

df_all <- cbind(df_all, ohe_features)
df_all_train <- df_all[1:nrow(train),]
df_all_test <- df_all[(nrow(train)+1):nrow(df_all),]
write_csv(df_all_train, "train_benz.csv")
write_csv(df_all_test, "test_benz.csv")
 
binary_df <- cbind(binary_df, ohe_features)

binary_df_train <- binary_df[1:nrow(train), ]
binary_df_test <- binary_df[(nrow(train)+1):nrow(binary_df),]

# visualize one-hot encoded features 
image(as.matrix(ohe_features), col=c("white", "black"))
n_levels <- apply(categorical_df, 2, function(x){length(unique(x))}) 
n_levels <- n_levels/sum(n_levels)
abline(h=cumsum(n_levels), col="red")
text(0.05, cumsum(n_levels)-.025, names(n_levels), col="red")
abline(v=0.5, col="darkgreen")
text(0.22, 0.025, "Train", col="darkgreen")
text(0.72, 0.025, "Test", col="darkgreen")



#DNN
library(readr)
library(caret)
library(h2o)
library(Metrics) # for mae() function
library(data.table)



# load data
tr <- fread('../input/train.csv', na.strings = "NA")
ts <- fread("../input/test.csv", sep=",", na.strings = "NA")


# initialize h2o
h2o.init(nthreads=-1,max_mem_size='6G')
trainHex = as.h2o(train_final)
testHex = as.h2o(test_final)

# define the features
features<-colnames(train_final)[!(colnames(train_final) %in% c("ID","y"))]

### Train & Cross-validate a GBM
TARGET<-"y"
SEED <- 1
dl <- h2o.deeplearning(x = features
                       ,y = "y" 
                       ,seed = SEED
                       #,reproducible = TRUE
                       ,hidden = c(100,100,100)
                       ,epochs = 2
                       #,variable_importances = TRUE
                       ,training_frame=trainHex
                       , variable_importances=T
                       #,nfolds = 10
)
summary(dl)
#Warning message:
#  In .h2o.startModelJob(algo, params, h2oRestApiVersion) :
#  Dropping bad and constant columns: [X107, X297, X330, X233, X2ab, X235, X0ae, X2ad, X0ag, X0, X1, X290, X0p, X2, X2u, X3, X4, X293, X2w, X5, X6, X8, X5b, X0av, X5a, X2aj, X0an, X0bb, X93, X289, X268, X347, X5t, X11, X2ax, X5z].

m3 <- h2o.deeplearning(
  model_id="dl_model_tuned", 
  training_frame=trainHex, 
  x=features, 
  y=TARGET, 
  overwrite_with_best_model=F,    ## Return the final model after 10 epochs, even if not the best
  hidden=c(128,128,128),          ## more hidden layers -> more complex interactions
  epochs=10,                      ## to keep it short enough
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  rate=0.01, 
  rate_annealing=2e-6,            
  momentum_start=0.2,             ## manually tuned momentum
  momentum_stable=0.4, 
  momentum_ramp=1e7, 
  l1=1e-5,                        ## add some L1/L2 regularization
  l2=1e-5,
  max_w2=10                       ## helps stability for Rectifier
) 
summary(m3)

# Tuned
hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(50,50),c(75,75),c(100,100),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.03,0.05),
  #rate=c(0.01,0.02,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 360, max_models = 100, seed=1234567, stopping_rounds=5, stopping_tolerance=1e-2)
dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=trainHex,
  x=features, 
  y=TARGET,
  nfolds=5,
  epochs=1,
  stopping_metric="RMSE",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)                                
grid <- h2o.getGrid("dl_grid_random",sort_by="r2",decreasing=TRUE)
grid

grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest logloss
best_model
best_params <- best_model@allparameters
best_params$activation
best_params$hidden
best_params$input_dropout_ratio
best_params$l1
best_params$l2

a=as.matrix(predict(best_model,testHex))
submission = data.table(ID=test_id, y=a)
colnames(submission)<-c("ID", "y")
write.table(submission, "mercedes_submission_dnn_v02.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)

# GBM hyperparamters
gbm_params2 <- list(learn_rate = seq(0.01, 0.1, 0.01),
                    max_depth = seq(2, 10, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria2 = list(strategy = "RandomDiscrete", 
                       max_runtime_secs = 600, 
                       max_models = 100, 
                       stopping_metric = "AUTO", 
                       stopping_tolerance = 0.00001, 
                       stopping_rounds = 5, 
                       seed = 1234567)

# Train and validate a grid of GBMs
gbm_grid1 <- h2o.grid("gbm", x = features, y = TARGET,
                      grid_id = "gbm_grid1",
                      training_frame = trainHex,
                      nfolds = 5,
                      ntrees = 100,
                      hyper_params = gbm_params2,
                      search_criteria = search_criteria2)
                           
grid1 <- h2o.getGrid("gbm_grid1",sort_by="r2",decreasing=TRUE)


grid1@summary_table[1,]
best_model1 <- h2o.getModel(grid1@model_ids[[1]]) ## model with lowest logloss
b=as.matrix(predict(best_model1,testHex))
submission = data.table(ID=test_id, y=b)
colnames(submission)<-c("ID", "y")
write.table(submission, "mercedes_submission_gbm_v0.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)


gbm <- h2o.gbm( y = "y", x=features, training_frame = trainHex,nfolds = 5, model_id = "gbm1", ntrees = 250, max_depth = 2, learn_rate = 0.2, learn_rate_annealing = 0.99, seed = 12345)
rf <- h2o.randomForest(y = "y", x=features, training_frame = trainHex, model_id = "rf", nfolds = 5, max_depth = 7, min_rows = 10, ntrees = 250, seed = 12345)
dl <- h2o.deeplearning(x = features
                       ,y = "y" 
                       ,hidden = c(100,100,100)
                       ,seed = 12345
                       ,model_id = "dl"
                       #,reproducible = TRUE
                       ,epochs = 2
                       #,variable_importances = TRUE
                       ,training_frame=trainHex
                       ,nfolds = 5
)
ensemble <- h2o.stackedEnsemble(x = features,
                                y = "y",
                                training_frame = trainHex,
                                model_id = "my_ensemble_binomial",
                                base_models = list(gbm@model_id, rf@model_id))


# Train & Cross-validate a GBM
my_gbm <- h2o.gbm(x = features,
                  y = "y",
                  training_frame = trainHex,
                  ntrees = 250,
                  max_depth = 3,
                  min_rows = 2,
                  learn_rate = 0.2,
                  nfolds = 5,
                  fold_assignment = "Modulo",
                  keep_cross_validation_predictions = TRUE,
                  seed = 1)

# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = features,
                          y = "y",
                          training_frame = trainHex,
                          ntrees = 250,
                          nfolds = 5,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)

# Train & Cross-validate a DNN
my_dl <- h2o.deeplearning(x = features,
                          y = "y",
                          training_frame = trainHex,
                          l1 = 0.001,
                          l2 = 0.001,
                          hidden = c(200, 200, 200),
                          nfolds = 5,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)
my_xgb2 <- h2o.xgboost(x = features,
                       y = "y",
                       training_frame = trainHex,
                       ntrees = 50,
                       max_depth = 8,
                       min_rows = 1,
                       learn_rate = 0.1,
                       sample_rate = 0.7,
                       col_sample_rate = 0.9,
                       nfolds = 5,
                       fold_assignment = "Modulo",
                       keep_cross_validation_predictions = TRUE,
                       seed = 1)
# Train a stacked ensemble using the GBM and RF above
base_models <- list(my_gbm@model_id, my_rf@model_id, my_dl@model_id,  
                    my_xgb2@model_id)
ensemble <- h2o.stackedEnsemble(x = features,
                                y = "y",
                                training_frame = trainHex,
                                model_id = "my_ensemble_v1",
                                base_models = base_models)
# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = testHex)
b=as.matrix(h2o.predict(ensemble,testHex))
submission = data.table(ID=test$ID, y=b)
colnames(submission)<-c("ID", "y")
write.table(submission, "mercedes_submission_ensemble_v3.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)


## Depth 10 is usually plenty of depth for most datasets, but you never know
hyper_params = list( max_depth = c(4,6,8,12,16,20) ) ##faster for larger datasets

grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## full Cartesian hyper-parameter search
  search_criteria = list(strategy = "Cartesian"),
  
  ## which algorithm to run
  algorithm="gbm",
  
  ## identifier for the grid, to later retrieve it
  grid_id="depth_grid",
  
  ## standard model parameters
  x = features, 
  y = "y", 
  training_frame = trainHex, 
  nfolds = 3,
  
  ## more trees is better if the learning rate is small enough 
  ## here, use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            
  
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  
  ## sample 80% of rows per tree
  sample_rate = 0.8,                                                       
  
  ## sample 80% of columns per split
  col_sample_rate = 0.8, 
  
  ## fix a random number generator seed for reproducibility
  seed = 1,                                                             
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5,
  stopping_tolerance = 1e-4,
  stopping_metric = "MSE", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10                                                
)

## by default, display the grid search results sorted by increasing logloss (since this is a classification task)
grid                                                                       

## sort the grid models by decreasing AUC
sortedGrid <- h2o.getGrid("depth_grid", sort_by="mse", decreasing = TRUE)    
sortedGrid

## find the range of max_depth for the top 5 models
topDepths = sortedGrid@summary_table$max_depth[1:5]                       
minDepth = min(as.numeric(topDepths))
maxDepth = max(as.numeric(topDepths))
minDepth
maxDepth

hyper_params = list( 
  ## restrict the search to the range of max_depth established above
  max_depth = seq(minDepth,maxDepth,1),                                      
  
  ## search a large space of row sampling rates per tree
  sample_rate = seq(0.2,1,0.01),                                             
  
  ## search a large space of column sampling rates per split
  col_sample_rate = seq(0.2,1,0.01),                                         
  
  ## search a large space of column sampling rates per tree
  col_sample_rate_per_tree = seq(0.2,1,0.01),                                
  
  ## search a large space of how column sampling per split should change as a function of the depth of the split
  col_sample_rate_change_per_level = seq(0.9,1.1,0.01),                      
  
  ## search a large space of the number of min rows in a terminal node
  min_rows = 2^seq(0,log2(nrow(train))-1,1),                                 
  
  ## search a large space of the number of bins for split-finding for continuous and integer columns
  nbins = 2^seq(4,10,1),                                                     
  
  ## search a large space of the number of bins for split-finding for categorical columns
  nbins_cats = 2^seq(4,12,1),                                                
  
  ## search a few minimum required relative error improvement thresholds for a split to happen
  min_split_improvement = c(0,1e-8,1e-6,1e-4),                               
  
  ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")       
)

search_criteria = list(
  ## Random grid search
  strategy = "RandomDiscrete",      
  
  ## limit the runtime to 60 minutes
  max_runtime_secs = 3600,         
  
  ## build no more than 100 models
  max_models = 100,                  
  
  ## random number generator seed to make sampling of parameter combinations reproducible
  seed = 1,                        
  
  ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
  stopping_rounds = 5,                
  stopping_metric = "MSE",
  stopping_tolerance = 1e-3
)

grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## hyper-parameter search configuration (see above)
  search_criteria = search_criteria,
  
  ## which algorithm to run
  algorithm = "gbm",
  
  ## identifier for the grid, to later retrieve it
  grid_id = "final_grid1", 
  
  ## standard model parameters
  x = features, 
  y = "y", 
  training_frame = trainHex, 
  nfolds=5,
  keep_cross_validation_predictions = TRUE,
  
  ## more trees is better if the learning rate is small enough
  ## use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            
  
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  
  ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
  max_runtime_secs = 3600,                                                 
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "MSE", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,                                                
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1                                                             
)
sortedGrid <- h2o.getGrid("final_grid", sort_by = "MSE", decreasing = FALSE)    
sortedGrid
gbm <- h2o.getModel(sortedGrid@model_ids[[1]])
perf <- h2o.performance(gbm, newdata = testHex)
b=as.matrix(h2o.predict(gbm,testHex))
submission = data.table(ID=test$ID, y=b)
colnames(submission)<-c("ID", "y")
write.table(submission, "mercedes_submission_gbm_v0.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)

ensemble <- h2o.stackedEnsemble(x = features,
                                y = "y",
                                training_frame = trainHex,
                                model_id = "my_ensemble_v3",
                                base_models = grid@model_ids)


ensemble <- h2o.stackedEnsemble(x = features,
                                y = "y",
                                training_frame = trainHex,
                                model_id = "my_ensemble_v1",
                                base_models = my_rf)


##################################################################################################################
# Preston Greene
# 07/04/2017
# Kaggle Zillow Prize competition
# 
# Model stacking and ensembling seems to be the way most Kaggle competitions are won. There are many methods for stacking/ensembling
# in R, including the caretEnsemble package or rolling your own ensembling functions. For those new to ensembling, h2o offers a good ensembling 
# function, has many algorithms that can be ensembled, etc. 
#
# In this script I use h2o to estimate 3 XGBoost models with randomly selected hyperparameters, then train a stacked ensemble
# of these and generate predictions. The performance on the public LB is 0.0666050. 
#
# My intent with this script is to illustrate basic ensembling in h2o. Although I've ensembled
# three XGBoost models, one could use a collection of different algorithms (e.g. 1 random forest,
# 1 GBM, 1 XGBoost, 1 neural net, etc.) and combine those into an ensemble, with or without grid search parameter tuning. 
# There are lots of possibilites, although increasing the size and complexity of models may be limited by your h2o cluster size. 
# Hope this is helpful! Please comment!
##################################################################################################################

library(data.table)
library(h2o)

######################################
# Load data, create training data set
######################################
print("Read in raw data")

properties <- fread("../input/properties_2016.csv", header=TRUE, stringsAsFactors=FALSE, colClasses = list(character = 50))
train      <- fread("../input/train_2016_v2.csv")
training   <- merge(properties, train, by="parcelid",all.y=TRUE)

######################################
# Setup h2o
######################################
h2o.init(nthreads = -1, max_mem_size = "8g")

# Identify predictors and response
x <- names(training)[which(names(training)!="logerror")]
y <- "logerror"

# 1.1 Import Data to h2o
train <- as.h2o(training)
test <- as.h2o(properties)

###################
# Specify a grid of XGBoost parameters to search 
###################

# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

# XGBoost Hyperparamters - see list of parameters to tune here:
eta_opt <- c(0.1,0.01,0.001)
learn_rate_opt <- c(0.01, 0.03)
max_depth_opt <- c(2, 4, 6, 8, 10)
sample_rate_opt <- c(0.5, 0.75, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(eta = eta_opt,
                     learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 3,
                        seed = 1)

xgb_grid <- h2o.grid(algorithm = "xgboost",
                     grid_id = "xgb_grid_gaussian1",
                     x = features,
                     y = "y",
                     training_frame = trainHex,
                     ntrees = 20, #change, or add to grid search list
                     seed = 1,
                     nfolds = 5,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

# Evaluate performance of xgb - not sure why I couldn't get these to work...
# perf_xgb_train <- h2o.performance(xgb_grid)
# perf_xgb_test <- h2o.performance(xgb_grid, newdata = test)
# print("XGB training performance: ")
# print(perf_xgb_train)
# print("XGB test performance: ")
# print(perf_xgb_test)

# Train a stacked ensemble of the 3 XGB grid models
ensemble <- h2o.stackedEnsemble(x = features,
                                y = "y",
                                training_frame = trainHex,
                                model_id = "ensemble_xgb_grid_gaussian1",
                                base_models = xgb_grid@model_ids)
sortedGrid <- h2o.getGrid("xgb_grid_gaussian1", sort_by = "mse", decreasing = FALSE)    
sortedGrid
h2o.predict(ensemble,testHex)


## Depth 10 is usually plenty of depth for most datasets, but you never know
hyper_params = list( max_depth = c(4,6,8,12,16,20) ) ##faster for larger datasets

grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## full Cartesian hyper-parameter search
  search_criteria = list(strategy = "Cartesian"),
  
  ## which algorithm to run
  algorithm="gbm",
  
  ## identifier for the grid, to later retrieve it
  grid_id="depth_grid",
  
  ## standard model parameters
  x = features, 
  y = "y", 
  training_frame = trainHex, 
  nfolds = 3,
  
  ## more trees is better if the learning rate is small enough 
  ## here, use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            
  
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  
  ## sample 80% of rows per tree
  sample_rate = 0.8,                                                       
  
  ## sample 80% of columns per split
  col_sample_rate = 0.8, 
  
  ## fix a random number generator seed for reproducibility
  seed = 1,                                                             
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5,
  stopping_tolerance = 1e-4,
  stopping_metric = "MSE", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10                                                
)

## by default, display the grid search results sorted by increasing logloss (since this is a classification task)
grid                                                                       

## sort the grid models by decreasing AUC
sortedGrid <- h2o.getGrid("depth_grid", sort_by="mse", decreasing = TRUE)    
sortedGrid

## find the range of max_depth for the top 5 models
topDepths = sortedGrid@summary_table$max_depth[1:5]                       
minDepth = min(as.numeric(topDepths))
maxDepth = max(as.numeric(topDepths))
minDepth
maxDepth

hyper_params = list( 
  ## restrict the search to the range of max_depth established above
  max_depth = seq(minDepth,maxDepth,1),                                      
  
  ## search a large space of row sampling rates per tree
  sample_rate = seq(0.2,1,0.01),                                             
  
  ## search a large space of column sampling rates per split
  col_sample_rate = seq(0.2,1,0.01),                                         
  
  ## search a large space of column sampling rates per tree
  col_sample_rate_per_tree = seq(0.2,1,0.01),                                
  
  ## search a large space of how column sampling per split should change as a function of the depth of the split
  col_sample_rate_change_per_level = seq(0.9,1.1,0.01),                      
  
  ## search a large space of the number of min rows in a terminal node
  min_rows = 2^seq(0,log2(nrow(train))-1,1),                                 
  
  ## search a large space of the number of bins for split-finding for continuous and integer columns
  nbins = 2^seq(4,10,1),                                                     
  
  ## search a large space of the number of bins for split-finding for categorical columns
  nbins_cats = 2^seq(4,12,1),                                                
  
  ## search a few minimum required relative error improvement thresholds for a split to happen
  min_split_improvement = c(0,1e-8,1e-6,1e-4),                               
  
  ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")       
)

search_criteria = list(
  ## Random grid search
  strategy = "RandomDiscrete",      
  
  ## limit the runtime to 60 minutes
  max_runtime_secs = 3600,         
  
  ## build no more than 100 models
  max_models = 100,                  
  
  ## random number generator seed to make sampling of parameter combinations reproducible
  seed = 1,                        
  
  ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
  stopping_rounds = 5,                
  stopping_metric = "MSE",
  stopping_tolerance = 1e-3
)

grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## hyper-parameter search configuration (see above)
  search_criteria = search_criteria,
  
  ## which algorithm to run
  algorithm = "gbm",
  
  ## identifier for the grid, to later retrieve it
  grid_id = "final_grid1", 
  
  ## standard model parameters
  x = features, 
  y = "y", 
  training_frame = trainHex, 
  nfolds=5,
  keep_cross_validation_predictions = TRUE,
  
  ## more trees is better if the learning rate is small enough
  ## use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            
  
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  
  ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
  max_runtime_secs = 3600,                                                 
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "MSE", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,                                                
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1                                                             
)
sortedGrid <- h2o.getGrid("final_grid", sort_by = "MSE", decreasing = FALSE)    
sortedGrid
gbm <- h2o.getModel(sortedGrid@model_ids[[1]])
perf <- h2o.performance(gbm, newdata = testHex)
b=as.matrix(h2o.predict(gbm,testHex))
submission = data.table(ID=test$ID, y=b)
colnames(submission)<-c("ID", "y")
write.table(submission, "mercedes_submission_gbm_v0.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)









