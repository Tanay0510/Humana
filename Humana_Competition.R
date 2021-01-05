library(h2o)
library(dplyr)

######### Read csv and remove ct ###########

h2o.init(nthreads = -1)

df <- read.csv("/Users/tanaybhardwaj/Desktop/Portfolio/Humana/2020_Competition_Training (1).csv")

######## TRY #########
library(tidyverse) 

df_new <- df %>% select(-ends_with("_ct"))

write.csv(df_new, file= "NoCount.csv" , row.names = FALSE)

############################# H2O ##########################

income_h2o <- h2o.importFile("NoCountPCA.csv")

is.na(income_h2o)


#h2o.na_omit(income_h2o)

birds.pca <- h2o.prcomp(training_frame = income_h2o, transform = "STANDARDIZE",
                        k = 15, pca_method="Power", use_all_factor_levels=TRUE,
                        impute_missing=TRUE)

birds.pca@model$importance

pca <- h2o.prcomp(training_frame = income_h2o, k = 8, transform = "STANDARDIZE")

summary(pca)


###################### MACHINE LEARNING #######################

h2o.str(income_h2o)
y <- "transportation_issues"
x <- setdiff(names(income_h2o), y)

income_h2o[, y] <- as.factor(income_h2o[, y])


income_h2o_split <- h2o.splitFrame(income_h2o, ratios = c(0.6,0.2), seed = 1234)
income_h2o_train <- income_h2o_split[[1]]
income_h2o_valid <- income_h2o_split[[2]]
income_h2o_test <- income_h2o_split[[3]]

########### Logistic Regression with Random Hyperparameter Search #############

predictors <- names(income_h2o_train)[-15]
predictors

hyper_params <- list(alpha = seq(from = 0, to = 1, by = 0.001),
                     lambda = seq(from = 0, to = 1, by = 0.000001)
)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_runtime_secs = 10*3600,
                        max_models = 100,
                        stopping_metric = "AUC", 
                        stopping_tolerance = 0.00001, 
                        stopping_rounds = 5, 
                        seed = 1234
)


models_glm <- h2o.grid(algorithm = "glm", grid_id = "grd_glm", x = predictors, y = "transportation_issues", training_frame = income_h2o_train, validation_frame = income_h2o_valid, nfolds = 0, family = "binomial", hyper_params = hyper_params, search_criteria = search_criteria, stopping_metric = "AUC", stopping_tolerance = 1e-5, stopping_rounds = 5, seed = 1234)


models_glm_sort <- h2o.getGrid(grid_id = "grd_glm", sort_by = "auc", decreasing = TRUE)

models_glm_best <- h2o.getModel(models_glm_sort@model_ids[[1]])

models_glm_best@allparameters

models_glm_best@model$validation_metrics@metrics$AUC


perf_glm_best <- h2o.performance(models_glm_best, income_h2o_valid)
plot(perf_glm_best, type="roc", main="ROC Curve for Best Logistic Regression Model")

h2o.varimp(models_glm_best)


################# Random Forest with Random Hyperparameter Search ######################

hyper_params <- list(ntrees = 10000,  ## early stopping
                     max_depth = 5:15, 
                     min_rows = c(1,5,10,20,50,100),
                     nbins = c(30,100,300),
                     nbins_cats = c(64,256,1024),
                     sample_rate = c(0.7,1),
                     mtries = c(-1,2,6)
)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_runtime_secs = 10*3600,
                        max_models = 100,
                        stopping_metric = "AUC", 
                        stopping_tolerance = 0.00001, 
                        stopping_rounds = 5, 
                        seed = 1234
)

models_rf <- h2o.grid(algorithm = "randomForest", grid_id = "grd_rf", x = predictors, y = "transportation_issues", training_frame = income_h2o_train, validation_frame = income_h2o_valid, nfolds = 0, hyper_params = hyper_params, search_criteria = search_criteria, stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234)

models_rf_sort <- h2o.getGrid(grid_id = "grd_rf", sort_by = "auc", decreasing = TRUE)
models_rf_best <- h2o.getModel(models_rf_sort@model_ids[[1]])

models_rf_best@allparameters

models_rf_best@model$validation_metrics@metrics$AUC


perf_rf_best <- h2o.performance(models_rf_best, income_h2o_valid)
plot(perf_rf_best, type="roc", main="ROC Curve for Best Random Forest Model")

h2o.varimp(models_rf_best)

############# Gradient Boosting Machine with Random Hyperparameter Search ###############


hyper_params <- list(ntrees = 10000,  ## early stopping
                     max_depth = 5:15, 
                     min_rows = c(1,5,10,20,50,100),
                     learn_rate = c(0.001,0.01,0.1),  
                     learn_rate_annealing = c(0.99,0.999,1),
                     sample_rate = c(0.7,1),
                     col_sample_rate = c(0.7,1),
                     nbins = c(30,100,300),
                     nbins_cats = c(64,256,1024)
)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_runtime_secs = 10*3600,
                        max_models = 100,
                        stopping_metric = "AUC", 
                        stopping_tolerance = 0.00001, 
                        stopping_rounds = 5, 
                        seed = 1234
)

models_gbm <- h2o.grid(algorithm = "gbm", grid_id = "grd_gbm", x = predictors, y = "transportation_issues", training_frame = income_h2o_train, validation_frame = income_h2o_valid, nfolds = 5, hyper_params = hyper_params, search_criteria = search_criteria, stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234)

models_gbm_sort <- h2o.getGrid(grid_id = "grd_gbm", sort_by = "auc", decreasing = TRUE)
models_gbm_best <- h2o.getModel(models_gbm_sort@model_ids[[1]])

models_gbm_best@allparameters

models_gbm_best@model$validation_metrics@metrics$AUC

perf_gbm_best <- h2o.performance(models_gbm_best, income_h2o_valid)
plot(perf_gbm_best, type="roc", main="ROC Curve for Best Gradient Boosting Model")

################# Neural Network with Random Hyperparameter Search ###############

hyper_params <- list(activation = c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout"), 
                     hidden = list(c(50, 50, 50, 50), c(200, 200), c(200, 200, 200), c(200, 200, 200, 200)), 
                     epochs = c(50, 100, 200), 
                     l1 = c(0, 0.00001, 0.0001), 
                     l2 = c(0, 0.00001, 0.0001), 
                     adaptive_rate = c(TRUE, FALSE), 
                     rate = c(0, 0.1, 0.005, 0.001), 
                     rate_annealing = c(1e-8, 1e-7, 1e-6), 
                     rho = c(0.9, 0.95, 0.99, 0.999), 
                     epsilon = c(1e-10, 1e-8, 1e-6, 1e-4), 
                     momentum_start = c(0, 0.5),
                     momentum_stable = c(0.99, 0.5, 0), 
                     input_dropout_ratio = c(0, 0.1, 0.2)
)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_runtime_secs = 10*3600,
                        max_models = 100,
                        stopping_metric = "AUC", 
                        stopping_tolerance = 0.00001, 
                        stopping_rounds = 5, 
                        seed = 1234
)

models_dl <- h2o.grid(algorithm = "deeplearning", grid_id = "grd_dl", x = predictors, y = "transportation_issues", training_frame = income_h2o_train, validation_frame = income_h2o_valid, nfolds = 0, hyper_params = hyper_params, search_criteria = search_criteria, stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234)

models_dl_sort <- h2o.getGrid(grid_id = "grd_dl", sort_by = "auc", decreasing = TRUE)
models_dl_best <- h2o.getModel(models_dl_sort@model_ids[[1]])



###################################################################################
splits <- h2o.splitFrame(train.hex, 0.75, seed=1234) 
dl <- h2o.deeplearning(x=1:3, y="transportation_issues",training_frame=splits[[1]], distribution="quantile", quantile_alpha=0.8)

z <- h2o.predict(dl, splits[[2]])


