#using gradient boosting algorithm

# Install XGBoost
devtools::install_github('dmlc/xgboost',subdir='R-package')

# Packages and dataset
require(devtools)
require(xgboost)
require(methods)
require(data.table)
require(magrittr)
require(ggplot2)
require(DiagrammeR)
require(Ckmeans.1d.dp)
train <- fread('~/Dropbox/Kaggle/Otto/train.csv', header = T, stringsAsFactors = F)
test <- fread('~/Dropbox/Kaggle/Otto/test.csv', header=TRUE, stringsAsFactors = F)


# Train dataset dimensions
dim(train)

# Training content
train[1:6,1:5, with =F]

# Test dataset dimensions
dim(train)

# Test content
test[1:6,1:5, with =F]

# Delete ID column in training dataset
train[, id := NULL]

# Delete ID column in testing dataset
test[, id := NULL]
testMat <- as.matrix(test)

# Check the content of the last column
train[1:6, ncol(train), with  = F]
# Save the name of the last column
nameLastCol <- names(train)[ncol(train)]

# Convert from classes to numbers for XGBoost support
y <- train[, nameLastCol, with = F][[1]] %>% gsub('Class_','',.) %>% {as.integer(.) -1}
# Display the first 5 levels
y[1:5]

# Delete label column or it will be used in prediction
train[, nameLastCol:=NULL, with = F]

# Convert data tables into numeric Matrices, also for XGBoost support
trainMatrix <- train[,lapply(.SD,as.numeric)] %>% as.matrix
testMatrix <- test[,lapply(.SD,as.numeric)] %>% as.matrix

# Train the model
numberOfClasses <- max(y) + 1
numberOfClasses
y

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)

cv.nround <- 5
cv.nfold <- 3

bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround)

nround = 50
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround)

# View the model
model <- xgb.dump(bst, with.stats = T)
model[1:10]

# Feature importance/selection, find the 10 most important features
# Get the feature real names
names <- dimnames(trainMatrix)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])

# Interaction between features
xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2, width = 3000, height = 1600)


# Predict
pred = predict(bst, testMatrix[, -1])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='~/Dropbox/Kaggle/Otto/gradient_boost_benchmark.csv', quote=FALSE,row.names=FALSE)




randomForest <- randomForest(trainData[,c(-1,-95)], as.factor(trainData$target), ntree=25, importance=TRUE)
results[,2:10] <- (predict(randomForest, testData[,-1], type="prob")+0.01)/1.09

gz_out <- gzfile("~/Dropbox/Kaggle/Otto/random_forest_benchmark.csv", "w")
writeChar(write_csv(results, ""), gz_out, eos=NULL)
resultsData <- read.csv("~/Dropbox/Kaggle/Otto/random_forest_benchmark.csv")
close(gz_out)