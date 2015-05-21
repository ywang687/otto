#simple product classification using random forest

testData <- read.csv("~/Dropbox/Kaggle/Otto/test.csv")
trainData <- read.csv("~/Dropbox/Kaggle/Otto/train.csv")
sampleResultsData <- read.csv("~/Dropbox/Kaggle/Otto/sampleSubmission.csv")
View(testData)

library(ggplot2)
library(randomForest)
library(readr)
library(mclust)

set.seed(6)

results <- data.frame(id=testData$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)

randomForest <- randomForest(trainData[,c(-1,-95)], as.factor(trainData$target), ntree=25, importance=TRUE)
results[,2:10] <- (predict(randomForest, testData[,-1], type="prob")+0.01)/1.09

gz_out <- gzfile("~/Dropbox/Kaggle/Otto/random_forest_benchmark.csv", "w")
writeChar(write_csv(results, ""), gz_out, eos=NULL)
resultsData <- read.csv("~/Dropbox/Kaggle/Otto/random_forest_benchmark.csv")
close(gz_out)

imp <- importance(randomForest, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("Importance") +
  ylab("") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

ggsave(file = "feature_importance.png", p, path = "~/Dropbox/Kaggle/Otto", height=20, width=8, units="in")
