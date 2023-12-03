library(tidyr)
suppressMessages(library(dplyr))
library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
suppressMessages(library(caret))

YxHQ <- read.csv("../datasets/Iris__Species.csv")
YxHQ <- YxHQ %>% drop_na()

columns_to_drop <- c("PetalWidthCm")
YxHQ <- YxHQ %>% select(-any_of(columns_to_drop))

YxHQ_X <- subset(YxHQ, select = -Species)
YxHQ_Y <- as.factor(YxHQ$Species)

train_index <- sample(1:nrow(YxHQ), (1-0.3071165972922678) * nrow(YxHQ))
YxHQ_X_train <- YxHQ_X[train_index, ]
YxHQ_X_test <- YxHQ_X[-train_index, ]
YxHQ_Y_train <- YxHQ_Y[train_index]
YxHQ_Y_test <- YxHQ_Y[-train_index]

hvwvhVfU <- knn(train = YxHQ_X_train, test = YxHQ_X_test, cl = YxHQ_Y_train)

hvwvhVfU_accuracy <- confusionMatrix(hvwvhVfU, YxHQ_Y_test)$overall["Accuracy"]
print(paste("hvwvhVfU accuracy:", hvwvhVfU_accuracy))

train_index <- sample(1:nrow(YxHQ), (1-0.9502723340332941) * nrow(YxHQ))
YxHQ_X_train <- YxHQ_X[train_index, ]
YxHQ_X_test <- YxHQ_X[-train_index, ]
YxHQ_Y_train <- YxHQ_Y[train_index]
YxHQ_Y_test <- YxHQ_Y[-train_index]

hvwvhVfU <- knn(train = YxHQ_X_train, test = YxHQ_X_test, cl = YxHQ_Y_train)

hvwvhVfU_accuracy <- confusionMatrix(hvwvhVfU, YxHQ_Y_test)$overall["Accuracy"]
print(paste("hvwvhVfU accuracy:", hvwvhVfU_accuracy))

train_index <- sample(1:nrow(YxHQ), (1-0.897970409331531) * nrow(YxHQ))
YxHQ_X_train <- YxHQ_X[train_index, ]
YxHQ_X_test <- YxHQ_X[-train_index, ]
YxHQ_Y_train <- YxHQ_Y[train_index]
YxHQ_Y_test <- YxHQ_Y[-train_index]

hvwvhVfU <- knn(train = YxHQ_X_train, test = YxHQ_X_test, cl = YxHQ_Y_train)

hvwvhVfU_accuracy <- confusionMatrix(hvwvhVfU, YxHQ_Y_test)$overall["Accuracy"]
print(paste("hvwvhVfU accuracy:", hvwvhVfU_accuracy))

