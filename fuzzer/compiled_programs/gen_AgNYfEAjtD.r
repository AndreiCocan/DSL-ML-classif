library(tidyr)
suppressMessages(library(dplyr))
library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
suppressMessages(library(caret))

iHdl <- read.csv("../datasets/Iris__Species.csv")
iHdl <- iHdl %>% drop_na()

columns_to_drop <- c("PetalWidthCm", "PetalLengthCm")
iHdl <- iHdl %>% select(-any_of(columns_to_drop))

iHdl_X <- subset(iHdl, select = -Species)
iHdl_Y <- as.factor(iHdl$Species)

iHdl_X <- apply(iHdl_X, 2, function(x) x / max(abs(x)))

train_index <- sample(1:nrow(iHdl), (1-0.7187335177509611) * nrow(iHdl))
iHdl_X_train <- iHdl_X[train_index, ]
iHdl_X_test <- iHdl_X[-train_index, ]
iHdl_Y_train <- iHdl_Y[train_index]
iHdl_Y_test <- iHdl_Y[-train_index]

DqZQI <- nnet(iHdl_Y_train ~ ., iHdl_X_train, size = 8)

train_index <- sample(1:nrow(iHdl), (1-0.7) * nrow(iHdl))
iHdl_X_train <- iHdl_X[train_index, ]
iHdl_X_test <- iHdl_X[-train_index, ]
iHdl_Y_train <- iHdl_Y[train_index]
iHdl_Y_test <- iHdl_Y[-train_index]

hDBcF <- knn(train = iHdl_X_train, test = iHdl_X_test, cl = iHdl_Y_train, k = 9)

hDBcF_accuracy <- confusionMatrix(hDBcF, iHdl_Y_test)$overall["Accuracy"]
print(paste("hDBcF accuracy:", hDBcF_accuracy))

train_index <- sample(1:nrow(iHdl), (1-0.7) * nrow(iHdl))
iHdl_X_train <- iHdl_X[train_index, ]
iHdl_X_test <- iHdl_X[-train_index, ]
iHdl_Y_train <- iHdl_Y[train_index]
iHdl_Y_test <- iHdl_Y[-train_index]

XZq <- rpart(iHdl_Y_train ~ ., iHdl_X_train, method = "class")

pred <- predict(XZq, iHdl_X_test, type = "class")
pred <- factor(pred, levels=levels(iHdl_Y_test))
XZq_accuracy <- confusionMatrix(pred, iHdl_Y_test)$overall["Accuracy"]
print(paste("XZq accuracy:", XZq_accuracy))

train_index <- sample(1:nrow(iHdl), (1-0.26410847941225735) * nrow(iHdl))
iHdl_X_train <- iHdl_X[train_index, ]
iHdl_X_test <- iHdl_X[-train_index, ]
iHdl_Y_train <- iHdl_Y[train_index]
iHdl_Y_test <- iHdl_Y[-train_index]

XZq <- rpart(iHdl_Y_train ~ ., iHdl_X_train, method = "class")

