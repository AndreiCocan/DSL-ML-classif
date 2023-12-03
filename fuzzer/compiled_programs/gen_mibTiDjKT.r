library(tidyr)
suppressMessages(library(dplyr))
library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
suppressMessages(library(caret))

rxT <- read.csv("../datasets/titanic__Survived.csv")
rxT <- rxT %>% drop_na()

rxT_X <- subset(rxT, select = -ncol(rxT))
rxT_Y <- as.factor(rxT[,ncol(rxT)])
rxT_X <- scale(rxT_X)

train_index <- sample(1:nrow(rxT), (1-0.6206397968364019) * nrow(rxT))
rxT_X_train <- rxT_X[train_index, ]
rxT_X_test <- rxT_X[-train_index, ]
rxT_Y_train <- rxT_Y[train_index]
rxT_Y_test <- rxT_Y[-train_index]

yVymq <- knn(train = rxT_X_train, test = rxT_X_test, cl = rxT_Y_train, k = 10)

train_index <- sample(1:nrow(rxT), (1-0.7) * nrow(rxT))
rxT_X_train <- rxT_X[train_index, ]
rxT_X_test <- rxT_X[-train_index, ]
rxT_Y_train <- rxT_Y[train_index]
rxT_Y_test <- rxT_Y[-train_index]

YPwdvlcJ <- svm(rxT_X_train, rxT_Y_traincost = 0.7552881731521942)

