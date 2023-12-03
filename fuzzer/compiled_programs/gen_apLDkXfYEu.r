library(tidyr)
suppressMessages(library(dplyr))
library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
suppressMessages(library(caret))

YSkeZKjE <- read.csv("../datasets/random-dataset__Class.csv")
YSkeZKjE <- YSkeZKjE %>% drop_na()

YSkeZKjE_X <- subset(YSkeZKjE, select = -Class)
YSkeZKjE_Y <- as.factor(YSkeZKjE$Class)

preprocess_range <- preProcess(YSkeZKjE_X, method = "range")
YSkeZKjE_X <- predict(preprocess_range, newdata = YSkeZKjE_X)

LpVhOdi <- read.csv("../datasets/Iris__Species.csv")
LpVhOdi <- LpVhOdi %>% drop_na()

LpVhOdi_X <- subset(LpVhOdi, select = -Species)
LpVhOdi_Y <- as.factor(LpVhOdi$Species)

train_index <- sample(1:nrow(YSkeZKjE), (1-0.7) * nrow(YSkeZKjE))
YSkeZKjE_X_train <- YSkeZKjE_X[train_index, ]
YSkeZKjE_X_test <- YSkeZKjE_X[-train_index, ]
YSkeZKjE_Y_train <- YSkeZKjE_Y[train_index]
YSkeZKjE_Y_test <- YSkeZKjE_Y[-train_index]

saVRw <- knn(train = YSkeZKjE_X_train, test = YSkeZKjE_X_test, cl = YSkeZKjE_Y_train)

train_index <- sample(1:nrow(YSkeZKjE), (1-0.7) * nrow(YSkeZKjE))
YSkeZKjE_X_train <- YSkeZKjE_X[train_index, ]
YSkeZKjE_X_test <- YSkeZKjE_X[-train_index, ]
YSkeZKjE_Y_train <- YSkeZKjE_Y[train_index]
YSkeZKjE_Y_test <- YSkeZKjE_Y[-train_index]

saVRw <- knn(train = YSkeZKjE_X_train, test = YSkeZKjE_X_test, cl = YSkeZKjE_Y_train)

train_index <- sample(1:nrow(LpVhOdi), (1-0.7) * nrow(LpVhOdi))
LpVhOdi_X_train <- LpVhOdi_X[train_index, ]
LpVhOdi_X_test <- LpVhOdi_X[-train_index, ]
LpVhOdi_Y_train <- LpVhOdi_Y[train_index]
LpVhOdi_Y_test <- LpVhOdi_Y[-train_index]

saVRw <- knn(train = LpVhOdi_X_train, test = LpVhOdi_X_test, cl = LpVhOdi_Y_train)

train_index <- sample(1:nrow(LpVhOdi), (1-0.7) * nrow(LpVhOdi))
LpVhOdi_X_train <- LpVhOdi_X[train_index, ]
LpVhOdi_X_test <- LpVhOdi_X[-train_index, ]
LpVhOdi_Y_train <- LpVhOdi_Y[train_index]
LpVhOdi_Y_test <- LpVhOdi_Y[-train_index]

HACWt <- nnet(LpVhOdi_Y_train ~ ., LpVhOdi_X_train, size = 8)

