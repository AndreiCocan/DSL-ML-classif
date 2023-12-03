library(tidyr)
suppressMessages(library(dplyr))
library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
suppressMessages(library(caret))

uWdkwKZ <- read.csv("../datasets/Iris__Species.csv")
uWdkwKZ <- uWdkwKZ %>% drop_na()

uWdkwKZ_X <- subset(uWdkwKZ, select = -Species)
uWdkwKZ_Y <- as.factor(uWdkwKZ$Species)

uWdkwKZ_X <- apply(uWdkwKZ_X, 2, function(x) x / max(abs(x)))

LkwzMD <- read.csv("../datasets/Iris__Species.csv")
LkwzMD <- LkwzMD %>% drop_na()

columns_to_drop <- c("PetalWidthCm")
LkwzMD <- LkwzMD %>% select(-any_of(columns_to_drop))

LkwzMD_X <- subset(LkwzMD, select = -ncol(LkwzMD))
LkwzMD_Y <- as.factor(LkwzMD[,ncol(LkwzMD)])
train_index <- sample(1:nrow(LkwzMD), (1-0.7) * nrow(LkwzMD))
LkwzMD_X_train <- LkwzMD_X[train_index, ]
LkwzMD_X_test <- LkwzMD_X[-train_index, ]
LkwzMD_Y_train <- LkwzMD_Y[train_index]
LkwzMD_Y_test <- LkwzMD_Y[-train_index]

fLWrcXw <- nnet(LkwzMD_Y_train ~ ., LkwzMD_X_train, size = 8)

pred <- predict(fLWrcXw, LkwzMD_X_test, type = "class")
pred <- factor(pred, levels=levels(LkwzMD_Y_test))
fLWrcXw_accuracy <- confusionMatrix(pred, LkwzMD_Y_test)$overall["Accuracy"]
print(paste("fLWrcXw accuracy:", fLWrcXw_accuracy))

train_index <- sample(1:nrow(uWdkwKZ), (1-0.323319472294745) * nrow(uWdkwKZ))
uWdkwKZ_X_train <- uWdkwKZ_X[train_index, ]
uWdkwKZ_X_test <- uWdkwKZ_X[-train_index, ]
uWdkwKZ_Y_train <- uWdkwKZ_Y[train_index]
uWdkwKZ_Y_test <- uWdkwKZ_Y[-train_index]

fLWrcXw <- nnet(uWdkwKZ_Y_train ~ ., uWdkwKZ_X_train, size = 8)

pred <- predict(fLWrcXw, uWdkwKZ_X_test, type = "class")
pred <- factor(pred, levels=levels(uWdkwKZ_Y_test))
fLWrcXw_accuracy <- confusionMatrix(pred, uWdkwKZ_Y_test)$overall["Accuracy"]
print(paste("fLWrcXw accuracy:", fLWrcXw_accuracy))

