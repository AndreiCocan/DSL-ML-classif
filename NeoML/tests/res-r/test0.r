library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
library(caret)

myData <- read.csv("tests/datasets/Iris.csv")

myData_X <- subset(myData, select = -Species)
myData_Y <- as.factor(myData$Species)

preprocess_range <- preProcess(myData_X, method = "range")
myData_X <- predict(preprocess_range, newdata = myData_X)

train_index <- sample(1:nrow(myData), (1-0.1) * nrow(myData))
myData_X_train <- myData_X[train_index, ]
myData_X_test <- myData_X[-train_index, ]
myData_Y_train <- myData_Y[train_index]
myData_Y_test <- myData_Y[-train_index]

myFirstModel <- svm(myData_X_train, myData_Y_train, kernel = "linear", cost = 1.0)

pred <- predict(myFirstModel, myData_X_test, type = "class")
pred <- factor(pred, levels=levels(myData_Y_test))
myFirstModel_accuracy <- confusionMatrix(pred, myData_Y_test)$overall["Accuracy"]
print(paste("myFirstModel accuracy:", myFirstModel_accuracy))
