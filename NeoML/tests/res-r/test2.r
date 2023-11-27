library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
library(caret)

myData <- read.csv("tests/datasets/random_dataset.csv")

myData_X <- subset(myData, select = -Class)
myData_Y <- as.factor(myData$Class)

train_index <- sample(1:nrow(myData), (1-0.2) * nrow(myData))
myData_X_train <- myData_X[train_index, ]
myData_X_test <- myData_X[-train_index, ]
myData_Y_train <- myData_Y[train_index]
myData_Y_test <- myData_Y[-train_index]

mySecondModel <- knn(train = myData_X_train, test = myData_X_test, cl = myData_Y_train, k = 8)
# Not possible to use distance-based weights in R


mySecondModel_accuracy <- confusionMatrix(mySecondModel, myData_Y_test)$overall["Accuracy"]
print(paste("mySecondModel accuracy:", mySecondModel_accuracy))
