library(tidyr)
suppressMessages(library(dplyr))
library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
suppressMessages(library(caret))

fRFTGD <- read.csv("../datasets/titanic__Survived.csv")
fRFTGD <- fRFTGD %>% drop_na()

columns_to_drop <- c("Fare", "Sex")
fRFTGD <- fRFTGD %>% select(-any_of(columns_to_drop))

fRFTGD_X <- subset(fRFTGD, select = -ncol(fRFTGD))
fRFTGD_Y <- as.factor(fRFTGD[,ncol(fRFTGD)])
eGjgETp <- read.csv("../datasets/random-dataset__Class.csv")
eGjgETp <- eGjgETp %>% drop_na()

eGjgETp_X <- subset(eGjgETp, select = -ncol(eGjgETp))
eGjgETp_Y <- as.factor(eGjgETp[,ncol(eGjgETp)])
train_index <- sample(1:nrow(eGjgETp), (1-0.951183882421597) * nrow(eGjgETp))
eGjgETp_X_train <- eGjgETp_X[train_index, ]
eGjgETp_X_test <- eGjgETp_X[-train_index, ]
eGjgETp_Y_train <- eGjgETp_Y[train_index]
eGjgETp_Y_test <- eGjgETp_Y[-train_index]

URQRSAO <- knn(train = eGjgETp_X_train, test = eGjgETp_X_test, cl = eGjgETp_Y_train)

URQRSAO_accuracy <- confusionMatrix(URQRSAO, eGjgETp_Y_test)$overall["Accuracy"]
print(paste("URQRSAO accuracy:", URQRSAO_accuracy))

