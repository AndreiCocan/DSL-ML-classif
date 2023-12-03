library(tidyr)
suppressMessages(library(dplyr))
library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
suppressMessages(library(caret))

xXKxpq <- read.csv("../datasets/Iris__Species.csv")
xXKxpq <- xXKxpq %>% drop_na()

columns_to_drop <- c("SepalWidthCm", "PetalWidthCm")
xXKxpq <- xXKxpq %>% select(-any_of(columns_to_drop))

xXKxpq_X <- subset(xXKxpq, select = -Species)
xXKxpq_Y <- as.factor(xXKxpq$Species)

