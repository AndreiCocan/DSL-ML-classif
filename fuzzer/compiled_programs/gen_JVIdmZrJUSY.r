library(tidyr)
suppressMessages(library(dplyr))
library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
suppressMessages(library(caret))

OIgtDTG <- read.csv("../datasets/Iris__Species.csv")
OIgtDTG <- OIgtDTG %>% drop_na()

OIgtDTG_X <- subset(OIgtDTG, select = -Species)
OIgtDTG_Y <- as.factor(OIgtDTG$Species)

OIgtDTG_X <- apply(OIgtDTG_X, 2, function(x) x / max(abs(x)))

ZCYMKxw <- read.csv("../datasets/titanic__Survived.csv")
ZCYMKxw <- ZCYMKxw %>% drop_na()

ZCYMKxw_X <- subset(ZCYMKxw, select = -ncol(ZCYMKxw))
ZCYMKxw_Y <- as.factor(ZCYMKxw[,ncol(ZCYMKxw)])
