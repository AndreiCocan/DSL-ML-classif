library(tidyr)
suppressMessages(library(dplyr))
library(e1071) # SVM implementation
library(class) # KNN implementation
library(rpart) # DecisionTree implementation
library(nnet) # MLP implementation
suppressMessages(library(caret))

DJmokB <- read.csv("../datasets/random-dataset__Class.csv")
DJmokB <- DJmokB %>% drop_na()

DJmokB_X <- subset(DJmokB, select = -ncol(DJmokB))
DJmokB_Y <- as.factor(DJmokB[,ncol(DJmokB)])
DJmokB_X <- apply(DJmokB_X, 2, function(x) x / max(abs(x)))

ISdRk <- read.csv("../datasets/Iris__Species.csv")
ISdRk <- ISdRk %>% drop_na()

columns_to_drop <- c("Id")
ISdRk <- ISdRk %>% select(-any_of(columns_to_drop))

ISdRk_X <- subset(ISdRk, select = -Species)
ISdRk_Y <- as.factor(ISdRk$Species)

aycXnM <- read.csv("../datasets/random-dataset__Class.csv")
aycXnM <- aycXnM %>% drop_na()

columns_to_drop <- c("Id2", "Id1")
aycXnM <- aycXnM %>% select(-any_of(columns_to_drop))

aycXnM_X <- subset(aycXnM, select = -Class)
aycXnM_Y <- as.factor(aycXnM$Class)

yrAzL <- read.csv("../datasets/Iris__Species.csv")
yrAzL <- yrAzL %>% drop_na()

columns_to_drop <- c("PetalWidthCm")
yrAzL <- yrAzL %>% select(-any_of(columns_to_drop))

yrAzL_X <- subset(yrAzL, select = -Species)
yrAzL_Y <- as.factor(yrAzL$Species)

