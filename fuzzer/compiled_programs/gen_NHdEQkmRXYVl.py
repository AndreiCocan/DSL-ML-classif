from sklearn import *
import pandas as pd

YxHQ = pd.read_csv("../datasets/Iris__Species.csv")
YxHQ = YxHQ.dropna()

YxHQ_Y = YxHQ["Species"]
YxHQ = YxHQ.drop(columns=["PetalWidthCm", "Species"])

hvwvhVfU = neighbors.KNeighborsClassifier()

YxHQ_X_train, YxHQ_X_test, YxHQ_Y_train, YxHQ_Y_test = model_selection.train_test_split(YxHQ, YxHQ_Y, test_size = 0.3071165972922678)
hvwvhVfU.fit(YxHQ_X_train, YxHQ_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(YxHQ_Y_test, hvwvhVfU.predict(YxHQ_X_test))))

YxHQ_X_train, YxHQ_X_test, YxHQ_Y_train, YxHQ_Y_test = model_selection.train_test_split(YxHQ, YxHQ_Y, test_size = 0.9502723340332941)
hvwvhVfU.fit(YxHQ_X_train, YxHQ_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(YxHQ_Y_test, hvwvhVfU.predict(YxHQ_X_test))))

YxHQ_X_train, YxHQ_X_test, YxHQ_Y_train, YxHQ_Y_test = model_selection.train_test_split(YxHQ, YxHQ_Y, test_size = 0.897970409331531)
hvwvhVfU.fit(YxHQ_X_train, YxHQ_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(YxHQ_Y_test, hvwvhVfU.predict(YxHQ_X_test))))

