from sklearn import *
import pandas as pd

YSkeZKjE = pd.read_csv("../datasets/random-dataset__Class.csv")
YSkeZKjE = YSkeZKjE.dropna()

YSkeZKjE_Y = YSkeZKjE["Class"]
YSkeZKjE = YSkeZKjE.drop(columns=["Class"])

YSkeZKjE_scaler = preprocessing.MinMaxScaler()
YSkeZKjE = YSkeZKjE_scaler.fit_transform(YSkeZKjE)

LpVhOdi = pd.read_csv("../datasets/Iris__Species.csv")
LpVhOdi = LpVhOdi.dropna()

LpVhOdi_Y = LpVhOdi["Species"]
LpVhOdi = LpVhOdi.drop(columns=["Species"])

HACWt = neural_network.MLPClassifier()

saVRw = neighbors.KNeighborsClassifier()

YSkeZKjE_X_train, YSkeZKjE_X_test, YSkeZKjE_Y_train, YSkeZKjE_Y_test = model_selection.train_test_split(YSkeZKjE, YSkeZKjE_Y, test_size = 0.7)
saVRw.fit(YSkeZKjE_X_train, YSkeZKjE_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(YSkeZKjE_Y_test, saVRw.predict(YSkeZKjE_X_test))))

YSkeZKjE_X_train, YSkeZKjE_X_test, YSkeZKjE_Y_train, YSkeZKjE_Y_test = model_selection.train_test_split(YSkeZKjE, YSkeZKjE_Y, test_size = 0.7)
saVRw.fit(YSkeZKjE_X_train, YSkeZKjE_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(YSkeZKjE_Y_test, saVRw.predict(YSkeZKjE_X_test))))

LpVhOdi_X_train, LpVhOdi_X_test, LpVhOdi_Y_train, LpVhOdi_Y_test = model_selection.train_test_split(LpVhOdi, LpVhOdi_Y, test_size = 0.7)
saVRw.fit(LpVhOdi_X_train, LpVhOdi_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(LpVhOdi_Y_test, saVRw.predict(LpVhOdi_X_test))))

LpVhOdi_X_train, LpVhOdi_X_test, LpVhOdi_Y_train, LpVhOdi_Y_test = model_selection.train_test_split(LpVhOdi, LpVhOdi_Y, test_size = 0.7)
HACWt.fit(LpVhOdi_X_train, LpVhOdi_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(LpVhOdi_Y_test, HACWt.predict(LpVhOdi_X_test))))

