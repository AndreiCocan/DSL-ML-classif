from sklearn import *
import pandas as pd

iHdl = pd.read_csv("../datasets/Iris__Species.csv")
iHdl = iHdl.dropna()

iHdl_Y = iHdl["Species"]
iHdl = iHdl.drop(columns=["PetalWidthCm", "PetalLengthCm", "Species"])

iHdl_scaler = preprocessing.AbsMaxScaler()
iHdl = iHdl_scaler.fit_transform(iHdl)

XZq = tree.DecisionTreeClassifier()

DqZQI = neural_network.MLPClassifier()

bje = tree.DecisionTreeClassifier()

hDBcF = neighbors.KNeighborsClassifier(n_neighbors = 9)

iHdl_X_train, iHdl_X_test, iHdl_Y_train, iHdl_Y_test = model_selection.train_test_split(iHdl, iHdl_Y, test_size = 0.7187335177509611)
DqZQI.fit(iHdl_X_train, iHdl_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(iHdl_Y_test, DqZQI.predict(iHdl_X_test))))

iHdl_X_train, iHdl_X_test, iHdl_Y_train, iHdl_Y_test = model_selection.train_test_split(iHdl, iHdl_Y, test_size = 0.7)
hDBcF.fit(iHdl_X_train, iHdl_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(iHdl_Y_test, hDBcF.predict(iHdl_X_test))))

iHdl_X_train, iHdl_X_test, iHdl_Y_train, iHdl_Y_test = model_selection.train_test_split(iHdl, iHdl_Y, test_size = 0.7)
XZq.fit(iHdl_X_train, iHdl_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(iHdl_Y_test, XZq.predict(iHdl_X_test))))

iHdl_X_train, iHdl_X_test, iHdl_Y_train, iHdl_Y_test = model_selection.train_test_split(iHdl, iHdl_Y, test_size = 0.26410847941225735)
XZq.fit(iHdl_X_train, iHdl_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(iHdl_Y_test, XZq.predict(iHdl_X_test))))

