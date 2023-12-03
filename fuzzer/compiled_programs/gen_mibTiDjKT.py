from sklearn import *
import pandas as pd

rxT = pd.read_csv("../datasets/titanic__Survived.csv")
rxT = rxT.dropna()

rxT_Y = rxT.iloc[:,-1]
rxT = rxT.iloc[:, :-1]

rxT_scaler = preprocessing.StandardScaler()
rxT = rxT_scaler.fit_transform(rxT)

BHPZMp = neural_network.MLPClassifier()

YPwdvlcJ = svm.SVC(C = 0.7552881731521942)

yVymq = neighbors.KNeighborsClassifier(n_neighbors = 10)

mwCsv = neighbors.KNeighborsClassifier(n_neighbors = 8)

wdrdjaKR = neighbors.KNeighborsClassifier(n_neighbors = 5)

rxT_X_train, rxT_X_test, rxT_Y_train, rxT_Y_test = model_selection.train_test_split(rxT, rxT_Y, test_size = 0.6206397968364019)
yVymq.fit(rxT_X_train, rxT_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(rxT_Y_test, yVymq.predict(rxT_X_test))))

rxT_X_train, rxT_X_test, rxT_Y_train, rxT_Y_test = model_selection.train_test_split(rxT, rxT_Y, test_size = 0.7)
YPwdvlcJ.fit(rxT_X_train, rxT_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(rxT_Y_test, YPwdvlcJ.predict(rxT_X_test))))

