from sklearn import *
import pandas as pd

uWdkwKZ = pd.read_csv("../datasets/Iris__Species.csv")
uWdkwKZ = uWdkwKZ.dropna()

uWdkwKZ_Y = uWdkwKZ["Species"]
uWdkwKZ = uWdkwKZ.drop(columns=["Species"])

uWdkwKZ_scaler = preprocessing.AbsMaxScaler()
uWdkwKZ = uWdkwKZ_scaler.fit_transform(uWdkwKZ)

LkwzMD = pd.read_csv("../datasets/Iris__Species.csv")
LkwzMD = LkwzMD.dropna()

LkwzMD_Y = LkwzMD.iloc[:,-1]
LkwzMD = LkwzMD.iloc[:, :-1]

LkwzMD = LkwzMD.drop(columns=["PetalWidthCm"])

fLWrcXw = neural_network.MLPClassifier()

GOQdtFN = neighbors.KNeighborsClassifier(n_neighbors = 10)

VwPl = neighbors.KNeighborsClassifier()

LkwzMD_X_train, LkwzMD_X_test, LkwzMD_Y_train, LkwzMD_Y_test = model_selection.train_test_split(LkwzMD, LkwzMD_Y, test_size = 0.7)
fLWrcXw.fit(LkwzMD_X_train, LkwzMD_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(LkwzMD_Y_test, fLWrcXw.predict(LkwzMD_X_test))))

uWdkwKZ_X_train, uWdkwKZ_X_test, uWdkwKZ_Y_train, uWdkwKZ_Y_test = model_selection.train_test_split(uWdkwKZ, uWdkwKZ_Y, test_size = 0.323319472294745)
fLWrcXw.fit(uWdkwKZ_X_train, uWdkwKZ_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(uWdkwKZ_Y_test, fLWrcXw.predict(uWdkwKZ_X_test))))

