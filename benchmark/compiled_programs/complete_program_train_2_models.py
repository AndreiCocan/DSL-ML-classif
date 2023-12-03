from sklearn import *
import pandas as pd

myData = pd.read_csv("../datasets/titanic__Survived.csv")
myData = myData.dropna()

myData_Y = myData["Survived"]
myData = myData.drop(columns=["PassengerId", "SibSp", "Survived"])

myData2 = pd.read_csv("../datasets/titanic__Survived.csv")
myData2 = myData2.dropna()

myData2_Y = myData2.iloc[:,-1]
myData2 = myData2.iloc[:, :-1]

mySvmModel = svm.SVC(kernel = "sigmoid", C = 0.8)

myKnnModel = neighbors.KNeighborsClassifier(n_neighbors = 8, weights = "distance")

myData_X_train, myData_X_test, myData_Y_train, myData_Y_test = model_selection.train_test_split(myData, myData_Y, test_size = 0.7)
mySvmModel.fit(myData_X_train, myData_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(myData_Y_test, mySvmModel.predict(myData_X_test))))

myData2_X_train, myData2_X_test, myData2_Y_train, myData2_Y_test = model_selection.train_test_split(myData2, myData2_Y, test_size = 0.7)
myKnnModel.fit(myData2_X_train, myData2_Y_train)



