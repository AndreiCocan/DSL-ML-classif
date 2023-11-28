from sklearn import *
import pandas as pd

myData = pd.read_csv("tests/datasets/Iris.csv")

myData_Y = myData["Species"]
myData = myData.drop(columns=["Species"])

myData_scaler = preprocessing.MinMaxScaler()
myData = myData_scaler.fit_transform(myData)

myFirstModel = svm.SVC(kernel = "linear", C = 1.0)

myData_X_train, myData_X_test, myData_Y_train, myData_Y_test = model_selection.train_test_split(myData, myData_Y, test_size = 0.1)
myFirstModel.fit(myData_X_train, myData_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(myData_Y_test, myFirstModel.predict(myData_X_test))))