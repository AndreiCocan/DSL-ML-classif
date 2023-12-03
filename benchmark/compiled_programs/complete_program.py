from sklearn import *
import pandas as pd

myData = pd.read_csv("../datasets/titanic__Survived.csv")
myData = myData.dropna()

myData_Y = myData["Survived"]
myData = myData.drop(columns=["PassengerId", "SibSp", "Survived"])

myData_scaler = preprocessing.MinMaxScaler()
myData = myData_scaler.fit_transform(myData)

mySvmModel = svm.SVC(kernel = "sigmoid", C = 0.8)

myData_X_train, myData_X_test, myData_Y_train, myData_Y_test = model_selection.train_test_split(myData, myData_Y, test_size = 0.7)
mySvmModel.fit(myData_X_train, myData_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(myData_Y_test, mySvmModel.predict(myData_X_test))))

