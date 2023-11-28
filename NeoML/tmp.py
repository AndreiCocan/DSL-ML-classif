from sklearn import *
import pandas as pd

myData = pd.read_csv("../datasets/random-dataset__Class.csv")
myData = myData.dropna()

myData_Y = myData["Class"]
myData = myData.drop(columns=["Class"])

mySecondModel = neighbors.KNeighborsClassifier(n_neighbors = 8, weights = "distance")

myData_X_train, myData_X_test, myData_Y_train, myData_Y_test = model_selection.train_test_split(myData, myData_Y, test_size = 0.2)
mySecondModel.fit(myData_X_train, myData_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(myData_Y_test, mySecondModel.predict(myData_X_test))))

