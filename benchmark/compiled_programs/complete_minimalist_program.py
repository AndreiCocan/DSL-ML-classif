from sklearn import *
import pandas as pd

myData = pd.read_csv("../datasets/Iris__Species.csv")
myData = myData.dropna()

myData_Y = myData.iloc[:,-1]
myData = myData.iloc[:, :-1]

mySvmModel = svm.SVC()

myData_X_train, myData_X_test, myData_Y_train, myData_Y_test = model_selection.train_test_split(myData, myData_Y, test_size = 0.7)
mySvmModel.fit(myData_X_train, myData_Y_train)



