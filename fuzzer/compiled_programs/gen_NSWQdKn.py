from sklearn import *
import pandas as pd

fRFTGD = pd.read_csv("../datasets/titanic__Survived.csv")
fRFTGD = fRFTGD.dropna()

fRFTGD_Y = fRFTGD.iloc[:,-1]
fRFTGD = fRFTGD.iloc[:, :-1]

fRFTGD = fRFTGD.drop(columns=["Fare", "Sex"])

eGjgETp = pd.read_csv("../datasets/random-dataset__Class.csv")
eGjgETp = eGjgETp.dropna()

eGjgETp_Y = eGjgETp.iloc[:,-1]
eGjgETp = eGjgETp.iloc[:, :-1]

zgMa = tree.DecisionTreeClassifier()

URQRSAO = neighbors.KNeighborsClassifier()

MVdI = svm.SVC(C = 0.6002271059668445)

eGjgETp_X_train, eGjgETp_X_test, eGjgETp_Y_train, eGjgETp_Y_test = model_selection.train_test_split(eGjgETp, eGjgETp_Y, test_size = 0.951183882421597)
URQRSAO.fit(eGjgETp_X_train, eGjgETp_Y_train)

print("Accuracy score : " + str(metrics.accuracy_score(eGjgETp_Y_test, URQRSAO.predict(eGjgETp_X_test))))

