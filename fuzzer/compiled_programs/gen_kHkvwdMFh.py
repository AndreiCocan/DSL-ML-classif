from sklearn import *
import pandas as pd

DJmokB = pd.read_csv("../datasets/random-dataset__Class.csv")
DJmokB = DJmokB.dropna()

DJmokB_Y = DJmokB.iloc[:,-1]
DJmokB = DJmokB.iloc[:, :-1]

DJmokB_scaler = preprocessing.AbsMaxScaler()
DJmokB = DJmokB_scaler.fit_transform(DJmokB)

ISdRk = pd.read_csv("../datasets/Iris__Species.csv")
ISdRk = ISdRk.dropna()

ISdRk_Y = ISdRk["Species"]
ISdRk = ISdRk.drop(columns=["Id", "Species"])

aycXnM = pd.read_csv("../datasets/random-dataset__Class.csv")
aycXnM = aycXnM.dropna()

aycXnM_Y = aycXnM["Class"]
aycXnM = aycXnM.drop(columns=["Id2", "Id1", "Class"])

yrAzL = pd.read_csv("../datasets/Iris__Species.csv")
yrAzL = yrAzL.dropna()

yrAzL_Y = yrAzL["Species"]
yrAzL = yrAzL.drop(columns=["PetalWidthCm", "Species"])

