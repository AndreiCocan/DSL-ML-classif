from sklearn import *
import pandas as pd

OIgtDTG = pd.read_csv("../datasets/Iris__Species.csv")
OIgtDTG = OIgtDTG.dropna()

OIgtDTG_Y = OIgtDTG["Species"]
OIgtDTG = OIgtDTG.drop(columns=["Species"])

OIgtDTG_scaler = preprocessing.AbsMaxScaler()
OIgtDTG = OIgtDTG_scaler.fit_transform(OIgtDTG)

ZCYMKxw = pd.read_csv("../datasets/titanic__Survived.csv")
ZCYMKxw = ZCYMKxw.dropna()

ZCYMKxw_Y = ZCYMKxw.iloc[:,-1]
ZCYMKxw = ZCYMKxw.iloc[:, :-1]

