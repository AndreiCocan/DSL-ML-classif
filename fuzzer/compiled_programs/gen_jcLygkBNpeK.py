from sklearn import *
import pandas as pd

xXKxpq = pd.read_csv("../datasets/Iris__Species.csv")
xXKxpq = xXKxpq.dropna()

xXKxpq_Y = xXKxpq["Species"]
xXKxpq = xXKxpq.drop(columns=["SepalWidthCm", "PetalWidthCm", "Species"])

