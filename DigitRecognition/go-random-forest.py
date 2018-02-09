#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def loadTrainData():
    data = pd.read_csv("./data/train.csv")
    for i in range(28*28):
        data.loc[(data['pixel%d'%i]>0), 'pixel%d'%i] = 1
    return data

def loadTestData():
    data = pd.read_csv("./data/test.csv")
    for i in range(28*28):
        data.loc[(data['pixel%d'%i]>0), 'pixel%d'%i] = 1
    return data

trains = loadTrainData()
tests = loadTestData()
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(trains.iloc[:,1:], trains.iloc[:,0])
predictNum = np.round(rfr.predict(tests))
print(predictNum)
np.savetxt("result.csv", np.dstack((np.arange(1, predictNum.size+1), predictNum))[0], "%d,%d", header="ImageId,Label")
