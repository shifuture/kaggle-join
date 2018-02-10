#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def loadTrainData():
    data = pd.read_csv("./data/train.csv")
    for i in range(28*28):
        data.loc[(data['pixel%d'%i]>0), 'pixel%d'%i] = 1
    for i in range(data.shape[0]):
        data.iloc[i,1:] = reintegrate(data.iloc[i,1:].as_matrix())
    return data

def loadTestData():
    data = pd.read_csv("./data/test.csv")
    for i in range(28*28):
        data.loc[(data['pixel%d'%i]>0), 'pixel%d'%i] = 1
    for i in range(data.shape[0]):
        data.iloc[i] = reintegrate(data.iloc[i].as_matrix())
    return data

def reintegrate(data):
    data=data.reshape(28,28)
    res = data[~np.all(data==0,axis=1)]
    toFilNum = len(data) - len(res)
    res = np.concatenate((res, np.zeros((toFilNum,28))))
    res = res[:,~np.all(res==0,axis=0)]
    toFilNum = data.shape[1] - res.shape[1]
    res = np.concatenate((res, np.zeros((28,toFilNum))), axis=1)
    return  np.array(list(res.flat))

trains = loadTrainData()
tests = loadTestData()
rfr = RandomForestRegressor(random_state=0, n_estimators=10000, n_jobs=-1)
rfr.fit(trains.iloc[:,1:], trains.iloc[:,0])
predictNum = np.round(rfr.predict(tests))
print(predictNum)
np.savetxt("result.csv", np.dstack((np.arange(1, predictNum.size+1), predictNum))[0], "%d,%d", header="ImageId,Label")
