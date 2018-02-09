#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import multiprocessing as mp

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

def distance(vx,vy):
    return sum((x-y)**2 for x,y in zip(vx,vy))**0.5

def possible(i, kl, t):
    res={}
    for k in kl:
        if k[1] in ko.keys():
            ko[k[1]] += 1
        else:
            ko[k[1]] = 1
    print(i, ko)
    l,c = -1,0
    for key in ko.keys():
        if ko[key] > c and ko[key] > len(kl)/3:
            l,c = key,ko[key]
    if l == -1:
        l=np.round(rfr.predict([t]))[0]
        print("######:%d"%l)
    return l
            

def knn(trains, test, index, n):
    kl = []
    for train in trains:
        dis=distance(train[1:], test)
        if len(kl) < n:
            kl.append([dis, train[0]])
        else:
            toRemove,tmpDis = -1, dis
            for i in range(n):
                if kl[i][0] > tmpDis:
                    toRemove,tmpDis = i, kl[i][0]
            if toRemove != -1:
                kl[toRemove] = [dis, train[0]]
    return (index, possible(index, kl, test))

def knn_entry(args):
    return knn(args[0],args[1],args[2], args[3])

def go(trains, tests, n):
    pool = mp.Pool()
    res=pool.map(knn_entry, [(trains, tests[i], i, n) for i in range(len(tests))] )
    pool.close()
    return res

trains = loadTrainData()
tests = loadTestData()
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(trains.iloc[:,1:], trains.iloc[:,0])
print(tests, trains)
res = go(trains, tests, 3000)
with open('result.csv', 'wb') as file:
    writer=csv.writer(file)
    writer.writerow("ImageId,Label")
    for i in len(res):
        writer.writerow("%d,%d"%(i,res[i]))
