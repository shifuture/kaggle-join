#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import csv
import numpy as np
import collections
from threading import Thread

def loadTrainData():
    l=[]
    with open('./data/train.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append([line[0]] + list(e if e=='0' else 1 for e in line[1:]))
    #remove csv head
    l.remove(l[0])
    data=np.array(l, int)
    data=data.reshape(42000,785)
    label=data[:,0]
    data=data[:,1:]
    return label,data

def loadTestData():
    l=[]
    with open('./data/test.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(list(e if e=='0' else 1 for e in line))
    #remove csv head
    l.remove(l[0])
    data=np.array(l, int)
    data=data.reshape(28000,784)
    return data

def distance(vx, vy):
    return sum((x-y)**2 for x,y in zip(vx, vy))**0.5
    #return np.sqrt(np.sum((vx- vy)**2))

def kadd(kl, dis, l, k):
    if len(kl) < k:
        kl.append([dis, l])
    else:
        toRemove,toRemoveDis = -1, dis
        for i in range(k):
            if kl[i][0] > toRemoveDis:
                toRemove,toRemoveDis = i, kl[i][0]
        if toRemove != -1:
            kl[toRemove] = [dis, l]
    return kl

def possible(i, kl):
    ko = {}
    for k in kl:
        if k[1] in ko.keys():
            ko[k[1]]+=1
        else:
            ko[k[1]] =1
    print(i, ko)
    l = -1
    c = 0
    for key in ko.keys():
        if ko[key] > c:
            l = key
            c = ko[key]
    return l

def knn(res, index, trains, test, labels, k):
    kl=[]
    for i in range(len(trains)):
        dis=distance(test, trains[i])
        kl=kadd(kl, dis, labels[i], k)
    res[index] = possible(index, kl)


def go(trains, tests, labels, k, tn):
    res, threads = {},[]
    for i in range(len(tests)):
        if len(threads)>=tn :
            for t in threads:
                if t.is_alive() :
                    t.join()
                threads.remove(t)
                break
        t = Thread(target=knn, args=(res, i, trains, tests[i], labels, k))
        t.start()
        threads.append(t)
    for t in threads:
        if t.is_alive() :
            t.join()
    return res

labels, trains = loadTrainData()
tests = loadTestData()
res = go(trains, tests, labels, 3000, 6)
res = collections.OrderedDict(sorted(res.items())).values()
with open('result.csv', 'wb') as file:
    writer=csv.writer(file)
    writer.writerow(res)
