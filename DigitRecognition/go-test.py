#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import csv
import numpy as np

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

tests = loadTestData()
for i in range(len(tests)): 
    with open('./extract_tdata/line_%d.txt'%i, 'w') as file: 
        file.write("\n".join(str(e) for e in tests[i].reshape(28,28)))
