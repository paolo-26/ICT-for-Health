#!/usr/bin/env python3
"""
@author = Paolo Grasso
"""
import pandas as pd

with open('chronic_kidney_disease.arff', 'r') as f:
    data = f.read()

data = data.replace(",\n", "\n")
data = data.replace("\t", "")
data = data.replace(",,", ",")

features = []
with open('chronic_kidney_disease.arff', 'r') as f:

    for k in range(1,200):
        t = f.readline()
        t = t.split(' ')

        if t[0] == '@attribute':
            features.append(t[1])

        if t[0] == '@data':
            break

with open('dataset.arff', 'w') as out:
     out.write(data)

data = pd.read_csv('dataset.arff', header=None, names=features, skiprows=29)
data.info()
