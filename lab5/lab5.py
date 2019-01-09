#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author = Paolo Grasso
"""

import pandas as pd
import numpy as np
from sklearn import svm

def main():
    TAR = 'class'

    data = pd.read_csv("final_data.csv", index_col=0, sep=',')
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    target = data[TAR]

    split = int(np.round(0.8*len(data)))

    train = data.iloc[1:split].drop(columns=TAR)
    test = data.iloc[split+1:].drop(columns=TAR)

    target_train = target.iloc[1:split]
    target_test = target.iloc[split+1:].values

    clf = svm.SVC(gamma ='scale')
    clf.fit(train, target_train)
    svm.SVC(random_state=42)
    y_pred = clf.predict(test)

    print("\nPredicted feature: %s" % TAR)
    print("\nPredicted = ", y_pred)
    print("\n    Real  = ", target_test)
    correct = np.sum((y_pred == target_test)*1)
    print("\nCorrect predictions: %d/%d (%.2f%%)" %(correct, len(y_pred),
                                                    correct/len(y_pred)*100))


if __name__ == '__main__':
    main()
