#!/usr/bin/env python3
"""
@author = Paolo Grasso

To visualize the tree: "dot -Tpdf Tree.dot -o Tree.pdf"
"""
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
from itertools import combinations

CAT_FEAT = ['rbc', 'pc', 'pcc', 'ba', 'htn',
            'dm', 'cad', 'appet', 'pe', 'ane', 'class']
INT_FEAT = ['age', 'bp', 'bgr', 'bu', 'al', 'su', 'sod', 'pcv', 'wbcc']
DEC_FEAT = ['pot', 'hemo', 'rbcc', 'sc']
LAMBDA = 10

def findCombinations(df):
    comb_list = []

    for k in df.index.values:
        nans = np.argwhere(np.isnan(df.loc[k]))
        nans = tuple(nans.reshape(len(nans),))

        if nans not in comb_list:
            comb_list.append(nans)

    return comb_list

def removePatients(df, n):
    """ Remove all patients with <= n valid values
        Return the new database 'df'.
    """
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] < n]
    df = df.drop(index)
    return df


def selectPatients(df, n):
    """ Keep only the patients with at least n valid values.
        Return the new database 'df'.
    """
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] < n]
    test = df.loc[index]
    df = df.drop(index)
    return (df, test)


def findPatients(df, feat_vect, ft):
    """ Find all patients whose missing features are contained in
        the feat_vect vector.
        Return the new database 'df' and a boolean value 'b' that
        tells if some patients are found.
    """
    try:
        L = len(feat_vect)

    except:
        L = 1

    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] == df.shape[1] - L]

    for c in range(L):
        index = [k for k in index if np.isnan(df.loc[k, ft[feat_vect[c]]])]

    df = df.loc[index]

    if index == []:
        b = 0  # Pruning

    else:
        print("Working on patients\t", index)
        b = 1

    return (df, b)


def roundValues(df, cat_f, int_f, dec_f):
    """ Round the values only for categorical features.
    """
    print("Rounding values...")
    for c in list(df):

        if (c in cat_f) or (c in int_f):
            df[c] = [int(round(x)) for x in df[c]]

        if c in dec_f:
            df[c] = [round(x, 1) for x in df[c]]

        if c == 'sg':
            df[c] = [round(x, 3) for x in df[c]]

    return df


class SolveRidge(object):

    def __init__(self, x_train, x_test, F0, feat_list, m, s):
        print("Regressing values of\t", feat_list)
        print("---------------------------------------")
        self.y = x_train.iloc[:, F0]  # Define y as F0 columns
        self.x_train = x_train.drop(columns=feat_list)  # Remove F0 from x
        self.x_test = x_test
        self.run(m, s, LAMBDA)

    def run(self, m, s, Lambda=10):
        x_train = self.x_train.values
        x_test = self.x_test.values
        y = self.y.values
        w = np.random.rand(x_train.shape[1], 1)
        I = np.eye(x_train.shape[1])
        w = np.dot(np.dot(np.linalg.inv(
            (np.dot(x_train.T, x_train) + Lambda * I)), x_train.T), y)
        self.w = w
        # print("w = \n", w, "\n")  # Utile
        self.y_hat_train = np.dot(x_train, w) * s + m
        # print("y_test = \n",np.dot(x_test, w) * s + m, "\n")
        #print("std = \n", s, "\n")
        #print("mean = \n", m, "\n")
        self.y_hat_test = np.dot(x_test, w) * s + m


if __name__ == '__main__':

    with open('chronic_kidney_disease.arff', 'r') as f:
        data = f.read()  # Read data as a text file

    data = data.replace(",\n", "\n")  # Delete extra commas at the end of lines
    data = data.replace("\t", "")  # Delete tabs
    data = data.replace(",,", ",")  # Replace double commas with single commas
    data = data.replace(", ", ",")  # Delete space after commas
    features = []

    with open('chronic_kidney_disease.arff', 'r') as f:

        for k in range(1, 200):
            t = f.readline()  # Read each line to detect features
            t = t.split(' ')

            if t[0] == '@attribute':  # Line that contain a feature name
                features.append(t[1].replace("'", ""))  # Add feature to list

            if t[0] == '@data':  # No more features, break
                break

    with open('dataset.arff', 'w') as out:
        out.write(data)  # Save the cleaned dataset

    # Import cleaned dataframe.
    data = pd.read_csv('dataset.arff', header=None, names=features, sep=',',
                       skiprows=29, na_values=['?'])

    data.info()
    cat = ['present', 'notpresent', 'abnormal', 'normal',
           'yes', 'no', 'good', 'poor', 'ckd', 'notckd']
    num = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    num = [float(x) for x in num]
    data = data.replace(to_replace=cat, value=num)

    with open('starting_data.csv', 'w') as outfile:
        data.to_csv(outfile)  # Save data as csv for better reading

    # Build x training with patients with full data.
    x = copy.deepcopy(data)  # Matrix with complete data
    x = removePatients(x, 20)
    (x, test) = selectPatients(x, 25)
    final = copy.deepcopy(x)

    # Standardize x training.
    mean = []
    std = []

    # with open('x.csv', 'w') as schifo:
    #     x.to_csv(schifo)

    for k in range(x.shape[1]):
        mean.append(np.mean(x.iloc[:, k]))
        x.iloc[:, k] -= mean[-1]
        std.append(np.std(x.iloc[:, k]))
        x.iloc[:, k] /= std[-1]

    # Regression.
    the_list = findCombinations(test)

    for F0 in the_list:

        try:
            F0 = list(F0)  # Convert tuple to list

        except:
            F0 = [F0]  # Convert integer to list of one element

        (x_test_or, b) = findPatients(test, F0, features)

        if b == 1:  # Pruning: run the algorithm only if there are patients
            x_test = copy.deepcopy(x_test_or)

            for k in range(x.shape[1]):
                x_test.iloc[:, k] -= mean[k]
                x_test.iloc[:, k] /= std[k]

            feat_list = [features[x] for x in F0]
            mean_list = [mean[x] for x in F0]
            std_list = [std[x] for x in F0]
            x_test = x_test.drop(columns=feat_list)
            ridge = SolveRidge(x, x_test, F0, feat_list, mean_list, std_list)
            x_test_or[feat_list] = ridge.y_hat_test
            final = pd.concat([final, x_test_or])

    # Reorder and save final results.
    final = final.sort_index()
    final = roundValues(final, CAT_FEAT, INT_FEAT, DEC_FEAT)

    with open('final_data.csv', 'w') as outfile:
        final.to_csv(outfile)

    print("features = ", features[0:-1])
    print("class =", features[-1])
    # Generate tree.
    data = final.iloc[:, 0:24]
    target = final['class']
    clf = tree.DecisionTreeClassifier("entropy")
    clf = clf.fit(data, target)
    dot_data = tree.export_graphviz(clf, out_file="Tree.dot",
                                    feature_names=features[0:-1],
                                    class_names=features[-1],
                                    filled=True, rounded=True,
                                    special_characters=True)
