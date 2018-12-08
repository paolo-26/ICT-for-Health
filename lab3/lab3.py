#!/usr/bin/env python3
"""
@author = Paolo Grasso
"""
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
import graphviz

CAT_FEAT = ['rbc', 'pc', 'pcc', 'ba', 'htn',
            'dm', 'cad', 'appet', 'pe', 'ane', 'class']
INT_FEAT = ['age', 'bp', 'bgr', 'bu', 'al', 'su', 'sod', 'pcv', 'wbcc']
DEC_FEAT = ['pot', 'hemo', 'rbcc', 'sc']
FEAT = ['age', 'blood pressure', 'specific gravity', 'albumin', 'sugar',
        'red blood cells', 'pus cell', 'pus cell clumps', 'bacteria',
        'blood glucose random', 'blood urea', 'serum creatinine', 'sodium',
        'potassium', 'hemoglobin', 'packed cell volume',
        'white blood cell count', 'red blood cell count', 'hypertension',
        'diabetes mellitus', 'coronary artery disease', 'appetite',
        'pedal edema', 'anemia', 'class']
CAT = ['present', 'notpresent', 'normal', 'abnormal',
       'yes', 'no', 'good', 'poor', 'ckd', 'notckd']
NUM = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
SG = [1.005, 1.010, 1.015, 1.020, 1.025]
SGN = [1, 2, 3, 4, 5]
LAMBDA = 10


def find_combinations(df):
    """ Find all possible combinations of missing features of a given dataset.
    """
    comb_list = []

    for k in df.index.values:
        nans = np.argwhere(np.isnan(df.loc[k]))
        nans = tuple(nans.reshape(len(nans),))

        if nans not in comb_list:
            comb_list.append(nans)

    return comb_list


def remove_patients(df, n):
    """ Remove all patients with < n valid values
        Return the new database 'df'.
    """
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] < n]
    df = df.drop(index)
    return df


def select_patients(df, n):
    """ Keep only the patients with at least n valid values.
        Return the new database 'df'.
    """
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] < n]
    test = df.loc[index]
    df = df.drop(index)
    return (df, test)


def find_patients(df, feat_vect, ft):
    """ Find all patients whose missing features are contained in
        the feat_vect vector.
        Return the new database 'df' and a boolean value 'b' that
        tells if some patients are found.
    """
    L = len(feat_vect)
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] == df.shape[1] - L]

    for c in range(L):
        index = [k for k in index if np.isnan(df.loc[k, ft[feat_vect[c]]])]

    df = df.loc[index]
    print("Working on patients\t", index)
    return df


def round_values(df, cat_f, int_f, dec_f):
    """ Round the values only for categorical features.
    """
    for c in list(df):

        if (c in cat_f) or (c in int_f):
            df[c] = [int(round(x)) for x in df[c]]

        if c in dec_f:
            df[c] = [round(x, 1) for x in df[c]]

        if c == 'sg':
            df[c] = [round(x) for x in df[c]]
            df[c] = df[c].replace(to_replace=SGN, value=SG)

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
        #print("w = \n", w, "\n")  # Utile
        self.y_hat_train = np.dot(x_train, w) * s + m
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
    num = [float(x) for x in NUM]
    data = data.replace(to_replace=CAT, value=NUM)
    data = data.replace(to_replace=SG, value=SGN)

    with open('starting_data.csv', 'w') as outfile:
        data.to_csv(outfile)  # Save data as csv for better reading

    # Build x training with patients with full data.
    x = copy.deepcopy(data)  # Matrix with complete data
    x = remove_patients(x, 20)
    (x, test) = select_patients(x, 25)
    final = copy.deepcopy(x)

    # Standardize x training.
    mean = []
    std = []

    for k in range(x.shape[1]):
        mean.append(np.mean(x.iloc[:, k]))
        x.iloc[:, k] -= mean[-1]
        std.append(np.std(x.iloc[:, k]))
        x.iloc[:, k] /= std[-1]

    # Regression.
    the_list = find_combinations(test)

    for F0 in the_list:
        F0 = list(F0)  # Convert tuple to list
        x_test_or = find_patients(test, F0, features)
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
    final = round_values(final, CAT_FEAT, INT_FEAT, DEC_FEAT)

    with open('final_data.csv', 'w') as outfile:
        final.to_csv(outfile)

    # Generate tree.
    data = final.iloc[:, 0:24]
    target = final['class']
    clf = tree.DecisionTreeClassifier("entropy")
    clf.fit(data, target)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=FEAT[0:-1],
                                    class_names=['0','1'],
                                    filled=True, rounded=True,
                                    special_characters=True,
                                    #proportion=True,
                                    #leaves_parallel=True,
                                    )
    graph = graphviz.Source(dot_data)
    graph.render("Tree")
