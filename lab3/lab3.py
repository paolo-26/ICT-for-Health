#!/usr/bin/env python3
"""
@author = Paolo Grasso
"""
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
from itertools import combinations

categorical_features = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','class']
LAMBDA = 10

def removePatients(df, n):
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] <= n]
    df = df.drop(index)
    #df.index = range(len(df))
    return df

def selectPatients(df, n):
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] < n]
    df = df.drop(index)
    #df.index = range(len(df))
    return df

def findPatients(df, col, ft, c):
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] == df.shape[1] -c]
    index = [k for k in index if np.isnan(df.iloc[k,col])]
    df = df.loc[index]
    #df.index = range(len(df))
    print("Patients found:", index)
    #print('We have %d patients' %cnt[-1])
    return df

def findTwo(df, vec, ft):
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] == df.shape[1] -len(vec)]

    for c in range(len(vec)):
        index = [k for k in index if np.isnan(df.iloc[k,vec[c]])]

    df = df.loc[index]

    if index == []:
        b = 0
    else:
        print(index)
        b = 1

    return (df, b)


def roundValues(df, cat_f):

    for c in list(df):
        if c in cat_f:
            df[c] = [round(x) for x in df[c]]

    return df

class SolveRidge(object):

    def __init__(self, x_train, x_test, F0, feat_list, m, s, round=0):
        print("Regressing", feat_list, "\n")
        self.y = x_train.iloc[:,F0]  # Define y as column F0
        self.x_train = x_train.drop(columns=feat_list) # Remove F0 from x
        self.x_test = x_test
        self.r = round
        self.run(m ,s, LAMBDA)

    def run(self, m, s, Lambda=10):
        it = 0
        x_train = self.x_train.values
        x_test = self.x_test.values
        y = self.y.values
        L = Lambda
        w = np.random.rand(x_train.shape[1],1)
        I = np.eye(x_train.shape[1])
        w = np.dot(np.dot(np.linalg.inv((np.dot(x_train.T,x_train) + L*I)),x_train.T),y)
        self.w = w
        # print("w = \n", w, "\n")  # Utile

        # plt.figure()
        # plt.stem(w)
        # plt.grid()
        # plt.ylabel(r'$\hat{\mathbf{w}}(f)$')
        # plt.show()

        self.y_hat_train = np.dot(x_train, w)*s + m

        # if self.r == 1:
        #     self.y_hat_train = [round(k) for k in self.y_hat_train]

        self.y_hat_test =  np.dot(x_test, w)*s + m

        # if self.r == 1:
        #     self.y_hat_test = [round(k) for k in self.y_hat_test]


        # plt.figure()
        # plt.plot(self.y_hat_test,'x')
        # plt.title('Regressed data for feature %d' % F0)
        # #plt.plot(self.y_hat_train, '.')
        # plt.show()



        # plt.figure()
        # tmp = [round(k) for k in (self.y_hat*s +m)]
        # plt.plot(tmp, '.')
        # tmp2 = y*s + m
        # plt.plot(tmp2, '.')
        # plt.xlabel('Nulla')
        # plt.ylabel('So io di cosa si tratta')
        # plt.legend(['ycappello', 'ysenzacappello'])
        # plt.show()
        #
        # plt.figure()
        # plt.plot(tmp-tmp2)
        # plt.show()

        #print("Regression complete\n")



    def printInfo(self):
        print("Optimum weight vector: \n", self.w)


if __name__ == '__main__':
    with open('chronic_kidney_disease.arff', 'r') as f:
        data = f.read()  # Read data as a text file

    data = data.replace(",\n", "\n")  # Delete extra commas at the end of lines
    data = data.replace("\t", "")  # Delete tabs
    data = data.replace(",,", ",")  # Replace double commas with single commas
    data = data.replace(", ", ",")
    features = []

    with open('chronic_kidney_disease.arff', 'r') as f:

        for k in range(1,200):
            t = f.readline()  # Read each line to detect features
            t = t.split(' ')

            if t[0] == '@attribute':  # Line that contain a feature name
                features.append(t[1].replace("'",""))  # Add feature to list
            if t[0] == '@data':  # No more features, break
                break

    with open('dataset.arff', 'w') as out:
         out.write(data)  # Save the cleaned dataset

    data = pd.read_csv('dataset.arff', header=None, names=features, sep=',',
                   skiprows=29, na_values=['?'])  # Import cleaned dataframe

    data.info()
    cat = ['present','notpresent','normal','abnormal','yes','no','good','poor','ckd','notckd']
    num = [1,0,1,0,1,0,1,0,1,0]
    num = [float(x) for x in num]
    # data = data.replace(to_replace='present', value=float(1))
    # data = data.replace(to_replace='notpresent', value=float(0))
    # data = data.replace(to_replace='normal', value=float(1))
    # data = data.replace(to_replace='abnormal', value=float(0))
    # data = data.replace(to_replace='yes', value=float(1))
    # data = data.replace(to_replace='no', value=float(0))
    # data = data.replace(to_replace='good', value=float(1))
    # data = data.replace(to_replace='poor', value=float(0))
    # data = data.replace(to_replace='ckd', value=float(1))
    # data = data.replace(to_replace='notckd', value=float(0))
    data = data.replace(to_replace=cat, value=num)

    with open('starting_data.csv', 'w') as outfile:
        data.to_csv(outfile)  # Save data as csv for better reading

    x = copy.deepcopy(data)  # Matrix with complete data
    x = removePatients(x, 20)
    x = selectPatients(x, 25)
    final = copy.deepcopy(x)

    # Standardize x.
    mean = []
    std = []
    for k in range(x.shape[1]):
        mean.append(np.mean(x.iloc[:,k]))
        x.iloc[:,k] -= mean[-1]

        std.append(np.std(x.iloc[:,k]))
        x.iloc[:,k] /= std[-1]


    print(len(mean))







    for F0 in range(25):

        if features[F0] in categorical_features:
            round_ = 1
        else:
            round_ = 0

        x_test_or = findPatients(data, F0, features, 1)  # Original data
        x_test = copy.deepcopy(x_test_or)

        # Standardize x_test
        for k in range(x.shape[1]):
            x_test.iloc[:,k] -= mean[k]
            x_test.iloc[:,k] /= std[k]

        feat_list = features[F0]
        x_test = x_test.drop(columns=features[F0])  # Remove column F0
        ridge = SolveRidge(x, x_test, F0, feat_list, mean[F0], std[F0], round_)
        x_test_or[features[F0]] = ridge.y_hat_test  # Add regressed data to original data
        final = pd.concat([final, x_test_or])







    # final['rbc'] = final['rbc'].replace(to_replace=1, value='normal')


    print("\n\n\n----EXTEND----\n\n\n")
    the_list = list(combinations(range(25), 2))
    the_list.extend(list(combinations(range(25), 3)))
    the_list.extend(list(combinations(range(25), 4)))
    #print (list(the_list))

    for F0 in the_list:
        F0 = list(F0)
        #print('Regressing %d features' %len(F0))

        (x_test_or, b) = findTwo(data, F0, features)

        if b == 1:
            x_test = copy.deepcopy(x_test_or)
            # print("-----------\n")
            # print(x_test)
            # print("\n-----------")
            for k in range(x.shape[1]):
                x_test.iloc[:,k] -= mean[k]
                x_test.iloc[:,k] /= std[k]

            feat_list = [features[x] for x in F0]
            mean_list = [mean[x] for x in F0]
            std_list = [std[x] for x in F0]

            round_list = []
            for k in range(len(feat_list)):
                if feat_list[k] in categorical_features:
                    round_list.append(1)
                else:
                    round_list.append(0)

            print(round_list)
            x_test = x_test.drop(columns=feat_list)
            ridge = SolveRidge(x, x_test, F0, feat_list, mean_list, std_list, feat_list)
            x_test_or[feat_list] = ridge.y_hat_test
            final = pd.concat([final, x_test_or])


    # Final results.
    final = final.sort_index()
    final = roundValues(final, categorical_features)
    with open('final_data.csv', 'w') as outfile:
        final.to_csv(outfile)  # Test final file


    # Tree.
    target = final.iloc[:,-1]
    data = final.drop(columns=['class'])
    clf = tree.DecisionTreeClassifier("entropy")
    clf = clf.fit(data, target)
    dot_data = tree.export_graphviz(clf, out_file="Tree.dot",
        filled=True, rounded=True, special_characters=True)
