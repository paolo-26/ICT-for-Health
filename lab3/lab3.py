#!/usr/bin/env python3
"""
@author = Paolo Grasso
"""
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt

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

def findPatients(df, col, ft):
    cnt = df.count(axis=1, level=None, numeric_only=False)
    index = [k for k in df.index.values if cnt[k] == df.shape[1] -1]
    index = [k for k in index if np.isnan(df.iloc[k,col])]
    df = df.loc[index]
    #df.index = range(len(df))
    print(index)
    #print('We have %d patients' %cnt[-1])
    return df

class SolveRidge(object):
    def __init__(self, x_train, x_test, F0, m, s, ft, round=0):
        self.y = x_train.iloc[:,F0]  # Define y as column F0
        self.x_train = x_train.drop(columns=[ft[F0]]) # Remove F0 from x
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
        # plt.figure()
        # plt.stem(w)
        # plt.grid()
        # plt.ylabel(r'$\hat{\mathbf{w}}(f)$')
        # plt.show()


        self.y_hat_train = np.dot(x_train, w)*s + m

        if self.r == 1:
            self.y_hat_train = [round(k) for k in self.y_hat_train]

        self.y_hat_test =  np.dot(x_test, w)*s + m

        if self.r == 1:
            self.y_hat_test = [round(k) for k in self.y_hat_test]


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
    data = data.replace(to_replace='present', value=float(1))
    data = data.replace(to_replace='notpresent', value=float(0))
    data = data.replace(to_replace='normal', value=float(1))
    data = data.replace(to_replace='abnormal', value=float(0))
    data = data.replace(to_replace='yes', value=float(1))
    data = data.replace(to_replace='no', value=float(0))
    data = data.replace(to_replace='good', value=float(1))
    data = data.replace(to_replace='poor', value=float(0))
    data = data.replace(to_replace='ckd', value=float(1))
    data = data.replace(to_replace='notckd', value=float(0))

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



    for F0 in range(25):

        if features[F0] in categorical_features:
            round_ = 1
        else:
            round_ = 0

        x_test_or = findPatients(data, F0, features)  # Original data
        x_test = copy.deepcopy(x_test_or)

        # Standardize x_test
        for k in range(x.shape[1]):
            x_test.iloc[:,k] -= mean[k]
            x_test.iloc[:,k] /= std[k]

        x_test = x_test.drop(columns=features[F0])  # Remove column F0
        ridge = SolveRidge(x, x_test, F0, mean[F0], std[F0], features, round_)
        x_test_or[features[F0]] = ridge.y_hat_test  # Add regressed data to original data
        final = pd.concat([final, x_test_or])


    # final['rbc'] = final['rbc'].replace(to_replace=1, value='normal')

    final = final.sort_index()
    with open('final_data.csv', 'w') as outfile:
        final.to_csv(outfile)  # Test final file
