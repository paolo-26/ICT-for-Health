#!/usr/bin/env python3
"""
@author = Paolo Grasso
"""
import pandas as pd

with open('chronic_kidney_disease.arff', 'r') as f:
    data = f.read()  # Read data as a text file

data = data.replace(",\n", "\n")  # Delete extra commas at the end of lines
data = data.replace("\t", "")  # Delete tabs
data = data.replace(",,", ",")  # Replace double commas with single commas

features = []
with open('chronic_kidney_disease.arff', 'r') as f:

    for k in range(1,200):
        t = f.readline()  # Read each line to detect features
        t = t.split(' ')

        if t[0] == '@attribute':  # Line that contain a feature name
            features.append(t[1])  # Add feature to list

        if t[0] == '@data':  # No more features, break
            break

with open('dataset.arff', 'w') as out:
     out.write(data)  # Save the cleaned dataset

data = pd.read_csv('dataset.arff', header=None, names=features, sep=',',
                   skiprows=29, na_values=['?'])  # Import cleaned dataframe

data = data.replace(to_replace='present', value=1)
data = data.replace(to_replace='notpresent', value=0)
data = data.replace(to_replace='normal', value=1)
data = data.replace(to_replace='abnormal', value=0)
data = data.replace(to_replace='yes', value=1)
data = data.replace(to_replace='no', value=0)
data = data.replace(to_replace='good', value=1)
data = data.replace(to_replace='poor', value=0)
data = data.replace(to_replace='ckd', value=1)
data = data.replace(to_replace='notckd', value=0)


data.info()
print(data.loc[0])
with open('data.csv', 'w') as outfile:
    data.to_csv(outfile)  # Save data as csv for better reading
