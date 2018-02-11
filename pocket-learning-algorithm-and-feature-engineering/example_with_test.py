from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from pocket_classifier import PocketClassifier
from perceptron_classifier import PerceptronClassifier
import csv
import numpy as np
import pandas as pd
import urllib.request

# Download Japanese Credit Data Set from http://archive.ics.uci.edu/ml/datasets/Japanese+Credit+Screening
URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
urllib.request.urlretrieve(URL, 'crx.data')
# Use pandas.read_csv module to load adult data set
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

# 1. Remove missing row
no_missing = open('crx_nomissing.csv', 'w', newline=None)
writer = csv.writer(no_missing)
with open('crx.data')  as csv_file:
    CRX_READER = csv.reader(csv_file)
    for row in CRX_READER:
        if '?' in row:
            continue
        writer.writerow(row)

# 2. Transfer the following category data to numerical data:
CRX_DATA = pd.read_csv('crx_nomissing.csv', header=None)

# Use scikit-learn's LabelEncoder to encode category data to numerical data.
# For example, (a, b) -> (0, 1)
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
LABEL_ENCODER = preprocessing.LabelEncoder()

A1 = LABEL_ENCODER.fit_transform(CRX_DATA.iloc[:, 0])
A4 = LABEL_ENCODER.fit_transform(CRX_DATA.iloc[:, 3])
A5 = LABEL_ENCODER.fit_transform(CRX_DATA.iloc[:, 4])
A6 = LABEL_ENCODER.fit_transform(CRX_DATA.iloc[:, 5])
A7 = LABEL_ENCODER.fit_transform(CRX_DATA.iloc[:, 6])
A9 = LABEL_ENCODER.fit_transform(CRX_DATA.iloc[:, 8])
A10 = LABEL_ENCODER.fit_transform(CRX_DATA.iloc[:, 9])
A12 = LABEL_ENCODER.fit_transform(CRX_DATA.iloc[:, 11])
A13 = LABEL_ENCODER.fit_transform(CRX_DATA.iloc[:, 12])

# Build the training data based on the encoded data and numerical data.
data = list()
label = list()
for idx in range(A1.size):
    temp = list()
    temp.append(A1[idx])
    temp.append(CRX_DATA[1][idx])
    temp.append(CRX_DATA[2][idx])
    temp.append(A4[idx])
    temp.append(A5[idx])
    temp.append(A6[idx])
    temp.append(A7[idx])
    temp.append(CRX_DATA[7][idx])
    temp.append(A9[idx])
    temp.append(A10[idx])
    temp.append(CRX_DATA[10][idx])
    temp.append(A12[idx])
    temp.append(A13[idx])
    temp.append(CRX_DATA[13][idx])
    temp.append(CRX_DATA[14][idx])
    data.append(temp)
    label.append(CRX_DATA[15][idx])


# 3. Use Normalizer from scikit-learn to normalize the data
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
data_normalized = preprocessing.normalize(data)

# FIXME should be before normalization???
DATA_TRAIN, DATA_TEST, LABELS_TRAIN, LABELS_TEST = train_test_split(data_normalized, label, test_size=0.25, random_state=1000)

pocket_classifier = PocketClassifier(15, ('+', '-'))
pocket_classifier.train(DATA_TRAIN, LABELS_TRAIN, 100)

result = pocket_classifier.classify(DATA_TEST)

misclassify = 0
for predict, answer in zip(result, LABELS_TEST):
    if predict != answer:
        misclassify += 1
print("Accuracy rate: " + str((len(result) - misclassify) / len(result)) + "%") 

