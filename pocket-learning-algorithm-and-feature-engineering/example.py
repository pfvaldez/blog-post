from sklearn import preprocessing
from pocket_classifier import PocketClassifier
from perceptron_classifier import PerceptronClassifier
import csv
import numpy as np
import pandas as pd
import urllib.request
from matplotlib import pyplot
import seaborn as sns

sns.set()

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
# A1:   b, a.
# A2:   continuous.
# A3:   continuous.
# A4:   u, y, l, t.
# A5:   g, p, gg.
# A6:   c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
# A7:   v, h, bb, j, n, z, dd, ff, o.
# A8:   continuous.
# A9:   t, f.
#A10:   t, f.
#A11:   continuous.
#A12:   t, f.
#A13:   g, p, s.
#A14:   continuous.
#A15:   continuous.
#A16:   +,-         (class attribute)

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

pocket_classifier_normalized = PocketClassifier(15, ('+', '-'))
pocket_classifier_normalized.train(data_normalized, label, 100)
pyplot.plot(pocket_classifier_normalized.misclassify_record, color='blue', label='Pocket (Normalized)')

# Compare to the non-normalized data.
pocket_classifier = PocketClassifier(15, ('+', '-'))
pocket_classifier.train(data, label, 100)
pyplot.plot(pocket_classifier.misclassify_record, color='green', label='Pocket (Non-normalized)')

# Compare to the Perceptron Learning Algorithm with both normalized and non-normalized data
perceptron_classifier_normalized = PerceptronClassifier(15, ('+', '-'))
perceptron_classifier_normalized.train(data_normalized, label, 100)
pyplot.plot(perceptron_classifier_normalized.misclassify_record, color='red', label='Perceptron (Normalized)')

perceptron_classifier = PerceptronClassifier(15, ('+', '-'))
perceptron_classifier.train(data, label, 100)
pyplot.plot(perceptron_classifier.misclassify_record, color='yellow', label='Perceptron (Non-normalized)')

# Plot the error rate and show it never converges
pyplot.xlabel('number of iteration')
pyplot.ylabel('number of misclassification')
pyplot.legend(loc='center right')
pyplot.show()





