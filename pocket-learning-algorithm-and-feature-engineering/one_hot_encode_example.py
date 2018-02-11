from sklearn import preprocessing
import csv
import numpy as np
import pandas as pd
import urllib.request

def one_hot_encoder(data=[]):
    '''Transfer categorical data to numerical data based on one hot encoding approach.

    Parameters
    ----------
    data : one dimension list

    Return
    ------
    The list of the encoded data based on one hot encoding approach
    '''
    # Since scikit-learn's OneHotEncoder only accepts numerical data, use LabelEncoder to transfer the
    # categorical data to numerical by using simple encoding approach.
    # For example, t -> 0; f -> 1
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    LABEL_ENCODER = preprocessing.LabelEncoder()
    numerical_data = LABEL_ENCODER.fit_transform(data)
    two_d_array = [[item] for item in numerical_data]

    # Use scikit-learn OneHotEncoder to encode the A9 feature
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(two_d_array)
    return encoder.transform(two_d_array).toarray()

if __name__ == '__main__':
    # Download the Japanese Credit Data Set from http://archive.ics.uci.edu/ml/datasets/Japanese+Credit+Screening
    URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
    urllib.request.urlretrieve(URL, 'crx.data')
    # Use pandas.read_csv module to load adult data set
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

    # Use one-hot encoding to transfer the A9 attribute of the Japanese Credit Data Set
    # A9:   t, f.
    # The encoded output may look like
    # t -> [0, 1]
    # f -> [1, 0]

    crx_data = pd.read_csv('crx.data', header=None)
    A9 = crx_data.iloc[:, 8].values
    A9_encoded = one_hot_encoder(A9)

    for index in range(len(A9_encoded)):
        print(str(A9[index]) + ' -> ' + str(A9_encoded[index]))