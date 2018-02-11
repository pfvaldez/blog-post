from sklearn import preprocessing
import csv
import numpy as np
import pandas as pd
import urllib.request
from collections import Counter

def imputer_by_most_frequent(missing_values=np.nan, data=[]):
    '''Input missing value by frequency, i.e., the value appeared most often.

    Parameters
    ----------
    missing_values :
        The missing value can be np.nan, '?', or whatever character which indicates missing value.

    data : one dimension list

    Return
    ------
    The list of the encoded data based on one hot encoding approach
    '''
    # Find the value appeared most often by using Counter.
    most = Counter(data).most_common(1)[0][0]
    complete_list = []
    for item in data:
        if item is missing_values:
            item = most
        complete_list.append(item)
    return complete_list

if __name__ == '__main__':
    # Download Japanese Credit Data Set from http://archive.ics.uci.edu/ml/datasets/Japanese+Credit+Screening
    URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
    urllib.request.urlretrieve(URL, 'crx.data')
    # Use pandas.read_csv module to load adult data set
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

    # Use the input by most frequent approach to input the missing values in A1
    # A1:   b, a
    #
    # Use the input by mean number approach to input the missing values in A2
    # A2:   continuous

    crx_data = pd.read_csv('crx.data', header=None)
    # Since the Japanese Credit Data Set uses '?' to denote missing, replace it to np.nan.
    # scikit-learn's Imputer only accepts np.nan or integer, therefore, convert '?' to np.nan.
    # This transformation is for A2 which uses scikit-learn's Imputer.
    # For A1 which uses imputer_by_most_frequent(), this transformation is not necessary.
    crx_data.replace('?', np.nan, inplace=True)

    A1_no_missing = imputer_by_most_frequent(np.nan, crx_data.iloc[:, 0].values)
    print(A1_no_missing)

    # Use scikit-learn Imputer to input missing values by mean number.
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
    imputer = preprocessing.Imputer(missing_values=np.nan, strategy='mean', axis=0)
    # Convert to two-dimension list, since Imputer only accepts two dimensions list.
    A2_two_d = np.array([[item] for item in crx_data.iloc[:, 1].values])
    A2_no_missing = imputer.fit_transform(A2_two_d)
    print(A2_no_missing)
