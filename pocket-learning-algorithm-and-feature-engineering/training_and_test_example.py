# https://archive.ics.uci.edu/ml/datasets/Iris
# Attribute Information:
# 0. sepal length in cm 
# 1. sepal width in cm 
# 2. petal length in cm 
# 3. petal width in cm 
# 4. class: 
# -- Iris Setosa 
# -- Iris Versicolour 
# -- Iris Virginica

# pandas is an open source library providing high-performance, 
# easy-to-use data structures and data analysis tools. http://pandas.pydata.org/
import pandas
import numpy as np

# matplotlib is a python 2D plotting library which produces publication quality 
# figures in a variety of hardcopy formats and interactive environments across platforms.
# http://matplotlib.org/2.0.0/index.html
from matplotlib import pyplot

from perceptron_classifier import PerceptronClassifier
from sklearn.model_selection import train_test_split

import seaborn as sns
import urllib.request

sns.set()

# Download Iris Data Set from http://archive.ics.uci.edu/ml/datasets/Iris
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urllib.request.urlretrieve(url, 'iris.data')
# use pandas' read_csv function to read iris.data into a python array. 
# Note: the iris.data is headerless, so header is None.
IRIS_DATA = pandas.read_csv('iris.data', header=None)

LABELS = IRIS_DATA.iloc[50:150, 4].values
DATA = IRIS_DATA.iloc[50:150, [0, 2]].values

DATA_TRAIN, DATA_TEST, LABELS_TRAIN, LABELS_TEST = train_test_split(DATA, LABELS, test_size=0.25, random_state=1000)

perceptron_classifier = PerceptronClassifier(2, ('Iris-versicolor', 'Iris-virginica'))
perceptron_classifier.train(DATA_TRAIN, LABELS_TRAIN, 100)
print(perceptron_classifier.misclassify_record)

result = perceptron_classifier.classify(DATA_TEST)

misclassify = 0
for predict, answer in zip(result, LABELS_TEST):
    if predict != answer:
        misclassify += 1
    print(str(predict) + " " + str(answer))
print(misclassify)
print("Accuracy rate: " + str((len(result) - misclassify) / len(result)) + "%") 