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

import seaborn as sns
import urllib.request

sns.set()

# Download Iris Data Set from http://archive.ics.uci.edu/ml/datasets/Iris
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urllib.request.urlretrieve(url, 'iris.data')
# use pandas' read_csv function to read iris.data into a python array. 
# Note: the iris.data is headerless, so header is None.
IRIS_DATA = pandas.read_csv('iris.data', header=None)

VERSICOLOR = IRIS_DATA.iloc[50:100, [0, 2]].values
VIRGINICA = IRIS_DATA.iloc[100:150, [0, 2]].values

pyplot.scatter(VERSICOLOR[:, 0], VERSICOLOR[:, 1], color='blue', marker='x', label='versicolor')
pyplot.scatter(VIRGINICA[:, 0], VIRGINICA[:, 1], color='green', marker='v', label='virginica')

pyplot.xlabel('sepal length')
pyplot.ylabel('petal length')
pyplot.legend(loc='upper left')
pyplot.show()

VERSICOLOR_LABEL = IRIS_DATA.iloc[50:100, 4].values
VIRGINICA_LABEL = IRIS_DATA.iloc[100:150, 4].values
LABELS = np.append(VERSICOLOR_LABEL, VIRGINICA_LABEL)
SAMPLES = np.append(VERSICOLOR, VIRGINICA, axis=0)

perceptron_classifier = PerceptronClassifier(2, ('Iris-versicolor', 'Iris-virginica'))
perceptron_classifier.train(SAMPLES, LABELS, 100)
pyplot.plot(perceptron_classifier.misclassify_record, color='purple', label='Perceptron (Non-normalized)')

# Plot the error rate and show it never converges
pyplot.xlabel('number of iteration')
pyplot.ylabel('number of misclassification')
pyplot.legend(loc='lower right')
pyplot.show()