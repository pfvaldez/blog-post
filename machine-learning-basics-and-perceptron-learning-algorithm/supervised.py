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

# matplotlib is a python 2D plotting library which produces publication quality 
# figures in a variety of hardcopy formats and interactive environments across platforms.
# http://matplotlib.org/2.0.0/index.html
from matplotlib import pyplot

import seaborn as sns
import urllib.request

sns.set()

# Download Iris Data Set from http://archive.ics.uci.edu/ml/datasets/Iris
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urllib.request.urlretrieve(url, 'iris.data')
# use pandas' read_csv function to read iris.data into a python array. 
# Note: the iris.data is headerless, so header is None.
IRIS_DATA = pandas.read_csv('iris.data', header=None)

SETOSA = IRIS_DATA.iloc[0:50, [0, 2]].values
VERSICOLOR = IRIS_DATA.iloc[50:100, [0, 2]].values
VIRGINICA = IRIS_DATA.iloc[100:150, [0, 2]].values

pyplot.scatter(SETOSA[:, 0], SETOSA[:, 1], color='red', marker='o', label='setosa')
pyplot.scatter(VERSICOLOR[:, 0], VERSICOLOR[:, 1], color='blue', marker='x', label='versicolor')
pyplot.scatter(VIRGINICA[:, 0], VIRGINICA[:, 1], color='green', marker='v', label='virginica')

pyplot.xlabel('sepal length')
pyplot.ylabel('petal length')
pyplot.legend(loc='upper left')
pyplot.show()
