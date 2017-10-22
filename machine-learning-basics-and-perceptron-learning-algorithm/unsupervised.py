# https://archive.ics.uci.edu/ml/datasets/Iris
# Attribute Information:
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm 
# 5. class: 
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
# use pandas' read_csv function to read iris.data into a python array. 
# Note: the iris.data is headerless, so header is None.
iris_data = pandas.read_csv('iris.data', header=None)

unlabeled_data = iris_data.iloc[:, [0, 2]].values

pyplot.scatter(unlabeled_data[:, 0], unlabeled_data[:, 1], color='green', marker='o')

pyplot.xlabel('sepal length')
pyplot.ylabel('petal length')
pyplot.legend(loc='upper left')
pyplot.show()
