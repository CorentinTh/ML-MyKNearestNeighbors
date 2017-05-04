# K-Nearest-Neighbors classfier
# Home-made by CorentinTh
# https://github.com/CorentinTh/ML-MyKNearestNeighbors

import time
start_time = time.time()

from sklearn import datasets
from MyKNN import *

iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

#from sklearn.neighbors import KNeighborsClassifier
my_classifier = MyKNN()
my_classifier.k = 5
my_classifier.train(x_train, y_train)

predictions = my_classifier.predict(x_test)

duration = time.time()-start_time
print("Precision : {}% en {} sec".format(getAccuracy(y_test, predictions), duration))
