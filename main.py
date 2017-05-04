# K-Nearest-Neighbors classfier
# Home-made by CorentinTh
# https://github.com/CorentinTh/ML-MyKNearestNeighbors

# To get execution time
import time
start_time = time.time()

from sklearn import datasets
from MyKNN import *

# We get the Iris dataset from the sklearn library
iris = datasets.load_iris()

# Features in x
# and labels in y
x = iris.data
y = iris.target

# We split our Iris dataset in two, one half to train our classifier, 
# and the other half to test its accuracy
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

my_classifier = MyKNN()
my_classifier.k = 5

# We train the classifier with the first half of our data
my_classifier.train(x_train, y_train)
# We test it with features of the other half
predictions = my_classifier.predict(x_test)
# We compute the accuracy by comparing the output results with reals ones
duration = time.time()-start_time
print("Precision : {}% en {} sec".format(getAccuracy(y_test, predictions), duration))
