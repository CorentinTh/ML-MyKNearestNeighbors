# K-Nearest-Neighbors classfier
# Home-made by CorentinTh
# https://github.com/CorentinTh/ML-MyKNearestNeighbors

import math
import sys
import operator

# Compute a multi dimentional euclidean distance 
def euclideanDistance(data1, data2):
	if len(data1) != len(data2):
		sys.exit("Error euclideanDistance : input doesn't have the same lenght")

	dist = 0
	for i in range(len(data1)):
		dist += pow(data1[i] - data2[i], 2)
	return math.sqrt(dist)

class MyKNN():
	# The constructor method
	def __init__(self):
		# Initialization of the k variable it representes
		# the amount of neighbors we will check
		self.k = 5

	# Use this method to train the classifier. The more
	# you train it, the better the prediction will be.
	# 
	# x_train is an array of features (like [[1, 2], [5, 9], [6, 8], [2, 3]])
	# y_train is an array of labels   (like [  'a' ,  'b'  ,  'b'  ,  'a'  ])
	#         labels index must match the corresponding feature index
	def train(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	# Use this method to predit output 
	def predict(self,x_test):
		predictions = []

		for j in range(len(x_test)):
			neighbors = self.getNeighbors(x_test[j])
			labels = {}
			for i in range(self.k):
				label = self.y_train[neighbors[i]]
				if label in labels:
					labels[label] += 1
				else:
					labels[label] = 1
			sortedLabels = sorted(labels.iteritems(), key=operator.itemgetter(1), reverse=True)
			predictions.append(sortedLabels[0][0])
		return predictions

	def getNeighbors(self, point):
		distances = []

		for i in range(len(self.x_train)):
			dist = euclideanDistance(point, self.x_train[i])
			distances.append((dist, i))

		distances.sort(key=operator.itemgetter(0))
		neighbors = []

		for i in range(self.k):
			neighbors.append(distances[i][1])

		return neighbors


def getAccuracy(y_test, y_good):
	good = 0
	for x in range(len(y_test)):
		if y_test[x] == y_good[x]:
			good += 1
	return (good / float(len(y_test)))*100.0

if __name__ == '__main__':
	classifier = MyKNN()
	train = [[1,1,1],[5,5,5],[8,8,8],[3,3,3],[10,10,10],[11,11,11]]
	y=['a','a','b','a','b','b']
	test = [[9,9,9], [1,1,1]]

	classifier.train(train, y)
	print(classifier.predict(test))
