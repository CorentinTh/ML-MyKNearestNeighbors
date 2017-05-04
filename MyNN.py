import math
import sys

# Compute a multi dimentional euclidean distance 
def euclideanDistance(data1, data2):
	if len(data1) != len(data2):
		sys.exit("Error euclideanDistance : input doesn't have the same lenght")

	dist = 0
	for i in range(len(data1)):
		dist += pow(data1[i] - data2[i], 2)
	return math.sqrt(dist)

class MyNN():
	def train(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []

		for row in x_test:
			# We look for the closest point for each data you want to test
			label = self.closest_point(row)
			#And we put it in our output array
			predictions.append(label)

		return predictions

	def closest_point(self, row):
		# Initialization of the best distance and index with the plot 0
		best_dist = euclideanDistance(row, self.x_train[0])
		best_index = 0

		# We look for the closest point in x_train
		for i in range(1, len(self.x_train)):
			dist = euclideanDistance(row, self.x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]

