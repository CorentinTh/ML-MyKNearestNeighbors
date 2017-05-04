# ML-MyKNearestNeighbors
## Overview
The `MyKNN.py` file contain my home-made version of the k-Nearest-Neighbors classifier.

By default `k = 5`, it means that the classifier will checks for the closest 5 neightbors of the data you want to test.

## Dependencies
The classifier only uses basic library (`math`, `sys` and `operator`), so you don't need dependencies to use it.

The main file (`main.py`) uses sklearn to get datasets, but you don't need it if you want to use just the classifier. Run this command to get **sklearn** :
```
pip install scikit-learn
```
## Run the script
You can run this script in terminal with this command line :
```
python main.py
```
## Use the classifier
First create the classifier.s
```python
from MyKNN import *
classifier = MyKNN()

# You can change the value of k by this
classifier.k = 5
```
Then, train it.
```python
# x_train is an array of features (like [[1, 2], [5, 9], [6, 8], [2, 3]])
# y_train is an array of labels   (like [  'a' ,  'b'  ,  'b'  ,  'a'  ])
# Labels index must match the corresponding feature index
classifier.train(x_train, y_train)
```
And now, you can predict some output.
```python
# x_test is an array you want to get the label (like [[6, 8], [0, 2]])
predictions = classifier.predict(x_test)
print(predictions)
# Display : ['b', 'a']
```
