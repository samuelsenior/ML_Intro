import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# rather than using a decision tree.
from sklearn.neighbors import KNeighborsClassifier

"""
Note - We have no model solution for task 3 - as this is a more experimental
task where you try things out yourself. Don't be afraid to ask for help during
the practical however. 
"""

def iris_dataset_2():
	iris = datasets.load_iris()

	X = iris.data
	y = iris.target

	"""
	Here we split Iris (150 data) into 75 training and 
	75 test datasets. since test_size = 0.5
	"""
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

	classifier = tree.DecisionTreeClassifier()
	classifier.fit(X_train, y_train)

	predictions = classifier.predict(X_test)

	print("Decision Tree: {}".format(accuracy_score(y_test, predictions)))
	return

def iris_dataset_3():

	iris = datasets.load_iris()

	X = iris.data
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

	classifier = tree.DecisionTreeClassifier()
	classifier.fit(X_train, y_train)

	predictions = classifier.predict(X_test)

	print("Decision Tree: {}".format(accuracy_score(y_test, predictions)))

	classifier_2 = KNeighborsClassifier()
	classifier_2.fit(X_train, y_train)

	predictions_2 = classifier_2.predict(X_test)

	print("K-Neighbours: {}".format(accuracy_score(y_test, predictions_2)))



