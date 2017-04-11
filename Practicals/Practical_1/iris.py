import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

"""
Continuing to make classifiers using Decision Trees
"""

def iris_dataset():
	iris = datasets.load_iris()
	"""
	At this stage, iris contains our features and labels
	"""

	# lets say we want to set 3 random indices (iris dataset is (150,5))
	test_idx = np.random.randint(0,150,3)

	"""now create our training features and labels by removing the elements
	from that index"""
	train_X = np.delete(iris.data, test_idx, axis=0)
	train_y = np.delete(iris.target, test_idx, axis=0)

	"""now create our testing features and labels by using our testing index"""
	test_X = iris.data[test_idx]
	test_y = iris.target[test_idx]

	# create the decision tree and fit using TRAINING DATA
	clf = tree.DecisionTreeClassifier()
	clf.fit(train_X, train_y)

	# see if our test labels match the prediction
	print(test_y)
	print(clf.predict(test_X))

	return


def iris_dataset_2():

	"""
	Task 1 - Implement the above function (iris_dataset) using the train_test_split()
	function instead of just randomly getting some integers. 

	the import package you want is - 'from sklearn.model_selection import train_test_split'

	Then look up the function, its inputs and outputs, and use it to generate your
	training and testing data. Split the dataset 50/50 (75 training, 75 testing).
	Try different splits (75/25) etc, see which one gets you a better accuracy.

	In addition, we've added a quantitative way of comparing the accuracy in this 
	example using the accuracy_score() method, simply put your y_test and predictions
	data into it, and it will return a percentage accuracy. Values closer to 100% are
	more accurate.
	"""
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target

	# implement here --------------------------




	# -----------------------------------------

	acc_score = accuracy_score(y_test, predictions)
	print("Decision Tree: {}".format(acc_score))

	pass


def iris_dataset_3():
	"""
	Task 2 - Using the same dataset, use a different classifier than the Decision Tree.
	In this task, you should try out the K-Nearest Neighbours Classifier.

	the import package you want is - 'from sklearn.neighbors import KNeighborsClassifier'

	Once you have the KNeighboursClassifier working, compare the accuracy score (from
	task 1) to using the Decision Tree, is there much difference? Which is better? Figure out the
	advantages and disadvantages of both. 

	This solution is much simplier than it sounds, don't overthink this!
	"""

	pass


def challenge_dataset():
	"""
	Task 3 - Using what you have learnt from this dataset, apply it to another dataset.

	Go to http://scikit-learn.org/stable/datasets/ and select the load_digits() 
	dataset, as it is also a classification problem. If you really want a challenge, 
	select the diabetes dataset.

	Like the iris dataset, these sets can be retrieved through the sklearn.datasets class
	by one simple function call to the relevant dataset.

	Good Luck!
	"""


if __name__ == '__main__':
	iris_dataset()


