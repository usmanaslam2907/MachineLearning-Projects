# Import Modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading datasets
iris = datasets.load_iris()

# Extract features and labels
features = iris.data
labels = iris.target

# Printing description features and labels
# print(iris.DESCR)
# print(features[0], labels[0])
# print(features)

# Make the classifier model and then we will train it for fitting classifier into data for prediction
classifier = KNeighborsClassifier()
classifier.fit(features, labels)

# taking prediction
prediction = classifier.predict([[5.1, 1.2, 1.1, 1]])  # apply classifiy on four features
print(prediction)
