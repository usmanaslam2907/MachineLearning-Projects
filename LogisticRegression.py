# train a logistic regression classifier to predict whether a flower is iris virginica or not

from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])

# take just 3rd column
# x = iris['data'][:, 3:]
# print(x)

# y = (iris["target"] == 2).astype(np.int_)
# print(y)

# Get features and labels for prediction
x = iris['data'][:, 3:]
y = iris['target'].astype(np.int_)

# Train the LogisticModel for prediction
classifier = LogisticRegression()
classifier.fit(x, y)
# just doing for one instances like we insert y as 2.6 its my think
prediction = classifier.predict(([[2.6]]))
# print(prediction)

# using matplotlib to plot visualization
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = classifier.predict_proba(x_new)
plt.plot(x_new, y_prob[:, 1], "g-", label="virgin ica")
plt.show()
