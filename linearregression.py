import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.keys())
# print(diabetes.data)
# print(diabetes.DESCR)

# for feature like X
diabetes_X = diabetes.data
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-20:]

# for label like Y
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

# model creation
model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_y_predicted = model.predict(diabetes_X_test)

# mean square error found
print("Mean Squared Error is: ", mean_squared_error(diabetes_Y_test, diabetes_y_predicted))
# plt.plot(diabetes_X_test, diabetes_Y_test)
# plt.scatter(diabetes_X_test, diabetes_Y_test)
# plt.show()
