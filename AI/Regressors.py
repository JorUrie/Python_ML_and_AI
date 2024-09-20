import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
# The source of our data is, where we have saved our data in the file
#input = 'C:/Users/myste/Documents/George/Mio/Cursos/AI/linear.txt'
data = np.loadtxt('linear.txt', delimiter = ',', dtype = int)
# print(data)
X, y = data[:, :-1], data[:, -1]
# The next step would be Split it into training and testing
num_training = int(0.8*len(X))
num_test = len(X) - num_training
# Training data
X_train, y_train = X[:num_training], y[:num_training]
# Test data
X_test, y_test = X[num_training:], y[num_training:]
# Create a linear regressor object and train it using training data:
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
# We need to do the prediction with the testing data
y_test_pred = regressor.predict(X_test)
# Make the plot
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_test, y_test_pred, color = 'black', linewidth = 4)
plt.xticks(())
plt.yticks(())
plt.show()