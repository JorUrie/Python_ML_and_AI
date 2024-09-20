import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Reading the data
df = pd.read_csv('FuelConsumption.csv')
# take a look at the dataset
df.head()
#print(df)
# summarize the data
df.describe()
#print(df)
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(4)
#print(cdf)
# plot the features
viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
plt.show()
# to see how linear is their relation
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()

''' Lets split our dataset into train and test sets, 80% of the entire data for
training, and the 20% for testing
We create a mask to select random rows using np.random.rand() function.'''
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
# Finally creating a Simple Regression Model
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()

# Using sklearn package to model data
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# We can plot the fit line over the data
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()

''' We compare the actual values and predicted values to calculate
the accuracy of a regression model 
Evaluation metrics provide a key role in the development of a model,
as it provides insight to areas that require improvement'''

from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
print('Mean absolute error: %2f' % np.mean((test_y_-test_y)))
print('Residual sum of squares (MSE): %2f' % np.mean((test_y_-test_y)**2))
print('R2-score: %2f' % r2_score(test_y_, test_y))
