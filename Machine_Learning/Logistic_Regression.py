import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the data
Churn_df = pd.read_csv('ChurnData.csv')
Churn_df.head()
#print(Churn_df)

# Data pre-processing and selction
x = Churn_df = Churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
y = Churn_df = Churn_df['churn'].astype('int')
Churn_df.head()
#print(Churn_df)

# Define x, and y for our dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 40)

# Normalized our dataset
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
#print(x)

# We split our dataset into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 40)
print('Train set: ', x_train.shape, x_test.shape)
print('Test set: ', x_test.shape, y_test.shape)

# Modeling (Logistic Regression with Scikit-learn)
from sklearn.linear_model import LogisticRegression
RL = LogisticRegression(C = 0.01, solver = 'saga', max_iter = 999999999).fit(x_train, y_train)
# We can predict using our test set
yhat = RL.predict(x_test)
print(yhat)

# Jaccard index: The subset accuracy is 1.0; otherwise it is 0.0
from sklearn.metrics import jaccard_score
print('Jaccard Similarity Score: ',round(jaccard_score(y_test, yhat)*100,2),'%')

''' In specific case of binaty classifier, such as this example, we can interpret these numbers as the count of 
true positives, false positives, true negatives, and false negatives'''
# Compute confussion matrix
from sklearn.metrics import classification_report,confusion_matrix
cnf_matrix = confusion_matrix(y_test, yhat)
np.set_printoptions(precision = 2)
# Plot non-normalized confusion matrix
plt.figure()
#plt.confusion_matrix(cnf_matrix, classes = ['churn = 1', 'churne'], normalized = False, title = 'Confusion matrix')
# Calculate precision and recall of each label
print('Precision and recall: ', classification_report(y_test, yhat))