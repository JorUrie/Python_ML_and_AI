from sklearn.datasets import load_breast_cancer
# Loading the dataset
data = load_breast_cancer()
# We can create new variables for each important set of information and assign the data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print(label_names)
print(labels[0])
# Produce the feature names and feature values
print(feature_names[0])
print(features[0])

# Naive Bayes: Bayes theorem describes the probability of an event occurring based on different condition that are related to this event
# There are three types of Naives Bayes models named: Gaussian, Multinomial and Bernoulli
from sklearn.model_selection import train_test_split
# Here we use 40% of the data for testing and the remaining data would be used for training the model
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.40, random_state = 42)
from sklearn.naive_bayes import GaussianNB
# To initialize the model, the following command is used:
gnb = GaussianNB()
# And gnb.fit() is used to train the model by fitting it to the data
model = gnb.fit(train, train_labels)
# Let's evaluate the model by making prediction on the test data and it can be done as follows:
preds = gnb.predict(test)
print(preds)
# So by comparing the two arrays namely test_labels and preds, we can find out the accuracy of our model
# Here we use the accuracy_score() function to determine the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, preds))