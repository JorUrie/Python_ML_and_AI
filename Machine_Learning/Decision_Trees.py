import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Read data using pandas dataframe
my_data = pd.read_csv('drug200.csv', delimiter = ',')
#print(my_data[0:5])

my_data.shape

#Pre-processing
# x is the Feature Matrix
x = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
#print(x[0:5])

# Sklearn Decision Trees don't handle categorical variables
# Convert features to numerical values
from sklearn import preprocessing
# Sex
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
x[:, 1] = le_sex.transform(x[:, 1])
# BP
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
x[:, 2] = le_BP.transform(x[:, 2])
# Cholesterol
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
x[:, 3] = le_Chol.transform(x[:, 3])

#print(x[0: 10])

# Fill the target variable
y = my_data['Drug']
# print(y[0: 10])

# We'll be using a train/test split on our decision tree
from sklearn.model_selection import train_test_split
x_trainset, x_testset, y_trainset, y_testset = train_test_split(x, y, test_size = 0.3, random_state = 3)

# We will first create an instance of the DecisionTreeClassifier called drugTree
# Inside of the classier, specify criterion = 'entropy' so we can see the information gain of each node
drugTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
# print(drugTree)

# We'll fit the data with the training feature natrix x_trainset and training response vector y_trainset
drugTree.fit(x_trainset, y_trainset)
# print(drugTree)

# Prediction
# Let's make some predictions on the testing dataset and store it into a variable called predTree
predTree = drugTree.predict(x_testset)

# We print out predTree and y_testset if we want to visually compare the prediction to the actual values
# print(predTree)
# print(y_testset)

# Evaluation
# We check the accuracy of our model
from sklearn import metrics
print('Decision tree accuracy: ', metrics.accuracy_score(y_testset, predTree))

# This part only work out with Anaconda
# Visualizing the decision tree
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
filename = 'drugtree.png'
dot_data = StringIO()
featureNames = my_data.columns[0:5]
export_graphviz(drugTree, feature_names = featureNames, out_file = dot_data,  filled = True, rounded = True, special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("dectree.png")
img = mpimg.imread("dectree.png")
plt.figure(figsize = (100, 200))
plt.imshow(img, interpolation = 'nearest')
