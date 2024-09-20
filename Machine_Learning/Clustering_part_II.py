# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data set
cust_df = pd.read_csv("Cust_Segmentation.csv")
# print(cust_df)

# Let's drop the feature and run clustering
cust_df = pd.read_csv("Cust_Segmentation.csv")
# print(cust_df)

# Pre-processing
df = cust_df.drop('Address', axis = 1)
df.head()

# Normalizing over the standard deviation
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
# print(Clus_dataSet)

# Modeling
# Let's apply k-means on our dataset, and take a look at cluster labels
clusterNum = 3
k_means = KMeans(n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
# print(labels)

# Insights
# We assign the labels to each row in dataframe
df['Clus_km'] = labels
# print(df.head(5))

# We can check the centroid values by averaging the features in each cluster
plt.scatter(X[:, 0], X[:, 3], c = labels.astype(np.float64), alpha = 0.5)
plt.xlabel('Age', fontsize = 18)
plt.ylabel('Income', fontsize = 16)
plt.show()

# Let's look at the distribution of costumers based on their age and income
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect = [0, 0, .95, 1], elev = 48, azim = 134)
plt.clf()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c = labels.astype(np.float64))