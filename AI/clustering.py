# This technique is used yo analyze data and find clusters within that data
# The main goal of clustering is to group the data on the basis of similarity and dissimilarity
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs
# This line will help on generating the 2D dataset, containing four blobs
X, y_true = make_blobs(n_samples = 500, centers = 4, cluster_std = 0.40, random_state = 0)
# Visualizing and initializing
plt.scatter(X[:, 0], X[:, 1], s = 50)
plt.show()
# We are initializing kmeans to be the KMeans algorithm, with the required parameter of how many clusters
kmeans = KMeans(n_clusters = 4)
# We need to train the K-means model with the input data
# The next code will help us plot and visualize the machine's findings based on our data, and the fitment according to the number of clusters that are to be found
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 50, cmap = 'viridis')
centers = kmeans.cluster_centers_
# ----------------------
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.5)
plt.show()