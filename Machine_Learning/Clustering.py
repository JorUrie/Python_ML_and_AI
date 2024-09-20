import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# K-Means on a randomly generated dataset
np.random.seed(0)

# We'll be make random clusters of points by using the make_blobs class
X, y = make_blobs(n_samples = 5000, centers = [(4, 4), (-2, -1), (2, -3), (1, 1)], cluster_std = 0.9)

# Displaying the scatter plot of the randomly generated data
plt.scatter(X[:, 0], X[:, 1], marker = '.')
plt.show()

# Setting up K-Means
'''The KMeans class has many parameters that can be used, but we will be using these three
init: Initialization method of the centroids
k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence
n_clusters: The number of clusters to form as well as the number of centroids to generated
Value will be 4 because there are 4 centres 
n_init: Number of times the k-means algorithm will be run with different centroid seeds. The value will be 12'''
# Normalized our dataset
k_means = KMeans(algorithm = 'lloyd', n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 12).fit(X)
# print(k_means)

# Let's grab the labels for each point in the model
k_means_labels = k_means.labels_
# print(k_means_labels)

# We also get the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
# print(k_means_cluster_centers)

# Creating the VisualPlot
# Initialize the plot with the specified dimensions
fig = plt.figure(figsize = (6, 4))
# colour used = color map: based on the number of labels there are
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
# Create a plot
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len([[4, 4], [-2, 1], [2, -3], [1, 1]])), colors):
    my_members = (k_means_labels, k)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show


