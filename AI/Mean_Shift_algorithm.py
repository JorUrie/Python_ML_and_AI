# Mean Shift is a powerful algorithm used in unsupervised learning
# It is a non-parametric algorithm used frequently for clustering
# It is non-parametric because it does not make any assumptions about the underlying distributions
# It's also called hierarchical clustering or mean shift cluster analysis
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
# This line will help in generating the two-dimensional dataset, containing four blobs, by using make_blob from the sklearn.dataset package
style.use('ggplot')
from sklearn.datasets._samples_generator import make_blobs
centers = [[2, 2],[4, 5], [3, 10]]
X, _ = make_blobs(n_samples = 500, centers = centers, cluster_std = 1)
# Visualizing the dataset
plt.scatter(X[:, 0], X[:, 1])
plt.show()