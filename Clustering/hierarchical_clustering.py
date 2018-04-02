#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Hierarchical clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
plt.figure()
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
Y_pred = clustering.fit_predict(X)

# Visualizing the clusters
plt.figure()
plt.scatter(X[Y_pred == 0, 0], X[Y_pred == 0, 1], s = 10, c = 'red', label = 'Stingy')
plt.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], s = 10, c = 'blue', label = 'Standard')
plt.scatter(X[Y_pred == 2, 0], X[Y_pred == 2, 1], s = 10, c = 'green', label = 'Target')
plt.scatter(X[Y_pred == 3, 0], X[Y_pred == 3, 1], s = 10, c = 'cyan', label = 'Careless')
plt.scatter(X[Y_pred == 4, 0], X[Y_pred == 4, 1], s = 10, c = 'magenta', label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual income (K$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()