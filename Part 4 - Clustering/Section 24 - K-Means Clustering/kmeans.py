#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# K-means clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using the elbow method to find the optimal # of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    clustering = KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    clustering.fit(X)
    wcss.append(clustering.inertia_)

plt.figure()
plt.plot(range(1,11),wcss)
plt.title('The Elbow method')
plt.xlabel('# of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the dataset with optimal # of clusters
clustering = KMeans(n_clusters=5, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
Y_pred = clustering.fit_predict(X)

# Visualizing the clusters
plt.figure()
plt.scatter(X[Y_pred == 0, 0], X[Y_pred == 0, 1], s = 10, c = 'red', label = 'Stingy')
plt.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], s = 10, c = 'blue', label = 'Standard')
plt.scatter(X[Y_pred == 2, 0], X[Y_pred == 2, 1], s = 10, c = 'green', label = 'Target')
plt.scatter(X[Y_pred == 3, 0], X[Y_pred == 3, 1], s = 10, c = 'cyan', label = 'Careless')
plt.scatter(X[Y_pred == 4, 0], X[Y_pred == 4, 1], s = 10, c = 'magenta', label = 'Sensible')
plt.scatter(clustering.cluster_centers_[:,0], clustering.cluster_centers_[:,1], s = 30, c='black', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual income (K$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()