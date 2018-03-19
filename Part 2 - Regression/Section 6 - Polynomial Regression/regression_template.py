#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: arunnemani
"""

# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

## Splitting the dataset into the training set and test set
# Only 10 observations, thus doesn't make sense to split dataset
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0)"""

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fit regression model to dataset
# Create regressor

# Predicting a new result with polynomial regression
Y_pred = regressor.predict(6.5)

# Visualize regression results
plt.figure()
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position grade')
plt.ylabel('Salary ($)')
plt.show()

# Visualize regression results with higher resolution curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.reshape(len(X_grid),1)
plt.figure()
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position grade')
plt.ylabel('Salary ($)')
plt.show()