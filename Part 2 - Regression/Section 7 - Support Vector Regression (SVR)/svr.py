# SVR
# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
X_orig = X
Y = dataset.iloc[:, -1].values
Y_orig = Y
Y = np.reshape(Y,(-1,1))


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y)

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

# Calculate R^2 and adjusted R^2
def calculateRSquared(y_orig, y_pred, X):
    Resid = sum((y_orig - y_pred)**2)
    Total = sum((y_orig - np.mean(y_orig))**2)
    R_squared = 1 - (float(Resid))/Total
    R_squared_adjusted = 1 - (1-R_squared)*(len(y_orig)-1)/(len(y_orig)-X.shape[1]-1)
    return R_squared, R_squared_adjusted

# Predicting a new result
Y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_orig)))
[R, R_adjust] = calculateRSquared(Y_orig,Y_pred,X_orig)

# Visualising the Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.text(0.05, 0.85, '$R^2$ = {}'.format(round(R,2))+'\n$R^2$ adjusted = {}'.format(round(R_adjust,2)), transform=plt.gca().transAxes)
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

## Visualising the Regression results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, Y, color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#plt.title('Truth or Bluff (Regression Model)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')