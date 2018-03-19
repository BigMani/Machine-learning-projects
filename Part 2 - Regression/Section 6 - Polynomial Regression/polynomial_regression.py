# Polynomial Regression

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

"""# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fit linear regression to dataset
from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()
LinReg.fit(X,Y)

# Fit polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
Degree = 3
PolyReg = PolynomialFeatures(degree = Degree)
X_poly = PolyReg.fit_transform(X)
LinReg2 = LinearRegression()
LinReg2.fit(X_poly, Y)

# Calculate R^2 and adjusted R^2
def calculateRSquared(y_orig, y_pred, X):
    Resid = sum((y_orig - y_pred)**2)
    Total = sum((y_orig - np.mean(y_orig))**2)
    R_squared = 1 - (float(Resid))/Total
    R_squared_adjusted = 1 - (1-R_squared)*(len(y_orig)-1)/(len(y_orig)-X.shape[1]-1)
    return R_squared, R_squared_adjusted

# Visualize regression fits
# Visualize linear fit
Y_Lin = LinReg.predict(X)
[R, R_adjust] = calculateRSquared(Y,Y_Lin,X)
plt.scatter(X, Y, color = 'red')
plt.plot(X,Y_Lin, color = 'blue')
plt.title('Salary vs Position (Linear Regression)')
plt.text(0.05, 0.85, '$R^2$ = {}'.format(round(R,2))+'\n$R^2$ adjusted = {}'.format(round(R_adjust,2)), transform=plt.gca().transAxes)
plt.xlabel('Position grade')
plt.ylabel('Salary ($)')

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
Y_Poly = LinReg2.predict(PolyReg.fit_transform(X_grid))
[R, R_adjust] = calculateRSquared(Y,LinReg2.predict(PolyReg.fit_transform(X)),X)
plt.figure()
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, Y_Poly, color = 'blue')
plt.title('Salary vs Position (Polynomial Regression, D = ' + str(Degree) + ')')
plt.text(0.05, 0.85, '$R^2$ = {}'.format(round(R,2))+'\n$R^2$ adjusted = {}'.format(round(R_adjust,2)), transform=plt.gca().transAxes)
plt.xlabel('Position grade')
plt.ylabel('Salary ($)')
plt.show()

# Predicting a new result with linear regression
print(LinReg.predict(6.5))

# Predicting a new result with polynomial regression
print(LinReg2.predict(PolyReg.fit_transform(6.5)))