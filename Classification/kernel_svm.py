#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,-1].values

# Splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Create and fit classifier to training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(X_train, Y_train)

# Predict test set results
Y_pred = classifier.predict(X_test)
Y_prob = classifier.decision_function(X_test)

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# Visualising the classification results method
def visualizeClassification(ClassifierObject, X, Y, Color1, Color2, Title):
    from matplotlib.colors import ListedColormap
    X_set, Y_set = X, Y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.figure()
    plt.contourf(X1, X2, ClassifierObject.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.20, cmap = ListedColormap((Color1, Color2)))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(Y_set)):
        plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                        c = ListedColormap((Color1, Color2))(i), label = j)
    plt.title(Title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
    return
    
# Plot ROC curve method
def plotROCcurve(Ypred, Yprob, LW, ROC_Color):
    from sklearn.metrics import roc_curve, auc
    [FPR, TPR, thresholds] = roc_curve(Ypred, Yprob)
    AUC = auc(FPR, TPR)
    plt.figure()
    plt.plot(FPR, TPR, color=ROC_Color, lw = LW, label = 'ROC curve (AUC = %0.2f)' %AUC)
    plt.plot([0, 1], [0, 1], 'k--', lw = LW)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc = 'lower right')
    plt.show()
    return

# Visualize training and test rest results
visualizeClassification(classifier, X_train, Y_train, 'red', 'green', 'Kernel SVM classifier (training)')
visualizeClassification(classifier, X_test, Y_test, 'red', 'green', 'Kernel SVM classifier (test)')

#Visualize ROC curve
plotROCcurve(Y_test, Y_prob, 2, 'darkorange')