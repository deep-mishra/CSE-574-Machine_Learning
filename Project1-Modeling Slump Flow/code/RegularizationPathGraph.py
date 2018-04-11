# -*- coding: utf-8 -*-
"""

Task: Graph the regularization paths for Ridge and Lasso regression on
      Slump Flow of Concrete

Created on Mon Mar 26 02:48:21 2018
@author: Deep
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

""" Load the data """
baseDataDir = 'C:/STUDY/COURSE/SEMII/ML/PA/'
df = pd.read_csv(baseDataDir + 'slump_testdata.csv', sep=',')

y = df.iloc[:, 9]
X = df.iloc[:, 1:8]  # index 0 is 'No' field in my file so eliminating that

"""Find the CV fold which produced minimum MSE"""

kf = KFold(n_splits=5)

ridge_cv_mse = []
lasso_cv_mse = []

# Extract he Fold index which gives least mean square
ridge_best_train_index = None
lasso_best_train_index = None



# Split data into Training and Test part (85 x 1 vector y_train; 85 x 7 matrix X_train; 18 x 1 vector y_test; and, 85 x 7 x_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=18, train_size=85, random_state= 43)

# X_train and y_train further split on each iteration for 5-fold validation
for train_index, test_index in kf.split(X_train):
    X_train1, X_cv = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train1, y_cv = y_train.iloc[train_index], y_train.iloc[test_index]

   # Train the ridge model and save the fold if MSE for the model is least
    rg_model = Ridge(alpha=20)
    rg_model.fit(X_train1, y_train1)
    mse1 = mean_squared_error(y_cv, rg_model.predict(X_cv))
    if ridge_cv_mse == [] or mse1 < min(ridge_cv_mse):
        best_rg_mse = mse1
        ridge_best_train_index = train_index
    ridge_cv_mse.append(mse1)

    # Train the Lasso model and save the fold if MSE for the model is least
    lasso_model = Lasso(alpha=20)
    lasso_model.fit(X_train1, y_train1)
    mse2 = mean_squared_error(y_cv, lasso_model.predict(X_cv))
    if lasso_cv_mse == [] or mse2 < min(lasso_cv_mse):
        best_lasso_mse = mse2
        lasso_best_train_index = train_index
    lasso_cv_mse.append(mse2)

# Print the MSE for the ridge best model from CV
print("Best Ridge Model was produced on the the CV fold with indices:\n", ridge_best_train_index)
print()

print("Best Lasso Model was produced on the the CV fold with indices:\n", lasso_best_train_index)
print()


# Extract the best folds for these regression
X_train_ridge, y_train_ridge = X_train.iloc[ridge_best_train_index], y_train.iloc[ridge_best_train_index]
X_train_lasso, y_train_lasso = X_train.iloc[lasso_best_train_index], y_train.iloc[lasso_best_train_index]

print("We will use these folds to identify the regularization path on the coefficient")


# Define the range of alphas to find the regularization path
# Keeping a very high range in terms of log so that we get a better graph
# For small range I am not able to find the variations much
alphas = np.logspace(0, 8, 200)


# Variables to hold coefficients
ridge_coefs = []
lasso_coefs = []


# Perform the ridge and lasso regression for a range of alpha
for a in alphas:
    # Ridge coefficient calculation
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train_ridge, y_train_ridge)
    ridge_coefs.append(ridge.coef_)
    # Lasso coefficient calculation
    lasso = Lasso(alpha=a, fit_intercept=False)
    lasso.fit(X_train_lasso, y_train_lasso)
    lasso_coefs.append(lasso.coef_)

# PLOT THE GRAPH FOR REGULARIZATION PATH WITH RESPECT TO FEATURES WEIGHT
fig = plt.figure()

# Plot Regularization path for Ridge Regression
plt.subplot(221)
ax = plt.gca()
ax.plot(alphas, ridge_coefs, '-')
plt.title('Ridge coefficients regularization path')
plt.xlabel('alpha')
plt.ylabel('features weight')
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.axis('tight')
plt.legend(X.columns.values, loc=2)

# Plot Regularization path for Lasso Regression
plt.subplot(222)
ax = plt.gca()
ax.plot(alphas, lasso_coefs, '-')
plt.title('Lasso coefficients regularization path')
plt.xlabel('alpha')
plt.ylabel('features weight')
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.axis('tight')
plt.legend(X.columns.values, loc=2)
plt.show()





