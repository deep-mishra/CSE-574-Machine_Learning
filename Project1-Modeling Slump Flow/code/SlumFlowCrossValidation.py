# -*- coding: utf-8 -*-
"""
Task:- 1.Split the slump flow dataset between test and training. The test data is further
       split between 5 folds to perform cross validation.
	   2. Compare Unregularized (Linear) and regularized (L2- Ridge & L1- Lasso)
	   regression.
			
Created on Sat Mar 14 22:55:04 2018
@author: Deep Narayan Mishra
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

""" Load the data """
df = pd.read_csv('slump_testdata.csv', sep=',')
y = df.iloc[:, 9]
X = df.iloc[:, 1:8]  # index 0 is 'No' field in my file so eliminating that

""" Identify the best co-efficient for ridge regularization which 
    produces minimum error and compare that with unregularized regression """

kf = KFold(n_splits=5)
linear_mse = []
lr_model_best = None

ridge_mse = []
rg_model_best = None

lasso_mse = []
lasso_model_best = None

# Iterate for 10 iterations
for i in range(10):
    print("Iteration ", i + 1)
    linear_cv_mse = []
    best_lm_mse = None
    ridge_cv_mse = []
    best_rg_mse = None
    lasso_cv_mse = []
    best_lasso_mse = None

    # Split data into Training and Test part (85 x 1 vector y_train; 85 x 7 matrix X_train; 18 x 1 vector y_test; and, 85 x 7 x_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=18, train_size=85)

    # X_train and y_train further split on each iteration for 5-fold validation
    for train_index, test_index in kf.split(X_train):
        X_train1, X_cv = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train1, y_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train the linear model and save if it is best model based on score
        lr_model = LinearRegression()
        lr_model.fit(X_train1, y_train1)
        mse1 = mean_squared_error(y_cv, lr_model.predict(X_cv))
        if linear_cv_mse == [] or mse1 < min(linear_cv_mse):
            best_lm_mse = mse1
            lr_model_best = lr_model
        linear_cv_mse.append(mse1)

        # Train the ridge model and save if it is best model
        rg_model = Ridge(alpha=20)
        rg_model.fit(X_train1, y_train1)
        mse2 = mean_squared_error(y_cv, rg_model.predict(X_cv))
        if ridge_cv_mse == [] or mse2 < min(ridge_cv_mse):
            best_rg_mse = mse2
            rg_model_best = rg_model
        ridge_cv_mse.append(mse2)

        # Train the Lasso model and save if it is best model
        lasso_model = Lasso(alpha=20)
        lasso_model.fit(X_train1, y_train1)
        mse3 = mean_squared_error(y_cv, lasso_model.predict(X_cv))
        if lasso_cv_mse == [] or mse3 < min(lasso_cv_mse):
            best_lasso_mse = mse3
            lasso_model_best = lasso_model
        lasso_cv_mse.append(mse3)

    ## Print the MSE for the linear best model from CV
    print("Best Linear model produced ", best_lm_mse, " MSE on CV")
    linear_predictions = lr_model_best.predict(X_test)
    linear_mse.append(mean_squared_error(y_test, linear_predictions))

    # Print the MSE for the ridge best model from CV
    print("Best Ridge model produced ", best_rg_mse, " MSE on CV")
    ridge_predictions = rg_model_best.predict(X_test)
    ridge_mse.append(mean_squared_error(y_test, ridge_predictions))
    print()

    # Print the MSE for the Lasso best model from CV
    print("Best Lasso model produced ", best_lasso_mse, " MSE on CV")
    lasso_predictions = lasso_model_best.predict(X_test)
    lasso_mse.append(mean_squared_error(y_test, lasso_predictions))
    print()

print("UNREGULARIZED REGRESSION RESULT ")
print("--------------------------")
print("Linear MSE:", linear_mse)
print("Linear MSE Average:", np.mean(linear_mse))
print("--------------------------")

print("RIDGE REGRESSION RESULT ")
print("--------------------------")
print("Ridge MSE ", ridge_mse)
print("Ridge MSE Average ", np.mean(ridge_mse))
print("--------------------------")

print("LASSO REGRESSION RESULT ")
print("--------------------------")
print("Lasso MSE ", lasso_mse)
print("Lasso MSE Average ", np.mean(lasso_mse))
print("--------------------------")


if np.mean(lasso_mse) < np.mean(ridge_mse) and np.mean(lasso_mse) < np.mean(linear_mse):
    print("Regularized (L1) performs better than others.")
elif np.mean(ridge_mse) < np.mean(linear_mse) and np.mean(ridge_mse) < np.mean(lasso_mse):
    print("Regularized (L2) performs better than others.")
elif np.mean(linear_mse) < np.mean(lasso_mse) and np.mean(linear_mse) < np.mean(ridge_mse):
    print("Unregularized performs better than others.")