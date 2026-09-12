# K-Nearest Neighbours Classification Algorithm
"""
File for the K-Nearest-Neighbours classification algorithm 

used by : data_analysis.ipynb
imports and calls from preprocessing.py

"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import get_features_and_labels, get_cv_splitter, check_class_balance


def train_knn(X, y, cv=None, param_grid=None):
    """
    train and tune the hyperparameters for K-NearestNeighbour classifier using GridSearchCV

    Parameters:
    X, y: feature matrix and labels from preprocessing.py get_features_and_labels()
    cv : a cross validation splitter, using preprocessing.py get_cv_splitter() : passing the same splitter to all algorithms ensures models are using identical data
    param_grid: dictionary of hyperparameters to search


    Returs:
    dictionary with : 
        best_estimator : fitted Pipeline (scalar + tuned KNeighbours)
        best_parameters : dictionary of best performing hyperparameters
        best_score : mean cv accuracy
        grid : fill fitted GridSearchCV object : used to call grid.cv_results_

    """

    if cv is None:
        cv = get_cv_splitter()

    if param_grid is None:
        param_grid = {
        "knn__n_neighbors" : [3, 5, 7, 9, 11, 15],
        "knn__weights" : ["uniform", "distance"],
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()), 
        ("knn", KNeighborsClassifier()),
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    grid.fit(X, y)

    return {
        "best_estimator" : grid.best_estimator_,
        "best_parameters" : grid.best_params_,
        "best_score" : grid.best_score_,
        "grid" : grid,
    }


def knn_summary(results, X, y, cv=None):
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best CV accuracy: {results['best_score']:.3f}")
    print()

    
    if cv is None:
        cv=get_cv_splitter()

    y_prediction = cross_val_predict(results["best_estimator"], X, y, cv=cv)
    print("Clasification Report summary: ")
    print(classification_report(y, y_prediction))
    print("Confusion Matrix: ")
    print(confusion_matrix(y, y_prediction))

    return y_prediction
