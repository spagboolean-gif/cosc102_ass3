"""
Support Vector Classifier
"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import get_cv_splitter


def train_svc(X, y, cv=None, param_grid=None):
    """
    Train and tune the SVC model.

    Params:
        X: Feature data.
        y: Activity labels.
        cv: Cross-validation splitter.
        param_grid: Parameters used for GridSearchCV.

    Returns:
        dict: Best model, parameters, score and GridSearchCV results.
    """

    if cv is None:
        cv = get_cv_splitter()

    # Parameters for linear and rbf kernels
    if param_grid is None:
        param_grid = [
            {
                "svc__kernel": ["linear"],
                "svc__C": [0.1, 1, 10, 100]
            },
            {
                "svc__kernel": ["rbf"],
                "svc__C": [0.1, 1, 10, 100],
                "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1]
            }
        ]

    # Scale data before running SVC
    svc_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC())
    ])

    grid = GridSearchCV(
        svc_pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    grid.fit(X, y)

    return {
        "best_estimator": grid.best_estimator_,
        "best_parameters": grid.best_params_,
        "best_score": grid.best_score_,
        "grid": grid
    }


def svc_summary(results, X, y, cv=None):
    """
    Print the results for the best SVC model.

    Params:
        results (dict): Results returned from train_svc().
        X: Feature data.
        y: Activity labels.
        cv: Cross-validation splitter.

    Returns:
        array: Predicted activity labels.
    """

    if cv is None:
        cv = get_cv_splitter()

    print("Best parameters:", results["best_parameters"])
    print(f"Best CV accuracy: {results['best_score']:.3f}")
    print()

    predictions = cross_val_predict(results["best_estimator"], X, y, cv=cv)

    print("Classification Report:")
    print(classification_report(y, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))

    return predictions