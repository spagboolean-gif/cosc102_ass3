# Support Vector Classifier

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import get_cv_splitter


def create_svc():
    """
    Create the SVC pipeline.

    Returns:
        Pipeline: StandardScaler followed by SVC model.
    """
    return Pipeline([("scaler", StandardScaler()), ("svc", SVC())])


def train_svc(X, y, cv=None):
    """
    Train the SVC model using GridSearchCV.

    Params:
        X: Feature data.
        y: Activity labels.
        cv: Cross-validation splitter.

    Returns:
        GridSearchCV: Fitted grid search object.
    """
    if cv is None:
        cv = get_cv_splitter()

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

    grid = GridSearchCV(
        estimator=create_svc(),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    grid.fit(X, y)
    return grid


def evaluate_svc(grid, X, y, cv=None):
    """
    Evaluate the best SVC model using cross-validated predictions.

    Params:
        grid: Fitted GridSearchCV object from train_svc().
        X: Feature data.
        y: Activity labels.
        cv: Cross-validation splitter.

    Returns:
        dict: Predictions, classification report and confusion matrix.
    """
    if cv is None:
        cv = get_cv_splitter()

    predictions = cross_val_predict(grid.best_estimator_, X, y, cv=cv)
    report = classification_report(y, predictions)
    matrix = confusion_matrix(y, predictions)

    print("Best parameters:", grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.3f}")
    print()
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

    return {
        "predictions": predictions,
        "classification_report": report,
        "confusion_matrix": matrix
    }


def run_svc(X, y, cv=None):
    """
    Train and evaluate the SVC model.

    Params:
        X: Feature data.
        y: Activity labels.
        cv: Cross-validation splitter.

    Returns:
        tuple: Fitted grid search and evaluation results.
    """
    grid = train_svc(X, y, cv)
    results = evaluate_svc(grid, X, y, cv)
    return grid, results