from preprocessing import get_features_and_labels, get_cv_splitter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


# not sure how I'm supposed to use x and y. I'll ask this evening and edit my work afterwards



def train_rforest(X, y, cv=None, param_grid=None):
    """
    Trains a random forest classifier. 

    Params:
        X: Feature matrix produced by preprocessing.py.
        y: Activity labels corresponding to the feature matrix.
        cv: Cross-validation splitter (if None, get_cv_splitter() 
            from preprocessing.py is used).

    Returns:
        
    """
    if cv is None:
        cv = get_cv_splitter()

    if param_grid is None:
        param_grid = [
            {
                "rforest__n_estimators":[125,250,500],
                "rforest__max_leaf_nodes":[4,8,16,32,64]
            }
        ]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rforest", RandomForestClassifier())
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X, y)

    return{"best_estimator":grid.best_estimator_,
           "best_parameters":grid.best_params_,
           "best_score":grid.best_score_,
           "grid":grid}

def rforest_summary(results, X, y, cv=None):
    print(f"Best parameters: {results['best_parameters']}\n\
        Best CV accuracy: {results['best_score']:.3f}\n")

    if cv is None:
        cv=get_cv_splitter()

    y_prediction = cross_val_predict(results["best_estimator"], X, y, cv=cv)
    report = classification_report(y, y_prediction)
    matrix = confusion_matrix(y, y_prediction)

    print(f"Classification Report Summary: \n{classification_report(y,y_prediction)}\n\
        Confusion Matrix: \n\
        {confusion_matrix(y,y_prediction)}")

    return {
        "predictions": y_prediction,
        "classification_report": report,
        "confusion_matrix": matrix,
    }

# commented out since they will be called in data_analysis.ipynb
# x,y = get_features_and_labels()


#forestclf = train_rforest(x,y)
#rforest_summary(forestclf, x, y)
