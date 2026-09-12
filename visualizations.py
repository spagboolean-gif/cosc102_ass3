"""
File for visualization templates 

used by: data_analysis.ipynb
uses data output from models defined in knn.py, svc.py and rforest.py

"""
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def  plot_confusion_matrix_comparison(predictions_by_model, y_true, labels):
    """
    plot confusion matrices for each algorithm and present them side by side for comparison.

    Parameters: 
    predictions_by_model : dictionary mapping the model names to the predicted label
    y_true: true activity labels, same for each model
    labels: class names, ordered by activity_labels in data_analysis.ipynb
    """

    n_models = len(predictions_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    for ax, (title, y_pred) in zip(axes, predictions_by_model.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=labels, cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, colorbar=False
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.show


def plot_knn_curve(grid, param_name="knn__n_neighbors", ax=None):
    """
    plot of mean cv accuracy against a single tuned K-NN hyperparameter

    Parameters:
    grid: fitted GridSearchCV object (from knn_results["grid"])
    param_name: the parameter name to plot
    ax: matplotlib Axes, to be combined in plot_hyperparameter_comparison()

    """

    results = grid.cv_results_
    df = pd.DataFrame({
        param_name: [p[param_name] for p in results["params"]],
        "score": results["mean_test_score"],
    })

    df = df.groupby(param_name, as_index=False)["score"].mean()

    ax.plot(df[param_name], df["score"], marker="o")
    ax.set_xlabel(param_name.split("__")[-1])
    ax.set_ylabel("Mean CV accuracy")
    ax.set_title("K-NN")



def plot_grid_heatmap(grid, param1, param2, title, ax=None):
    """
    Heatmap of mean cv accuracy across two tuned hyperparameters

    Parameters:
    grid: fitted GridSearchCV object
    param1, param2: parameter names for heatmap rows/columns
    title: subplot title
    ax: matplotlib Axes, to be combines in plot_hyperparameter_comparison
    """

    results = grid.cv_results_
    df = pd.DataFrame({
        param1: [p.get(param1) for p in results["params"]],
        param2: [p.get(param2) for p in results["params"]],
        "score": results["mean_test_score"],
    }).dropna()

    pivot = df.pivot(index=param1, columns=param2, values="score")
    im = ax.imshow(pivot.values, cmap="plasma", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(param2.split("__")[-1])
    ax.set_ylabel(param1.split("__")[-1])
    ax.set_title(title)

    #annotate each cell with its score : this seems a retarded way to do this but idk how else and it works so its fine I guess
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val: .2f}", ha="center", va="center", color="white")


def plot_hyperparameter_comparison(knn_grid, svc_grid, rforest_grid):
    """
    Combine the hyperparameter tuning plots in one figure for comparison
    line plot for knn 
    heatmaps for svc and random forest
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    plot_knn_curve(knn_grid, ax=axes[0])
    plot_grid_heatmap(svc_grid, "svc__C", "svc__gamma", title="SVC (rbf kernel)", ax=axes[1])
    plot_grid_heatmap(rforest_grid, "rforest__n_estimators", "rforest__max_leaf_nodes", title="Random Forest", ax=axes[2])

    plt.tight_layout()
    plt.show()