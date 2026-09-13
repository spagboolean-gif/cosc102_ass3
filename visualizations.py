"""
File for visualization templates 

used by: data_analysis.ipynb
uses data output from models defined in knn.py, svc.py and rforest.py

"""
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_activity_alignment(imu_df, annotations_df, start_sec=None, end_sec=None):
    """
    Plot IMU data and activity transitions to check alignment.

    Params:
        imu_df: Aligned IMU data.
        annotations_df: Activity annotations produced by load_annotations().
        start_sec: Optional start time for the plot.
        end_sec: Optional end time for the plot.

    Returns:
        tuple: Matplotlib figure and axes containing the plot.
    """
    imu_data = imu_df.copy()
    annotations = annotations_df.copy()

    duration = (imu_data["timestamp"].max() - imu_data["timestamp"].min())
    imu_data["time_sec"] = np.linspace(0, duration, len(imu_data))

    if start_sec is not None:
        imu_data = imu_data[imu_data["time_sec"] >= start_sec]
        annotations = annotations[
            annotations["start_time_sec"] >= start_sec
        ]

    if end_sec is not None:
        imu_data = imu_data[imu_data["time_sec"] <= end_sec]
        annotations = annotations[
            annotations["start_time_sec"] <= end_sec
        ]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(imu_data["time_sec"], imu_data["accel_x"], label="Accel X")
    ax.plot(imu_data["time_sec"], imu_data["accel_y"], label="Accel Y")
    ax.plot(imu_data["time_sec"], imu_data["accel_z"], label="Accel Z")

    for transition_time in annotations["start_time_sec"]:
        ax.axvline(transition_time, color="black", linestyle="--")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Acceleration")
    ax.set_title("IMU Data and Activity Transitions")
    ax.legend()

    plt.tight_layout()
    plt.show()
    return fig, ax

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
            y_true, y_pred,
            labels=labels,
            cmap=plt.cm.Blues,
            ax=ax, xticks_rotation=45,
            colorbar=False
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


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


    def plot_knn_pca(estimator, X, y, y_prediction, title="K-NN PCA projection"):
        """
        PCA plot of scaled feature space with plot points coloured by their activity label, misclassified points marked with an X

        """

        X_scaled = estimator.named_steps["scaler"].transform(X)
        X_2d = PCA(n_components=2).fit_transform(X_scaled)

        correct == (y.values == y_prediction)

        fig, ax = plt.subplots(figsize=(9,7))
        for label in sorted(y.unique()):
            mask = (y.values == label)
            ax.scatter(X_2d[mask & correct, 0], X_2d[mask & correct, 1], 
                       label=label, alpha=0.6, s=25)

        # overlay misclassified points as X markers
        ax.scatter(X_2d[~correct, 0], X_2d[~correct, 1], marker="x", color="black", s=40, 
                    label="misclassified")

        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.legend()
        plt.tight_layout()
        plt.show()