"""
File for visualization templates 

used by: data_analysis.ipynb
uses data output from models defined in knn.py, svc.py and rforest.py

"""


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



#def plot_confusion_matrix(y_true, y_pred, labels, title):
#    fig, ax = plt.subplots(figsize=(6, 6))
#    ConfusionMatrixDisplay.from_predictions(
#        y_true, y_pred, labels=labels, cmap=plt.cm.Blues, ax=ax,
#        xticks_rotation=45
#    )
#    ax.set_title(title)
#    plt.show()


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