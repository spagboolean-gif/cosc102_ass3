"""
File for visualization templates 

used by: data_analysis.ipynb
uses data output from models defined in knn.py, svc.py and rforest.py

"""


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, labels, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, cmap=plt.cm.Blues, ax=ax,
        xticks_rotation=45
    )
    ax.set_title(title)
    plt.show()