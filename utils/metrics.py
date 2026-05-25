import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def create_confusion_matrix_figure(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray, target: str, title: str) -> Figure:
    """
    Generates a matplotlib figure containing a styled Confusion Matrix.
    """
    if target == "KL":
        num_classes = 5
        classes = [f"Grade {i}" for i in range(num_classes)]
    elif target == "OA":
        num_classes = 2
        classes = ["Healthy (0)", "OA (1)"]
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Grade', xlabel='Predicted Grade')
           
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
                    
    fig.tight_layout()
    return fig