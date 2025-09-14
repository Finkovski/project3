
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm: np.ndarray, labels = ('Neg','Neu','Pos'), title='Confusion Matrix'):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    return fig
