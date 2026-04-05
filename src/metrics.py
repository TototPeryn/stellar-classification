from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def metrics(pred, real, to_return=True, output=True, cmap="Blues", vmin=0):  # X predicted, Y real
    X = pred
    Y = real
    if output:
        print(classification_report(Y, X, zero_division=0))
        ax = sns.heatmap(confusion_matrix(Y, X), annot=True, vmax=X.size, vmin=vmin,
                         fmt="d", cmap=cmap)
        ax.set(xlabel="Predicted class", ylabel="Real class")
        plt.show()

    if to_return:
        return {"Accuracy": accuracy_score(Y, X),
                "Precision": precision_score(Y, X, zero_division=0),
                "Recall": recall_score(Y, X, zero_division=0),
                "F1": f1_score(Y, X, zero_division=0)}


def metrics_balanced(pred, real, to_return=True, output=True, cmap="Blues", vmin=0):  # X predicted, Y real
    X = pred
    Y = real
    if output:
        print(classification_report(Y, X, zero_division=0))
        cm = confusion_matrix(Y, X)
        # all_index = cm.argmax()
        # cm[all_index // cm.shape[0]][all_index % cm.shape[1]] = 0
        ax = sns.heatmap(cm, annot=True, vmax=np.partition(cm.flatten(), -2)[-2], vmin=vmin,
                         fmt="d", cmap=cmap)
        ax.set(xlabel="Predicted class", ylabel="Real class")
        plt.show()

    if to_return:
        return {"Accuracy": accuracy_score(Y, X),
                "Precision": precision_score(Y, X, zero_division=0),
                "Recall": recall_score(Y, X, zero_division=0),
                "F1": f1_score(Y, X, zero_division=0)}