
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sub_conf_mat(subject_id, y_true, y_pred, classes, model_name):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle(f'Confusion Matrices for Subject {subject_id}', fontsize=16)

    predictions = y_pred


    cm = confusion_matrix(y_true, predictions)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=classes, yticklabels=classes)

    ax.set_title(f'{model_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.show()