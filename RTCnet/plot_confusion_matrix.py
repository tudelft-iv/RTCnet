"""
================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""



import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from copy import deepcopy
import matplotlib
font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)

class_names = np.array(['Others', 'Ped.', 'Rider', 'Car'])




def plot_confusion_matrix(confusion_matrix, classes,
                          title=None,
                          cmap=plt.cm.Blues,
                          ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    # Compute confusion matrix
    cm = confusion_matrix
    # Only use the labels that appear in the data

    # if normalize:
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_col = cm.astype('float') / cm.sum(axis=0)
    print(cm)
    print(cm_norm)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    
    cm_draw = deepcopy(cm_norm)
    cm_draw[0, 0] = 1
    im = ax.imshow(cm_draw, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i==j:
                ax.text(j, i, "{:d}\n(R: {:.2f})\n(P: {:.2f})".format(cm[i, j], cm_norm[i,j],cm_norm_col[i,j]), 
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black")
            else:
                ax.text(j, i, "{:d}\n({:.2f})\n({:.2f})".format(cm[i, j], cm_norm[i,j],cm_norm_col[i,j]), 
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black")

    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, "\n({:.2f})".format(cm_norm_col[i,j]), 
    #                 ha="center", va="center",
    #                 color="white" if cm_norm[i, j] > thresh else "black")
    # if ax is None:
        # fig.tight_layout()
    plt.tight_layout()
    return ax

if __name__ == "__main__":
    print(__doc__)
    np.set_printoptions(precision=2)
    methods = ["Schumann", "Prophet", "Low-level"]
    confusion_matrices = [np.array([[55060,2110,1503,53],
                                [7531,5946,611,33],
                                [4560,248,2853,13],
                                [1492,22,246,103]]),
                        np.array([[55281,1664,1700,81],
                                [7446,4712,1924,39],
                                [4384,603,2651,36],
                                [1494,51,223,95]]),
                        np.array([[54785,2424,1477,40],
                                [7979,5480,658,4],
                                [3704,320,3604,46],
                                [1144,33,598,88]])]

    for method, cm in zip(methods, confusion_matrices):
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(cm, classes=class_names,
                            title='Confusion matrix, {}, without normalization'.format(method))



        plt.show()
