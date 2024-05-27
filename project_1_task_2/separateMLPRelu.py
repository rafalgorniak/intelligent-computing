import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from warmUp.warmup import load


def plot_decision_boundary(X, y_true, func, title):
    """
    Function responsible for drawing classification diagram. It's more subtle, and general regions, mean boundaries
    are seen rather than independent area of singular dot. Firstly, to properly prepare fundamentals, meshgrid it to be
    appointed. Next, prediction function compute labels based on nested points. Then, plot functions are added,
    one draw boundary, and the next dots representing points from array.
    """

    # Generate a grid of points covering the range of data points
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the labels for the grid points
    z = func(np.c_[xx.ravel(), yy.ravel()])

    # Reshape the predictions to the same shape as the meshgrid
    z = z.reshape(xx.shape)

    # Plot the decision boundary and the data points
    plt.contourf(xx, yy, z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, marker='o', edgecolors='k')
    plt.title(title)

    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels, title):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.show()


def scorePlot(range, points1, points2, x_label):
    plt.plot(range, points1, marker='', label='train accuracy')
    plt.plot(range, points2, marker='', label='test accuracy')

    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    plt.legend(loc='lower right')

    plt.show()


def getSets():
    data = []
    labels = []
    file_1 = "2_1.csv"
    file_2 = "2_2.csv"
    file_3 = "2_3.csv"

    data.append(StandardScaler().fit_transform(load(file_1)[0]))
    data.append(StandardScaler().fit_transform(load(file_2)[0]))
    data.append(StandardScaler().fit_transform(load(file_3)[0]))
    labels.append(load(file_1)[1])
    labels.append(load(file_2)[1])
    labels.append(load(file_3)[1])

    return np.array(data), np.array(labels)


data, labels = getSets()

for index, (points_data, labels_data) in enumerate(zip(data, labels)):

    # MLP section
    mlp_activations = ['relu']
    hidden_layer_neurons = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    for activation in mlp_activations:
        score = 0.0
        neurons_value = 0

        best_data = points_data
        best_labels = labels_data

        for neurons in hidden_layer_neurons:
            train_data = points_data
            train_labels = labels_data

            mlp = MLPClassifier(random_state=42, hidden_layer_sizes=neurons, activation=activation, max_iter=100,
                                n_iter_no_change=100, tol=0, solver='sgd')
            mlp.fit(train_data, train_labels)
            actual_score = mlp.score(train_data, train_labels)

            plot_decision_boundary(best_data, best_labels, mlp.predict,
                                   f'MLP Dec. Bound. for set {index + 1}, act. = {activation}, neurons = {neurons}, acc. = {"{:.1f}".format(actual_score)}')

            if (actual_score > score):
                print("Dataset:", index, ", activation:", activation, ", neurons:", neurons, ", actual score:",
                      actual_score, "score:",
                      score)
                score = actual_score
                neurons_value = neurons

        mlp = MLPClassifier(random_state=42, hidden_layer_sizes=neurons_value, activation=activation, max_iter=500,
                            n_iter_no_change=500, tol=0, solver='sgd')
        mlp.fit(best_data, best_labels)
        plot_decision_boundary(best_data, best_labels, mlp.predict,
                               f'MLP Dec. Bound. for set {index + 1}, act. = {activation}, neurons = {neurons_value}, acc. = {"{:.1f}".format(score)}')
