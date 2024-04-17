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

from warmUp.warmup import plot_voronoi_diagram,load


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

    for i, n_cluster in enumerate(range):
        plt.axvline(x=n_cluster, color='gray', linestyle='--', alpha=0.3)

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

    return np.array(data),np.array(labels)

def pageOne():

    data, labels = getSets()
    svm_kernels = ['linear', 'rbf']
    svm_c_values = [(10**i) for i in range(-2, 7)]

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):

        # SVC section
        score=0.0
        kernel_best = ''
        c_value = 0.0
        for kernel in svm_kernels:
            for C in svm_c_values:
                svc = SVC(gamma='auto', kernel=kernel, C=C)
                svc.fit(points_data, labels_data)
                actual_score = svc.score( points_data,labels_data )
                if( actual_score > score ):
                    print("Dataset:", index, ", kernel:", kernel, ", log(C):", math.log(C), ", actual score:", actual_score, "score:",
                          score)
                    score = actual_score
                    kernel_best = kernel
                    c_value = C
        svc = SVC(gamma='auto', kernel=kernel_best, C=c_value)
        svc.fit(points_data, labels_data)
        plot_decision_boundary(points_data, labels_data, svc.predict,f'SVM Dec. Bound. for set {index+1}, kernel = {kernel_best}, log(C) = {math.log10(c_value)}, acc. = {"{:.1f}".format(score)}')

        # MLP section
        mlp_activations = ['relu', 'identity']
        hidden_layer_neurons = [50,100,150,200,250,300,350,400,450,500]
        score = 0.0
        activation_best = ''
        neurons_value = 0
        for activation in mlp_activations:
            for neurons in hidden_layer_neurons:
                mlp = MLPClassifier(random_state=42, hidden_layer_sizes=neurons, activation=activation, max_iter=100,n_iter_no_change=100,tol=0, solver='sgd')
                mlp.fit(points_data, labels_data)
                actual_score = mlp.score( points_data,labels_data )
                if( actual_score > score ):
                    print("Dataset:", index, ", activation:", activation, ", neurons:", neurons, ", actual score:", actual_score, "score:",
                          score)
                    score = actual_score
                    activation_best = activation
                    neurons_value = neurons
        mlp = MLPClassifier(random_state=42, hidden_layer_sizes=neurons_value, activation=activation_best, max_iter=100,n_iter_no_change=100,tol=0, solver='sgd')
        mlp.fit(points_data, labels_data)
        plot_decision_boundary(points_data, labels_data, mlp.predict,f'MLP Dec. Bound. for set {index+1}, act. = {activation_best}, neurons = {neurons_value}, acc. = {"{:.1f}".format(score)}')

def pageTwo():
    data, labels = getSets()
    data = data[1:]
    labels = labels[1:]

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):
        X_train, X_test, labels_train, labels_test = train_test_split(points_data, labels_data, test_size=0.2,
                                                                      random_state=42)

        train_score = []
        test_score = []
        best_n_value_train = 0
        best_n_value_test = 0
        best_accuracy_train = 0
        best_accuracy_test = 0

        # SVM section
        for neighbour in range(1, 15):
            neigh = KNeighborsClassifier(n_neighbors=neighbour)
            neigh.fit(X_train, labels_train)

            predictionsTrain = neigh.predict(X_train)
            train_accuracy = accuracy_score(labels_train, predictionsTrain)
            train_score.append(train_accuracy)

            if (train_accuracy > best_accuracy_train):
                best_accuracy_train = train_accuracy
                best_n_value_train = neighbour

            predictionsTest = neigh.predict(X_test)
            test_accuracy = accuracy_score(labels_test, predictionsTest)
            test_score.append(test_accuracy)

            if (test_accuracy > best_accuracy_test):
                best_accuracy_test = test_accuracy
                best_n_value_test = neighbour

        scorePlot(range(1, 15), train_score, test_score,"n_neighbours")

        # Decision boundary for train set / test set

        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(X_train, labels_train)
        plot_decision_boundary(X_train, labels_train, neigh.predict,
                               f'K-NN Dec. Bound. for train set {index + 1}, n_neighbors = {1}')
        plot_confusion_matrix(labels_train, neigh.predict(X_train), f"Confusion matrix for train set {index + 1}, n_neighbors = {1}' ")
        plot_decision_boundary(X_test, labels_test, neigh.predict,
                               f'K-NN Dec. Bound. for test set {index + 1}, n_neighbors = {1}')
        plot_confusion_matrix(labels_test, neigh.predict(X_test),
                              f"K-NN confusion matrix for train set {index + 1}, n_neighbors = {1}' ")

        neigh = KNeighborsClassifier(n_neighbors=15)
        neigh.fit(X_train, labels_train)
        plot_decision_boundary(X_train, labels_train, neigh.predict,
                               f'K-NN Dec. Bound. for train set {index + 1}, n_neighbors = {15}')
        plot_confusion_matrix(labels_train, neigh.predict(X_train),
                              f"K-NN confusion matrix for train set {index + 1}, n_neighbors = {15}' ")
        plot_decision_boundary(X_test, labels_test, neigh.predict,
                               f'K-NN Dec. Bound. for train set {index + 1}, n_neighbors = {15}')
        plot_confusion_matrix(labels_test, neigh.predict(X_test),
                              f"K-NN confusion matrix for train set {index + 1}, n_neighbors = {15}' ")

        neigh = KNeighborsClassifier(n_neighbors=best_n_value_train)
        neigh.fit(X_train, labels_train)
        plot_decision_boundary(X_train, labels_train, neigh.predict,
                               f'K-NN Dec. Bound. for train set {index + 1}, n_neighbors = {best_n_value_train}, acc. = {"{:.1f}".format(best_accuracy_train)}')
        plot_confusion_matrix(labels_train, neigh.predict(X_train),
                              f"K-NN confusion matrix for train set {index + 1}, n_neighbors = {best_n_value_train}")
        neigh = KNeighborsClassifier(n_neighbors=best_n_value_test)
        neigh.fit(X_train, labels_train)
        plot_decision_boundary(X_test, labels_test, neigh.predict,
                               f'K-NN Dec. Bound. for train set {index + 1}, n_neighbors = {best_n_value_test}, acc. = {"{:.1f}".format(best_accuracy_test)}')
        plot_confusion_matrix(labels_test, neigh.predict(X_test),
                              f"K-NN confusion matrix for train set {index + 1}, n_neighbors = {best_n_value_test}")

def pageThree():
    data, labels = getSets()
    data = data[1:]
    labels = labels[1:]

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):
        X_train, X_test, labels_train, labels_test = train_test_split(points_data, labels_data, test_size=0.2,
                                                                      random_state=42)

        svm_c_values = [(10**i) for i in np.arange(-2.0, 6.0, 0.25)]
        train_score = []
        test_score = []
        best_c_value_train = 0
        best_c_value_test = 0
        best_accuracy_train = 0
        best_accuracy_test = 0

        # SVM section
        for C in svm_c_values:
            svc = SVC(gamma='auto', kernel='rbf', C=C)
            svc.fit(X_train, labels_train)

            predictionsTrain = svc.predict(X_train)
            train_accuracy = accuracy_score(labels_train, predictionsTrain)
            train_score.append(train_accuracy)

            if (train_accuracy >= best_accuracy_train):
                best_accuracy_train = train_accuracy
                best_c_value_train = C

            predictionsTest = svc.predict(X_test)
            test_accuracy = accuracy_score(labels_test, predictionsTest)
            test_score.append(test_accuracy)

            if (test_accuracy >= best_accuracy_test):
                best_accuracy_test = test_accuracy
                best_c_value_test = C

        log_svm_c_values = [math.log10(c) for c in svm_c_values]
        scorePlot(log_svm_c_values, train_score, test_score,"log(C)")

        # Decision boundary for train set / test set

        svc = SVC(gamma='auto', kernel='rbf', C=0.01)
        svc.fit(X_train, labels_train)
        plot_decision_boundary(X_train, labels_train, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {-2}')
        plot_confusion_matrix(labels_train, svc.predict(X_train),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {-2} ")
        plot_decision_boundary(X_test, labels_test, svc.predict,
                               f'SVM Dec. Bound. for test set {index + 1}, log(C) = {-2}')
        plot_confusion_matrix(labels_test, svc.predict(X_test),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {-2}")

        svc = SVC(gamma='auto', kernel='rbf', C=1000000)
        svc.fit(X_train, labels_train)
        plot_decision_boundary(X_train, labels_train, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {6}')
        plot_confusion_matrix(labels_train, svc.predict(X_train),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {6} ")
        plot_decision_boundary(X_test, labels_test, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {6}')
        plot_confusion_matrix(labels_test, svc.predict(X_test),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {6} ")

        svc = SVC(gamma='auto', kernel='rbf', C=best_c_value_train)
        svc.fit(X_train, labels_train)
        plot_decision_boundary(X_train, labels_train, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {math.log10(best_c_value_train)}, acc. = {"{:.1f}".format(best_accuracy_train)}')
        plot_confusion_matrix(labels_train, svc.predict(X_train),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {math.log10(best_c_value_train)}")
        svc = SVC(gamma='auto', kernel='rbf', C=best_c_value_test)
        svc.fit(X_train, labels_train)
        plot_decision_boundary(X_test, labels_test, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {math.log10(best_c_value_test)}, acc. = {"{:.1f}".format(best_accuracy_test)}')
        plot_confusion_matrix(labels_test, svc.predict(X_test),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {math.log10(best_c_value_test)}")

def pageFour():
    data, labels = getSets()
    data = data[1:]
    labels = labels[1:]

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):
        X_train, X_test, labels_train, labels_test = train_test_split(points_data, labels_data, test_size=0.2,
                                                                      random_state=42)

        svm_c_values = [(10 ** i) for i in np.arange(-2.0, 6.0, 0.25)]
        train_score = []
        test_score = []
        best_c_value_train = 0
        best_c_value_test = 0
        best_accuracy_train = 0
        best_accuracy_test = 0

        # SVM section
        for C in svm_c_values:
            svc = SVC(gamma='auto', kernel='rbf', C=C)
            svc.fit(X_train, labels_train)

            predictionsTrain = svc.predict(X_train)
            train_accuracy = accuracy_score(labels_train, predictionsTrain)
            train_score.append(train_accuracy)

            if (train_accuracy >= best_accuracy_train):
                best_accuracy_train = train_accuracy
                best_c_value_train = C

            predictionsTest = svc.predict(X_test)
            test_accuracy = accuracy_score(labels_test, predictionsTest)
            test_score.append(test_accuracy)

            if (test_accuracy >= best_accuracy_test):
                best_accuracy_test = test_accuracy
                best_c_value_test = C

        log_svm_c_values = [math.log10(c) for c in svm_c_values]
        scorePlot(log_svm_c_values, train_score, test_score, "log(C)")

        # Decision boundary for train set / test set

        svc = SVC(gamma='auto', kernel='rbf', C=0.01)
        svc.fit(X_train, labels_train)
        plot_decision_boundary(X_train, labels_train, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {-2}')
        plot_confusion_matrix(labels_train, svc.predict(X_train),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {-2} ")
        plot_decision_boundary(X_test, labels_test, svc.predict,
                               f'SVM Dec. Bound. for test set {index + 1}, log(C) = {-2}')
        plot_confusion_matrix(labels_test, svc.predict(X_test),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {-2}")

        svc = SVC(gamma='auto', kernel='rbf', C=1000000)
        svc.fit(X_train, labels_train)
        plot_decision_boundary(X_train, labels_train, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {6}')
        plot_confusion_matrix(labels_train, svc.predict(X_train),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {6} ")
        plot_decision_boundary(X_test, labels_test, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {6}')
        plot_confusion_matrix(labels_test, svc.predict(X_test),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {6} ")

        svc = SVC(gamma='auto', kernel='rbf', C=best_c_value_train)
        svc.fit(X_train, labels_train)
        plot_decision_boundary(X_train, labels_train, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {math.log10(best_c_value_train)}, acc. = {"{:.1f}".format(best_accuracy_train)}')
        plot_confusion_matrix(labels_train, svc.predict(X_train),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {math.log10(best_c_value_train)}")
        svc = SVC(gamma='auto', kernel='rbf', C=best_c_value_test)
        svc.fit(X_train, labels_train)
        plot_decision_boundary(X_test, labels_test, svc.predict,
                               f'SVM Dec. Bound. for train set {index + 1}, log(C) = {math.log10(best_c_value_test)}, acc. = {"{:.1f}".format(best_accuracy_test)}')
        plot_confusion_matrix(labels_test, svc.predict(X_test),
                              f"SVM confusion matrix for train set {index + 1}, log(C) = {math.log10(best_c_value_test)}")

if __name__ == '__main__':

    #pageOne()
    #pageTwo()
    #pageThree()
    pageFour()
