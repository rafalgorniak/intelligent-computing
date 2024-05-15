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


def pageOne():

    data, labels = getSets()

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):

        # SVC section
        svm_kernels = ['linear', 'rbf']
        svm_c_values = [(10**i) for i in range(-2, 7)]

        for kernel in svm_kernels:
            score = 0.0
            c_value = 0.0

            for C in svm_c_values:
                svc = SVC(gamma='auto', kernel=kernel, C=C)
                svc.fit(points_data, labels_data)
                actual_score = svc.score( points_data, labels_data)

                if( actual_score > score ):
                    print("Dataset:", index, ", kernel:", kernel, ", log(C):", math.log(C), ", actual score:", actual_score, "score:",
                          score)
                    score = actual_score
                    c_value = C

            svc = SVC(gamma='auto', kernel=kernel, C=c_value)
            svc.fit(points_data, labels_data)
            plot_decision_boundary(points_data, labels_data, svc.predict,f'SVM Dec. Bound. for set {index+1}, kernel = {kernel}, log(C) = {math.log10(c_value)}, acc. = {"{:.1f}".format(score)}')

        # MLP section
        mlp_activations = ['relu', 'identity']
        hidden_layer_neurons = [50,100,150,200,250,300,350,400,450,500]

        for activation in mlp_activations:
            score = 0.0
            neurons_value = 0

            for neurons in hidden_layer_neurons:
                mlp = MLPClassifier(random_state=42, hidden_layer_sizes=neurons, activation=activation, max_iter=500, n_iter_no_change=500, tol=0, solver='sgd')
                mlp.fit(points_data, labels_data)
                actual_score = mlp.score(points_data, labels_data)

                if( actual_score > score ):
                    print("Dataset:", index, ", activation:", activation, ", neurons:", neurons, ", actual score:", actual_score, "score:",
                          score)
                    score = actual_score
                    neurons_value = neurons

            mlp = MLPClassifier(random_state=42, hidden_layer_sizes=neurons_value, activation=activation, max_iter=500, n_iter_no_change=500, tol=0, solver='sgd')
            mlp.fit(points_data, labels_data)
            plot_decision_boundary(points_data, labels_data, mlp.predict,f'MLP Dec. Bound. for set {index+1}, act. = {activation}, neurons = {neurons_value}, acc. = {"{:.1f}".format(score)}')


def pageTwo():
    data, labels = getSets()
    data = data[1:]
    labels = labels[1:]

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):
        X_train, X_test, labels_train, labels_test = train_test_split(points_data, labels_data, test_size=0.2,
                                                                      random_state=42)

        train_score = []
        test_score = []
        best_n_value_test = 0
        best_accuracy_test = 0

        # SVM section
        for neighbour in range(1, 15):
            neigh = KNeighborsClassifier(n_neighbors=neighbour)
            neigh.fit(X_train, labels_train)

            predictionsTrain = neigh.predict(X_train)
            train_accuracy = accuracy_score(labels_train, predictionsTrain)
            train_score.append(train_accuracy)

            predictionsTest = neigh.predict(X_test)
            test_accuracy = accuracy_score(labels_test, predictionsTest)
            test_score.append(test_accuracy)

            if test_accuracy >= best_accuracy_test:
                best_accuracy_test = test_accuracy
                best_n_value_test = neighbour

        scorePlot(range(1, 15), train_score, test_score,"n_neighbours")

        # Decision boundary for train set / test set

        values = [1, 15, best_n_value_test]

        for val in values:

            neigh = KNeighborsClassifier(n_neighbors=val)
            neigh.fit(X_train, labels_train)

            if val != best_n_value_test:
                plot_decision_boundary(X_train, labels_train, neigh.predict,
                                       f'K-NN Dec. Bound. for train set {index + 1}, n_neighbors = {val}')
                plot_confusion_matrix(labels_train, neigh.predict(X_train),
                                      f"K-NN Confusion matrix for train set {index + 1}, n_neighbors = {val} ")

            plot_decision_boundary(X_test, labels_test, neigh.predict,
                                   f'K-NN Dec. Bound. for test set {index + 1}, n_neighbors = {val}')
            plot_confusion_matrix(labels_test, neigh.predict(X_test),
                                  f"K-NN confusion matrix for train set {index + 1}, n_neighbors = {val} ")


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
        best_c_value_test = 0
        best_accuracy_test = 0

        # SVM section
        for C in svm_c_values:
            svc = SVC(gamma='auto', kernel='rbf', C=C)
            svc.fit(X_train, labels_train)

            predictionsTrain = svc.predict(X_train)
            train_accuracy = accuracy_score(labels_train, predictionsTrain)
            train_score.append(train_accuracy)

            predictionsTest = svc.predict(X_test)
            test_accuracy = accuracy_score(labels_test, predictionsTest)
            test_score.append(test_accuracy)

            if test_accuracy >= best_accuracy_test:
                best_accuracy_test = test_accuracy
                best_c_value_test = C

        log_svm_c_values = [math.log10(c) for c in svm_c_values]
        scorePlot(log_svm_c_values, train_score, test_score,"log(C)")

        # Decision boundary for train set / test set

        values = [0.01, 1000000, best_c_value_test]

        for val in values:

            svc = SVC(gamma='auto', kernel='rbf', C=val)
            svc.fit(X_train, labels_train)

            if val != best_c_value_test:
                plot_decision_boundary(X_train, labels_train, svc.predict,
                                       f'SVM Dec. Bound. for train set {index + 1}, log(C) = {math.log10(val)}')
                plot_confusion_matrix(labels_train, svc.predict(X_train),
                                      f"SVM confusion matrix for train set {index + 1}, log(C) = {math.log10(val)} ")

            plot_decision_boundary(X_test, labels_test, svc.predict,
                                   f'SVM Dec. Bound. for test set {index + 1}, log(C) = {math.log10(val)}')
            plot_confusion_matrix(labels_test, svc.predict(X_test),
                                  f"SVM confusion matrix for train set {index + 1}, log(C) = {math.log10(val)}")


def pageFour():
    data, labels = getSets()
    data = data[1:]
    labels = labels[1:]

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):
        X_train, X_test, labels_train, labels_test = train_test_split(points_data, labels_data, test_size=0.2,
                                                                      random_state=42)

        mlp_values = range(1,70,5)
        train_score = []
        test_score = []
        best_mlp_value_test = 0
        best_accuracy_test = 0

        # SVM section
        for neurons in mlp_values:
            mlp = MLPClassifier(random_state=42, hidden_layer_sizes=neurons, activation='relu', max_iter=100,
                                n_iter_no_change=100, tol=0, solver='sgd')
            mlp.fit(X_train, labels_train)

            predictionsTrain = mlp.predict(X_train)
            train_accuracy = accuracy_score(labels_train, predictionsTrain)
            train_score.append(train_accuracy)

            predictionsTest = mlp.predict(X_test)
            test_accuracy = accuracy_score(labels_test, predictionsTest)
            test_score.append(test_accuracy)

            if test_accuracy >= best_accuracy_test:
                best_accuracy_test = test_accuracy
                best_mlp_value_test = neurons

        scorePlot(mlp_values, train_score, test_score, "hidden_layer_size")

        # Decision boundary for train set / test set

        mlp_values = [1, 70, best_mlp_value_test]

        for val in mlp_values:

            mlp = MLPClassifier(random_state=42, hidden_layer_sizes=val, activation='relu', max_iter=100,
                                n_iter_no_change=100, tol=0, solver='sgd')
            mlp.fit(X_train, labels_train)

            if val != best_mlp_value_test:
                plot_decision_boundary(X_train, labels_train, mlp.predict,
                                       f'MLP Dec. Bound. for train set {index + 1}, hidden layer neurons = {val}')
                plot_confusion_matrix(labels_train, mlp.predict(X_train),
                                      f"MLP confusion matrix for train set {index + 1}, hidden layer neurons = {val} ")

            plot_decision_boundary(X_test, labels_test, mlp.predict,
                                   f'MLP Dec. Bound. for test set {index + 1}, hidden layer neurons = {val}')
            plot_confusion_matrix(labels_test, mlp.predict(X_test),
                                  f"MLP confusion matrix for train set {index + 1}, hidden layer neurons = {val}")


def pageFive():
    data, labels = getSets()
    data = data[1:]
    labels = labels[1:]

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):
        X_train, X_test, labels_train, labels_test = train_test_split(points_data, labels_data, test_size=0.2,
                                                                      train_size=0.2, random_state=42)

        train_score = []
        test_score = []
        best_n_value_test = 0
        best_accuracy_test = 0

        # SVM section
        for neighbour in range(1, 15):
            neigh = KNeighborsClassifier(n_neighbors=neighbour)
            neigh.fit(X_train, labels_train)

            predictionsTrain = neigh.predict(X_train)
            train_accuracy = accuracy_score(labels_train, predictionsTrain)
            train_score.append(train_accuracy)

            predictionsTest = neigh.predict(X_test)
            test_accuracy = accuracy_score(labels_test, predictionsTest)
            test_score.append(test_accuracy)

            if test_accuracy >= best_accuracy_test:
                best_accuracy_test = test_accuracy
                best_n_value_test = neighbour

        scorePlot(range(1, 15), train_score, test_score, "n_neighbours")

        # Decision boundary for train set / test set

        values = [1, 15, best_n_value_test]

        for val in values:

            neigh = KNeighborsClassifier(n_neighbors=val)
            neigh.fit(X_train, labels_train)

            if val != best_n_value_test or val == 1 or val == 15:
                plot_decision_boundary(X_train, labels_train, neigh.predict,
                                       f'K-NN Dec. Bound. for train set {index + 1}, n_neighbors = {val}')
                plot_confusion_matrix(labels_train, neigh.predict(X_train),
                                      f"K-NN Confusion matrix for train set {index + 1}, n_neighbors = {val}")

            plot_decision_boundary(X_test, labels_test, neigh.predict,
                                   f'K-NN Dec. Bound. for test set {index + 1}, n_neighbors = {val}')
            plot_confusion_matrix(labels_test, neigh.predict(X_test),
                                  f"K-NN confusion matrix for test set {index + 1}, n_neighbors = {val}")


def pageSix():
    data, labels = getSets()
    data = data[1:]
    labels = labels[1:]

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):
        X_train, X_test, labels_train, labels_test = train_test_split(points_data, labels_data, test_size=0.2,
                                                                      train_size=0.2, random_state=42)

        svm_c_values = [(10**i) for i in np.arange(-2.0, 6.0, 0.25)]
        train_score = []
        test_score = []
        best_c_value_test = 0
        best_accuracy_test = 0

        # SVM section
        for C in svm_c_values:
            svc = SVC(gamma='auto', kernel='rbf', C=C)
            svc.fit(X_train, labels_train)

            predictionsTrain = svc.predict(X_train)
            train_accuracy = accuracy_score(labels_train, predictionsTrain)
            train_score.append(train_accuracy)

            predictionsTest = svc.predict(X_test)
            test_accuracy = accuracy_score(labels_test, predictionsTest)
            test_score.append(test_accuracy)

            if test_accuracy >= best_accuracy_test:
                best_accuracy_test = test_accuracy
                best_c_value_test = C

        log_svm_c_values = [math.log10(c) for c in svm_c_values]
        scorePlot(log_svm_c_values, train_score, test_score,"log(C)")

        # Decision boundary for train set / test set

        values = [0.01, 1000000, best_c_value_test]

        for val in values:

            svc = SVC(gamma='auto', kernel='rbf', C=val)
            svc.fit(X_train, labels_train)

            if val != best_c_value_test or val == 0.01 or val == 1000000 :
                plot_decision_boundary(X_train, labels_train, svc.predict,
                                       f'SVM Dec. Bound. for train set {index + 1}, log(C) = {math.log10(val)}')
                plot_confusion_matrix(labels_train, svc.predict(X_train),
                                      f"SVM confusion matrix for train set {index + 1}, log(C) = {math.log10(val)} ")

            plot_decision_boundary(X_test, labels_test, svc.predict,
                                   f'SVM Dec. Bound. for test set {index + 1}, log(C) = {math.log10(val)}')
            plot_confusion_matrix(labels_test, svc.predict(X_test),
                                  f"SVM confusion matrix for test set {index + 1}, log(C) = {math.log10(val)}")


def pageSeven():
    data, labels = getSets()
    data = data[1:]
    labels = labels[1:]

    for index, (points_data, labels_data) in enumerate(zip(data, labels)):
        X_train, X_test, labels_train, labels_test = train_test_split(points_data, labels_data, test_size=0.2,
                                                                      train_size=0.2, random_state=42)

        mlp_values = range(1, 70, 5)
        train_score = []
        test_score = []
        best_mlp_value_test = 0
        best_accuracy_test = 0

        # SVM section
        for neurons in mlp_values:
            mlp = MLPClassifier(random_state=42, hidden_layer_sizes=neurons, activation='relu', max_iter=100,
                                n_iter_no_change=100, tol=0, solver='sgd')
            mlp.fit(X_train, labels_train)

            predictionsTrain = mlp.predict(X_train)
            train_accuracy = accuracy_score(labels_train, predictionsTrain)
            train_score.append(train_accuracy)

            predictionsTest = mlp.predict(X_test)
            test_accuracy = accuracy_score(labels_test, predictionsTest)
            test_score.append(test_accuracy)

            if test_accuracy >= best_accuracy_test:
                best_accuracy_test = test_accuracy
                best_mlp_value_test = neurons

        scorePlot(mlp_values, train_score, test_score, "hidden_layer_size")

        # Decision boundary for train set / test set

        mlp_values = [1, 70, best_mlp_value_test]

        for val in mlp_values:

            mlp = MLPClassifier(random_state=42, hidden_layer_sizes=val, activation='relu', max_iter=100,
                                n_iter_no_change=100, tol=0, solver='sgd')
            mlp.fit(X_train, labels_train)

            if val != best_mlp_value_test or val == 1 or val == 70 :
                plot_decision_boundary(X_train, labels_train, mlp.predict,
                                       f'MLP Dec. Bound. for train set {index + 1}, hidden layer neurons = {val}')
                plot_confusion_matrix(labels_train, mlp.predict(X_train),
                                      f"MLP confusion matrix for train set {index + 1}, hidden layer neurons = {val} ")

            plot_decision_boundary(X_test, labels_test, mlp.predict,
                                   f'MLP Dec. Bound. for test set {index + 1}, hidden layer neurons = {val}')
            plot_confusion_matrix(labels_test, mlp.predict(X_test),
                                  f"MLP confusion matrix for test set {index + 1}, hidden layer neurons = {val}")


def pageEight():
    data, labels = getSets()
    data, labels = data[2], labels[2]

    train_test_sizes = [[0.2, 0.8], [0.2, 0.2]]

    for test_set, train_set in train_test_sizes:

        X_train, X_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_set,
                                                                      train_size=train_set, random_state=42)

        mlp = MLPClassifier(random_state=42, hidden_layer_sizes=50, activation='relu', max_iter=1,
                            n_iter_no_change=1, tol=0, solver='sgd')

        train_scores = []
        test_scores = []

        accuracy_train_best = 0
        best_points_train = []
        best_labels_train = []
        epochs_train = 0

        accuracy_test_best = 0
        best_points_test = []
        best_labels_test = []
        epochs_test = 0

        for i in range(100_001):

            mlp.partial_fit( X_train, labels_train, classes=np.unique(labels_train) )

            if i == 0 or i == 100_000:

                plot_decision_boundary(X_train, labels_train, mlp.predict,
                                       f'MLP Dec. Bound. for train set, where train_size = {train_set}, epochs = {i}. ')
                plot_decision_boundary(X_test, labels_test, mlp.predict,
                                       f'MLP Dec. Bound. for test set, where train_size = {train_set}, epochs = {i}. ')

            train_score = mlp.score(X_train, labels_train)
            test_score = mlp.score(X_test, labels_test)

            if train_score >= accuracy_train_best:
                accuracy_train_best = train_score
                best_points_train, best_labels_train = X_train, labels_train
                epochs_train = i

            if test_score >= accuracy_test_best:
                accuracy_test_best = test_score
                best_points_test, best_labels_test = X_test, labels_test
                epochs_test = i

            train_scores.append(train_score)
            test_scores.append(test_score)

        plot_decision_boundary(best_points_train, best_labels_train, mlp.predict,
                               f'MLP Dec. Bound. for train set for {epochs_train} epochs (best accuracy). ')
        plot_decision_boundary(best_points_test, best_labels_test, mlp.predict,
                               f'MLP Dec. Bound. for test set for {epochs_test} epochs (best accuracy). ')

        scorePlot(range(100_001), train_scores, test_scores, "epoch")

if __name__ == '__main__':

    #pageOne()
    #pageTwo()
    #pageThree()
    #pageFour()
    #pageFive()
    #pageSix()
    #pageSeven()
    #pageEight()
    pass
