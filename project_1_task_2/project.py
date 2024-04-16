import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, \
    v_measure_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
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
    svm_c_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4, 4.0, 4.5, 5.0]
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
                    print("Dataset:", index, ", kernel:", kernel, ", C:", C, ", actual score:", actual_score, "score:",
                          score)
                    score = actual_score
                    kernel_best = kernel
                    c_value = C
        svc = SVC(gamma='auto', kernel=kernel_best, C=c_value)
        svc.fit(points_data, labels_data)
        plot_decision_boundary(points_data, labels_data, svc.predict,f'SVM Dec. Bound. for set {index+1}, kernel = {kernel_best}, C = {c_value}, acc. = {"{:.1f}".format(score)}')

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
        X_train, X_test, labels_train, labels_test = train_test_split(points_data, labels_data, test_size = 0.8, random_state = 42)

        # K-NN section

        neigh = KNeighborsClassifier(n_neighbors=8)
        neigh.fit(X_train, labels_train)
        predictionsTrain = neigh.predict(X_train)
        predictionsTest = neigh.predict(X_test)
        print(precision_score(labels_train, predictionsTrain))
        print(precision_score(labels_test, predictionsTest))


        # SVM section
        # MLP section

if __name__ == '__main__':

    #pageOne()
    pageTwo()
