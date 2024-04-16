import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def load(path):
    """
    Function responsible for converting/loading .csv file into Numpy arrays.
    Firstly, numbers are enucleated with help of pandas library. Because each float
    is separated with semicolon, it's treated as delimiter. Then corresponding columns
    mapped into numpy arrays are inscribed into returned values.
    """

    data = pd.read_csv(path, delimiter=';')

    points = data.iloc[:, :-1].to_numpy()
    labels = data.iloc[:, -1].astype(int).to_numpy()

    return points, labels


def plot_voronoi_diagram(X, y_true, y_pred):
    """
    Function responsible for drawing Voronoi diagram. It takes one 2D and two 1 dimensional numpy array as input
    parameters. Firstly, to fulfill requirement of painting all calculated areas, external area and additional scale
    should be taken under consideration. So after appointing points as a border and min/max horizontal and vertical
    coordinates as well as computing means for points of same label, one should draw a Voronoi basic plot. Then, for
    each region colorize area, and the same for points. Surely, tints are different depending on true/predict labels.
    After all, apply the scaling of output plot.
    """

    scale = 1.05

    # Store x1 and x2 min and max values for plot scaling
    x1_max = X[:, 0].max()
    x2_max = X[:, 1].max()
    x1_min = X[:, 0].min()
    x2_min = X[:, 1].min()

    # Append dummy points to fix border cells not being colored
    X = np.append(X, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)
    y_pred = np.append(y_pred, [0, 0, 0, 0])
    if y_true is not None:
        y_true = np.append(y_true, [0, 0, 0, 0])

    means = []
    for label in np.unique(y_pred):
        cluster_mean = np.mean(X[y_pred == label], axis=0)
        means.append(cluster_mean)

    # Plot Voronoi diagram
    vor = Voronoi(X)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=10)

    # Colorize Voronoi regions
    for region_index in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[region_index]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            point = vor.points[region_index]
            closest_mean_idx = np.argmin(np.linalg.norm(np.array(means) - point, axis=1))
            ax.fill(*zip(*polygon), color=plt.cm.jet(closest_mean_idx / len(means)), alpha=0.2)

    # Colorize points based on true labels
    if y_true is not None:
        unique_labels_true = np.unique(y_true)
        for label in unique_labels_true:
            idx = np.where(y_true == label)[0]
            ax.plot(X[idx, 0], X[idx, 1], 'o', color=plt.cm.jet(label / len(unique_labels_true)), markersize=4,
                    alpha=1)

    # Scale the output plot area
    plt.xlim([x1_min * scale, x1_max * scale])
    plt.ylim([x2_min * scale, x2_max * scale])
    plt.show()


def plot_decision_boundary(X, y_true, func):
    """
    Function responsible for drawing classification diagram. It's more subtle, and general regions, mean boundaries
    are seen rather than independent area of singular dot. Firstly, to properly prepare fundamentals, meshgrid it to be
    appointed. Next, prediction function compute labels based on nested points. Then, plot functions are added,
    one draw boundary, and the next dots representing points from array.
    """

    # Generate a grid of points covering the range of data points
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict the labels for the grid points
    z = func(np.c_[xx.ravel(), yy.ravel()])

    # Reshape the predictions to the same shape as the meshgrid
    z = z.reshape(xx.shape)

    # Plot the decision boundary and the data points
    plt.contourf(xx, yy, z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, marker='o', edgecolors='k')

    plt.show()


if __name__ == "__main__":

    X, y_true = load("./warmup.csv")
    X = StandardScaler().fit_transform(X)

    algorithm = cluster.KMeans(n_clusters=3)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)

    plot_voronoi_diagram(X, y_true, y_pred)
    plot_voronoi_diagram(X, None, y_pred)

    algorithm = KNeighborsClassifier(n_neighbors=3)
    algorithm.fit(X, y_true)

    plot_decision_boundary(X, y_true, algorithm.predict)