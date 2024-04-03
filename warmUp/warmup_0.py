import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import Voronoi, voronoi_plot_2d


def load(path: str) -> tuple:
    data: DataFrame = pd.read_csv(path, delimiter=';')
    points: np.ndarray = data.iloc[:, :-1].to_numpy()
    labels: np.ndarray = data.iloc[:, -1].astype(int).to_numpy()

    return points, labels


def plot_voronoi_diagram(
        X: np.ndarray, y_true_labels: np.ndarray, y_pred_labels: np.ndarray, title: str = 'Voronoi Diagram'
) -> None:

    plot_margin: float = 0.1

    x1_max: float = X[:, 0].max() + plot_margin
    x1_min: float = X[:, 0].min() - plot_margin
    x2_max: float = X[:, 1].max() + plot_margin
    x2_min: float = X[:, 1].min() - plot_margin

    border_point1: list = [999, 999]
    border_point2: list = [-999, 999]
    border_point3: list = [999, -999]
    border_point4: list = [-999, -999]

    X: np.ndarray = np.append(X, [border_point1, border_point2, border_point3, border_point4], axis=0)
    y_pred_labels: np.ndarray = np.append(y_pred_labels, [0, 0, 0, 0])

    if y_true_labels is not None:
        y_true_labels: np.ndarray = np.append(y_true_labels, [0, 0, 0, 0])

    group_colors: list = ['#377EB8', '#FF7F00', '#4DAF4A', '#DFFF00', '#FFBF00',
                          '#FF7F50', '#DE3163', '#9FE2BF', '#40E0D0', '#6495ED',
                          '#20B2AA', '#FFD700', '#ADD8E6', '#FF69B4', '#778899', '#FFFF00',
                          '#DA70D6', '#FF6347', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887',
                          '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC',
                          '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9',
                          '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC',
                          '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F', '#00CED1',
                          '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF', '#B22222',
                          '#FFFAF0']
    cluster_means: list = []

    voronoi: Voronoi = Voronoi(X)
    fig, ax = plt.subplots()

    for label in np.unique(y_pred_labels):
        cluster_mean: np.ndarray = np.mean(X[y_pred_labels == label], axis=0)
        cluster_means.append(cluster_mean)

    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_widths=1, line_alpha=0.4)

    for region_index in range(len(voronoi.point_region)):
        region: list = voronoi.regions[voronoi.point_region[region_index]]

        if not -1 in region:
            polygon: list = [voronoi.vertices[i] for i in region]
            point_label: int = y_pred_labels[region_index]
            color: str = group_colors[point_label % len(group_colors)]

            ax.fill(*zip(*polygon), color=color, alpha=0.4)

    if y_true_labels is not None:
        unique_true_labels: np.ndarray = np.unique(y_true_labels)

        for i, label in enumerate(unique_true_labels):
            idx: np.ndarray = np.where(y_true_labels == label)[0]
            color: str = group_colors[i % len(group_colors)]

            ax.plot(X[idx, 0], X[idx, 1], 'o', color=color, markersize=5, alpha=1)

    else:
        ax.plot(X[:, 0], X[:, 1], 'o', color='black', markersize=5, alpha=1)

    plt.xlim([x1_min, x1_max])
    plt.ylim([x2_min, x2_max])
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def plot_decision_boundary(
        X: np.ndarray, y_true_labels: np.ndarray, decision_function, title: str = 'Decision Boundary'
) -> None:
    plot_margin: float = 0.1
    grid_step_size: float = 0.01

    x_min: float = X[:, 0].min() - plot_margin
    x_max: float = X[:, 0].max() + plot_margin
    y_min: float = X[:, 1].min() - plot_margin
    y_max: float = X[:, 1].max() + plot_margin

    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step_size), np.arange(y_min, y_max, grid_step_size))

    Z = decision_function(np.c_[xx.ravel(), yy.ravel()])

    group_colors = ['#377EB8', '#4DAF4A', '#FF7F00']
    colormap = LinearSegmentedColormap.from_list('colormap', group_colors, N=256)

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=colormap, alpha=0.4)
    plt.contour(xx, yy, Z, colors='k', linewidths=1)

    if y_true_labels is not None:
        true_label_colors = [group_colors[label] for label in y_true_labels]
        plt.scatter(X[:, 0], X[:, 1], c=true_label_colors, edgecolors='k')

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    try:
        os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    except:
        pass

    X, y_true = load("warmup.csv")

    X = StandardScaler().fit_transform(X)

    algorithm = cluster.KMeans(n_clusters=3)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    plot_voronoi_diagram(X, y_true, y_pred)
    plot_voronoi_diagram(X, None, y_pred)

    algorithm = KNeighborsClassifier(n_neighbors=3)
    algorithm.fit(X, y_true)

    plot_decision_boundary(X, y_true, algorithm.predict)
