import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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


def run_first_experiment():
    for a in range(6):
        path = f"./{int(a/3)+1}_{a%3+1}.csv"
        X = load(path)[0]
        y_true = load(path)[1]
        X = StandardScaler().fit_transform(X)

        min_clusters = 2
        max_clusters = 10
        cluster_range = range(min_clusters, max_clusters + 1)
        silhouette_scores = []

        for n_clusters in cluster_range:
            # Fit KMeans clustering algorithm
            kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(X)

            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        plt.plot(cluster_range, silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.grid(True)
        plt.show()

        best_case = silhouette_scores.index(max(silhouette_scores))

        algorithm = cluster.KMeans(n_clusters=best_case+min_clusters)
        algorithm.fit(X)
        y_pred=algorithm.labels_.astype(int)

        plot_voronoi_diagram(X, y_true, y_pred)

        worst_case = silhouette_scores.index(min(silhouette_scores))

        algorithm = cluster.KMeans(n_clusters=worst_case+min_clusters)
        algorithm.fit(X)
        y_pred=algorithm.labels_.astype(int)

        plot_voronoi_diagram(X, y_true, y_pred)


def run_second_experiment():
    for a in range(6):

        path = f"./{int(a / 3) + 1}_{a % 3 + 1}.csv"
        X = load(path)[0]
        y_true = load(path)[1]
        X = StandardScaler().fit_transform(X)

        min_eps = 0.05
        max_eps = 0.5
        eps_range = np.arange(min_eps, max_eps,0.05)
        silhouette_scores = []
        clusters = []
        best_case, worst_case = 0, 0
        best_score,worst_score = 0, 1

        for eps in eps_range:
            # Fit KMeans clustering algorithm
            algorithm = cluster.DBSCAN(eps=eps, min_samples=1)
            cluster_labels = algorithm.fit_predict(X)

            y_pred = algorithm.labels_.astype(int)
            num_unique_labels = len(np.unique(y_pred))
            print(f'Number of unique labels for eps={eps}: {num_unique_labels}')  # Print for debugging

            clusters.append(num_unique_labels)

            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_case = len(np.unique(y_pred))
            if silhouette_avg < worst_score:
                worst_score=silhouette_avg
                worst_case = len(np.unique(y_pred))

        plt.plot(eps_range, silhouette_scores, marker='')
        plt.xlabel('eps')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.grid(True)
        clusters=np.array(clusters)

        # Add labels indicating number of clusters near marker points
        for i,eps in enumerate(eps_range):
            plt.annotate(clusters[i], (eps, silhouette_scores[i]), textcoords="offset points",
                         xytext=(5, -10), ha='center')

        plt.show()

        algorithm = cluster.KMeans(n_clusters=worst_case)
        algorithm.fit(X)
        y_pred = algorithm.labels_.astype(int)
        plot_voronoi_diagram(X, y_true, y_pred)

        algorithm = cluster.KMeans(n_clusters=best_case)
        algorithm.fit(X)
        y_pred = algorithm.labels_.astype(int)
        plot_voronoi_diagram(X, y_true, y_pred)


if __name__ == '__main__':

    pass
    #run_first_experiment()
    #run_second_experiment()

