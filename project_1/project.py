import os
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from warmUp.warmup import plot_voronoi_diagram,load


def load_datasets() -> tuple:
    points = []
    labels = []

    for i in range(1, 4):
        temp_points, temp_labels = load(f"1_{i}.csv")
        temp_points = StandardScaler().fit_transform(temp_points)
        points.append(temp_points)
        labels.append(temp_labels)

        temp_points, temp_labels = load(f"2_{i}.csv")
        temp_points = StandardScaler().fit_transform(temp_points)
        points.append(temp_points)
        labels.append(temp_labels)

    return points, labels


def run_first_experiment():
    points, labels = load_datasets()

    for index, (points_data, labels_data) in enumerate(zip(points, labels)):
        best_silhouette_score = float('-inf')
        worst_silhouette_score = float('inf')
        best_clusters = None
        worst_clusters = None

        silhouette_scores = []
        for n_clusters in range(2, 9):
            kmeans = cluster.KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(points_data)

            silhouette = silhouette_score(points_data, cluster_labels)
            silhouette_scores.append(silhouette)

            if silhouette > best_silhouette_score:
                best_silhouette_score = silhouette
                best_clusters = n_clusters
                best_cluster_labels = cluster_labels

            if silhouette < worst_silhouette_score:
                worst_silhouette_score = silhouette
                worst_clusters = n_clusters
                worst_cluster_labels = cluster_labels

        plot_voronoi_diagram(
            points_data,
            labels_data,
            best_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for {best_clusters} n_clusters (Best Scenario)"
        )

        plot_voronoi_diagram(
            points_data,
            labels_data,
            worst_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for {worst_clusters} n_clusters (Worst Scenario)"
        )

        digits = range(2, 9)
        plt.plot(digits, silhouette_scores, marker='', label='silhouette score')
        plt.title(f'Dataset {index + 1}. Silhouette Score')
        plt.xlabel("n-clusters")
        plt.ylabel("silhouette score")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.show()


def run_second_experiment():
    pass


def run_third_experiment():
    pass


def run_fourth_experiment():
    pass


if __name__ == "__main__":
    try:
        os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    except:
        pass

    run_first_experiment()