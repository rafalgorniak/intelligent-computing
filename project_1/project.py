import os
import numpy as np
from typing import List
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, \
    v_measure_score
from sklearn.preprocessing import StandardScaler
from warmup import plot_voronoi_diagram, load


def load_datasets() -> tuple:
    points = []
    labels = []

    dataset_first_indexes: List[float] = [1, 2]
    dataset_second_indexes: List[float] = [1, 2, 3]

    for first_index in dataset_first_indexes:
        for second_index in dataset_second_indexes:
            temp_points, temp_labels = load(f"{first_index}_{second_index}.csv")
            temp_points = StandardScaler().fit_transform(temp_points)
            points.append(temp_points)
            labels.append(temp_labels)

    return points, labels


def run_first_experiment() -> None:
    points, labels = load_datasets()

    for index, (points_data, labels_data) in enumerate(zip(points, labels)):
        best_silhouette_score: float = float('-inf')
        worst_silhouette_score: float = float('inf')
        best_n_clusters: int = -1
        worst_n_clusters: int = -1
        best_cluster_labels: np.ndarray = np.array([])
        worst_cluster_labels: np.ndarray = np.array([])
        silhouette_scores: List[float] = []
        n_clusters_range: range = range(2, 10)

        for n_clusters in n_clusters_range:
            kmeans = cluster.KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(points_data)

            silhouette = silhouette_score(points_data, cluster_labels)
            silhouette_scores.append(silhouette)

            if silhouette > best_silhouette_score:
                best_silhouette_score = silhouette
                best_cluster_labels = cluster_labels
                best_n_clusters = n_clusters

            if silhouette < worst_silhouette_score:
                worst_silhouette_score = silhouette
                worst_cluster_labels = cluster_labels
                worst_n_clusters = n_clusters

        plot_voronoi_diagram(
            points_data,
            labels_data,
            best_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for {best_n_clusters} n_clusters (Best Scenario)"
        )

        plot_voronoi_diagram(
            points_data,
            labels_data,
            worst_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for {worst_n_clusters} n_clusters (Worst Scenario)"
        )

        plt.plot(n_clusters_range, silhouette_scores, marker='', label='silhouette score')
        plt.title(f'Dataset {index + 1}. Silhouette Score')
        plt.xlabel("n-clusters")
        plt.ylabel("silhouette score")
        plt.ylim(0.0, 1.0)
        plt.xticks(np.arange(2, 10, 1))

        for i, n_cluster in enumerate(n_clusters_range):
            plt.axvline(x=n_cluster, color='gray', linestyle='--', alpha=0.3)

        plt.show()


def run_second_experiment() -> None:
    points, labels = load_datasets()

    for index, (points_data, labels_data) in enumerate(zip(points, labels)):
        best_silhouette_score: float = float('-inf')
        worst_silhouette_score: float = float('inf')
        best_eps: float = -1.0
        worst_eps: float = -1.0
        best_cluster_labels: np.ndarray = np.array([])
        worst_cluster_labels: np.ndarray = np.array([])
        silhouette_scores: List[float] = []
        clusters: List[int] = []

        eps_range: np.ndarray = np.arange(0.05, 0.5, 0.05)

        for eps in eps_range:
            dbscan = cluster.DBSCAN(eps=eps, min_samples=1)

            cluster_labels = dbscan.fit_predict(points_data)
            dbscan_labels = dbscan.labels_.astype(int)

            dbscan_unique_labels = len(np.unique(dbscan_labels))
            clusters.append(dbscan_unique_labels)

            silhouette = silhouette_score(points_data, cluster_labels)
            silhouette_scores.append(silhouette)

            if silhouette > best_silhouette_score:
                best_silhouette_score = silhouette
                best_cluster_labels = cluster_labels
                best_eps = eps

            if silhouette < worst_silhouette_score:
                worst_silhouette_score = silhouette
                worst_eps = eps
                worst_cluster_labels = cluster_labels

        plot_voronoi_diagram(
            points_data,
            labels_data,
            best_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for eps={best_eps} (Best Scenario)"
        )

        plot_voronoi_diagram(
            points_data,
            labels_data,
            worst_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for eps={worst_eps} (Worst Scenario)"
        )

        plt.plot(eps_range, silhouette_scores, marker='')
        plt.title(f'Dataset {index + 1}. Silhouette Score')
        plt.xlabel("eps")
        plt.ylabel("silhouette score")
        plt.ylim(0.0, 1.0)
        plt.xticks(np.arange(0.05, 0.5, 0.05))

        for i, eps in enumerate(eps_range):
            plt.axvline(x=eps, color='gray', linestyle='--', alpha=0.3)
            plt.annotate(str(clusters[i]), (eps, 0.1), ha='center')

        plt.show()


def run_third_experiment() -> None:
    points, labels = load_datasets()

    for index, (points_data, labels_data) in enumerate(zip(points, labels)):
        best_calculated_mean_score = float('-inf')
        worst_calculated_mean_score = float('inf')
        best_num_clusters: int = -1
        worst_num_clusters: int = -1
        adjusted_rand_score_list: List[float] = []
        homogeneity_score_list: List[float] = []
        completeness_score_list: List[float] = []
        v_measure_score_one_list: List[float] = []
        v_measure_score_two_list: List[float] = []
        v_measure_score_three_list: List[float] = []
        n_clusters_range: range = range(2, 10)

        for i, num_clusters in enumerate(n_clusters_range):
            kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(points_data)

            adjusted_rand_score_list.append(
                adjusted_rand_score(labels_data, cluster_labels)
            )
            homogeneity_score_list.append(
                homogeneity_score(labels_data, cluster_labels)
            )
            completeness_score_list.append(
                completeness_score(labels_data, cluster_labels)
            )
            v_measure_score_one_list.append(
                v_measure_score(labels_data, cluster_labels, beta=0.5)
            )
            v_measure_score_two_list.append(
                v_measure_score(labels_data, cluster_labels, beta=1.0)
            )
            v_measure_score_three_list.append(
                v_measure_score(labels_data, cluster_labels, beta=2.0)
            )

            calculated_mean_score = np.mean(
                [adjusted_rand_score_list[i],
                 homogeneity_score_list[i],
                 completeness_score_list[i],
                 v_measure_score_one_list[i]]
            )

            if calculated_mean_score > best_calculated_mean_score:
                best_calculated_mean_score = calculated_mean_score
                best_num_clusters = num_clusters
            if calculated_mean_score < worst_calculated_mean_score:
                worst_calculated_mean_score = calculated_mean_score
                worst_num_clusters = num_clusters

        algorithm = cluster.KMeans(n_clusters=best_num_clusters)
        algorithm.fit(points_data)
        best_cluster_labels = algorithm.labels_.astype(int)

        plot_voronoi_diagram(
            points_data,
            labels_data,
            best_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for {best_num_clusters} n_clusters (Best Scenario)"
        )

        algorithm = cluster.KMeans(n_clusters=worst_num_clusters)
        algorithm.fit(points_data)
        worst_cluster_labels = algorithm.labels_.astype(int)

        plot_voronoi_diagram(
            points_data,
            labels_data,
            worst_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for {worst_num_clusters} n_clusters (Worst Scenario)"
        )

        plt.plot(n_clusters_range, adjusted_rand_score_list, marker='', label='adjusted rand')
        plt.plot(n_clusters_range, homogeneity_score_list, marker='', label='homogeneity')
        plt.plot(n_clusters_range, completeness_score_list, marker='', label='completeness')
        plt.plot(n_clusters_range, v_measure_score_one_list, marker='', label='V-measure (beta=0.5)')
        plt.plot(n_clusters_range, v_measure_score_two_list, marker='', label='V-measure (beta=1.0)')
        plt.plot(n_clusters_range, v_measure_score_three_list, marker='', label='V-measure (beta=2.0)')
        plt.xlabel('n_clusters')
        plt.ylabel('score')
        plt.title(f'Dataset {index + 1}. Scores of measurements (Kmeans Method)')
        plt.legend(loc='upper right')
        plt.ylim(0.0, 1.0)
        plt.xticks(np.arange(2, 10, 1))

        for i, n_cluster in enumerate(n_clusters_range):
            plt.axvline(x=n_cluster, color='gray', linestyle='--', alpha=0.3)

        plt.show()


def run_fourth_experiment() -> None:
    points, labels = load_datasets()

    for index, (points_data, labels_data) in enumerate(zip(points, labels)):
        best_calculated_mean_score = float('-inf')
        worst_calculated_mean_score = float('inf')
        best_eps = None
        worst_eps = None
        best_cluster_labels: np.ndarray = np.array([])
        worst_cluster_labels: np.ndarray = np.array([])
        adjusted_rand_score_list: List[float] = []
        homogeneity_score_list: List[float] = []
        completeness_score_list: List[float] = []
        v_measure_score_one_list: List[float] = []
        v_measure_score_two_list: List[float] = []
        v_measure_score_three_list: List[float] = []
        clusters: List[int] = []
        eps_range: np.ndarray = np.arange(0.05, 0.5, 0.05)

        for i, eps in enumerate(eps_range):
            dbscan = cluster.DBSCAN(eps=eps, min_samples=1)

            cluster_labels = dbscan.fit_predict(points_data)
            dbscan_labels = dbscan.labels_.astype(int)

            dbscan_unique_labels = len(np.unique(dbscan_labels))
            clusters.append(dbscan_unique_labels)

            adjusted_rand_score_list.append(
                adjusted_rand_score(labels_data, cluster_labels)
            )
            homogeneity_score_list.append(
                homogeneity_score(labels_data, cluster_labels)
            )
            completeness_score_list.append(
                completeness_score(labels_data, cluster_labels))
            v_measure_score_one_list.append(
                v_measure_score(labels_data, cluster_labels, beta=0.5)
            )
            v_measure_score_two_list.append(
                v_measure_score(labels_data, cluster_labels, beta=1.0)
            )
            v_measure_score_three_list.append(
                v_measure_score(labels_data, cluster_labels, beta=2.0)
            )

            calculated_mean_score = np.mean(
                [adjusted_rand_score_list[i],
                 homogeneity_score_list[i],
                 completeness_score_list[i],
                 v_measure_score_one_list[i]]
            )

            if calculated_mean_score > best_calculated_mean_score:
                best_calculated_mean_score = calculated_mean_score
                best_cluster_labels = cluster_labels
                best_eps = eps
            if calculated_mean_score < worst_calculated_mean_score:
                worst_calculated_mean_score = calculated_mean_score
                worst_cluster_labels = cluster_labels
                worst_eps = eps

        plot_voronoi_diagram(
            points_data,
            labels_data,
            best_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for eps={best_eps} (Best Scenario)"
        )

        plot_voronoi_diagram(
            points_data,
            labels_data,
            worst_cluster_labels,
            f"Dataset {index + 1}. Voronoi Diagram for eps={worst_eps} (Worst Scenario)"
        )

        plt.plot(eps_range, adjusted_rand_score_list, marker='', label='adjusted rand')
        plt.plot(eps_range, homogeneity_score_list, marker='', label='homogeneity')
        plt.plot(eps_range, completeness_score_list, marker='', label='completeness')
        plt.plot(eps_range, v_measure_score_one_list, marker='', label='V-measure (beta=0.5)')
        plt.plot(eps_range, v_measure_score_two_list, marker='', label='V-measure (beta=1.0)')
        plt.plot(eps_range, v_measure_score_three_list, marker='', label='V-measure (beta=2.0)')

        plt.title(f'Dataset {index + 1}. Scores of measurements (DBSCAN Method)')
        plt.xlabel("eps")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.0)
        plt.legend(loc='upper right')
        plt.xticks(np.arange(0.05, 0.5, 0.05))

        for i, eps in enumerate(eps_range):
            plt.axvline(x=eps, color='gray', linestyle='--', alpha=0.3)
            plt.annotate(str(clusters[i]), (eps, 0.1), ha='center')

        plt.show()


def run_sixth_experiment():
    pass


if __name__ == "__main__":
    try:
        os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    except:
        pass

    #run_first_experiment()
    #run_second_experiment()
    #run_third_experiment()
    #run_fourth_experiment()