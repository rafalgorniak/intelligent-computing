import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from warmUp.warmup import plot_voronoi_diagram,load

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

