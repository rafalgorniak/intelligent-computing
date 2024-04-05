import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

from sklearn.metrics import silhouette_score,adjusted_rand_score,homogeneity_score,completeness_score,v_measure_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris,load_wine,load_breast_cancer

from warmUp.warmup import plot_voronoi_diagram,load



def runScorePlot(data, y_true,min_eps,max_eps,gap,title):
    X = StandardScaler().fit_transform(data)

    eps_range = np.arange(min_eps, max_eps, gap)
    clusters = []

    adjusted_score = []
    hom_score = []
    comp_score = []
    v_meas_1_score = []
    v_meas_2_score = []
    v_meas_3_score = []

    worst_label, best_label = [], []
    best_case, worst_case = 0, 0
    best_score, worst_score = 0, 1

    for i, eps in enumerate(eps_range):
        # Fit KMeans clustering algorithm
        algorithm = cluster.DBSCAN(eps=eps, min_samples=1)
        cluster_labels = algorithm.fit_predict(X)

        y_pred = algorithm.labels_.astype(int)
        num_unique_labels = len(np.unique(y_pred))
        clusters.append(num_unique_labels)

        adjusted_score.append(adjusted_rand_score(y_true, cluster_labels))
        hom_score.append(homogeneity_score(y_true, cluster_labels))
        comp_score.append(completeness_score(y_true, cluster_labels))
        v_meas_1_score.append(v_measure_score(y_true, cluster_labels, beta=0.5))
        v_meas_2_score.append(v_measure_score(y_true, cluster_labels, beta=1.0))
        v_meas_3_score.append(v_measure_score(y_true, cluster_labels, beta=2.0))

        score = np.mean([adjusted_score[i], hom_score[i], comp_score[i], v_meas_1_score[i]])

        if score > best_score:
            best_score = score
            best_label = cluster_labels
        if score < worst_score:
            worst_score = score
            worst_label = cluster_labels

    plt.plot(eps_range, adjusted_score, marker='', label='adjusted rand')
    plt.plot(eps_range, hom_score, marker='', label='homogeneity')
    plt.plot(eps_range, comp_score, marker='', label='completeness')
    plt.plot(eps_range, v_meas_1_score, marker='', label='V-measure (beta=0.5)')
    plt.plot(eps_range, v_meas_2_score, marker='', label='V-measure (beta=1.0)')
    plt.plot(eps_range, v_meas_3_score, marker='', label='V-measure (beta=2.0)')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    clusters = np.array(clusters)

    # Add labels indicating number of clusters near marker points
    for i, eps in enumerate(eps_range):
        plt.annotate(clusters[i], (eps, worst_score), textcoords="offset points",
                     xytext=(5, -10), ha='center')

    plt.show()

    return worst_label, best_label

def runScatterPlot(data, data_labels, title):
    plt.figure(figsize=(6, 4))
    plt.scatter(data[:, 0], data[:, 1], c=data_labels, cmap='viridis', edgecolor='k')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def run_first_task():
    for a in range(6):
        path = f"project_1/{int(a/3)+1}_{a%3+1}.csv"
        X = load(path)[0]
        y_true = load(path)[1]
        X = StandardScaler().fit_transform(X)

        min_clusters = 2
        max_clusters = 9
        cluster_range = range(min_clusters, max_clusters + 1)
        silhouette_scores = []

        for n_clusters in cluster_range:
            # Fit KMeans clustering algorithm
            kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(X)

            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        plt.plot(cluster_range, silhouette_scores, marker='')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.ylim(0.0, 1.0)
        plt.title('Silhouette Score vs Number of Clusters')

        for i, n_cluster in enumerate(cluster_range):
            plt.axvline(x=n_cluster, color='gray', linestyle='--', alpha=0.3)

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


def run_second_task():
    for a in range(6):

        path = f"project_1/{int(a / 3) + 1}_{a % 3 + 1}.csv"
        X = load(path)[0]
        y_true = load(path)[1]
        X = StandardScaler().fit_transform(X)

        min_eps = 0.05
        max_eps = 0.5
        eps_range = np.arange(min_eps, max_eps,0.05)
        silhouette_scores = []
        clusters = []
        best_labels,worst_labels= [], []
        best_case, worst_case = 0, 0
        best_score,worst_score = 0, 1

        for eps in eps_range:
            # Fit KMeans clustering algorithm
            algorithm = cluster.DBSCAN(eps=eps, min_samples=1)
            cluster_labels = algorithm.fit_predict(X)

            y_pred = algorithm.labels_.astype(int)
            num_unique_labels = len(np.unique(y_pred))

            clusters.append(num_unique_labels)

            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_labels = cluster_labels

            if silhouette_avg < worst_score:
                worst_score=silhouette_avg
                worst_labels = cluster_labels

        plt.plot(eps_range, silhouette_scores, marker='')
        plt.xlabel('eps')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.ylim(0.0, 1.0)
        plt.title('Silhouette Score vs Number of Clusters')

        for i, n_cluster in enumerate(eps_range):
            plt.axvline(x=n_cluster, color='gray', linestyle='--', alpha=0.3)
        clusters=np.array(clusters)

        # Add labels indicating number of clusters near marker points
        for i,eps in enumerate(eps_range):
            plt.annotate(clusters[i], (eps, silhouette_scores[i]), textcoords="offset points",
                         xytext=(5, -10), ha='center')

        plt.show()

        plot_voronoi_diagram(X, y_true, worst_labels)

        plot_voronoi_diagram(X, y_true, best_labels)


def run_third_task():
    for a in range(6):
        path = f"project_1/{int(a / 3) + 1}_{a % 3 + 1}.csv"
        X = load(path)[0]
        y_true = load(path)[1]
        X = StandardScaler().fit_transform(X)

        min_clusters = 2
        max_clusters = 9
        cluster_range = range(min_clusters, max_clusters + 1)

        adjusted_score = []
        hom_score = []
        comp_score = []
        v_meas_1_score = []
        v_meas_2_score = []
        v_meas_3_score = []

        best_case, worst_case = 0, 0
        best_score, worst_score = 0, 1

        for i, n_clusters in enumerate(cluster_range):
            # Fit KMeans clustering algorithm
            kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(X)

            y_pred = kmeans.labels_.astype(int)
            num_unique_labels = len(np.unique(y_pred))

            adjusted_score.append(adjusted_rand_score(y_true,cluster_labels))
            hom_score.append(homogeneity_score(y_true,cluster_labels))
            comp_score.append(completeness_score(y_true, cluster_labels))
            v_meas_1_score.append(v_measure_score(y_true, cluster_labels, beta=0.5))
            v_meas_2_score.append(v_measure_score(y_true, cluster_labels, beta=1.0))
            v_meas_3_score.append(v_measure_score(y_true, cluster_labels, beta=2.0))

            score = np.mean([adjusted_score[i],hom_score[i],comp_score[i],v_meas_1_score[i]])

            if score > best_score:
                best_score = score
                best_case = num_unique_labels
            if score < worst_score:
                worst_score = score
                worst_case = num_unique_labels

        plt.plot(cluster_range, adjusted_score, marker='.', label='Adjusted rand')
        plt.plot(cluster_range, hom_score, marker='.', label='Homogeneity')
        plt.plot(cluster_range, comp_score, marker='.', label='Completeness')
        plt.plot(cluster_range, v_meas_1_score, marker='.', label='V measure (beta=0.5)')
        plt.plot(cluster_range, v_meas_2_score, marker='.', label='V measure (beta=1.0)')
        plt.plot(cluster_range, v_meas_3_score, marker='.', label='V measure (beta=2.0)')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.legend()
        plt.ylim(0.0, 1.0)

        for i, n_cluster in enumerate(cluster_range):
            plt.axvline(x=n_cluster, color='gray', linestyle='--', alpha=0.3)

        plt.show()

        algorithm = cluster.KMeans(n_clusters=best_case)
        algorithm.fit(X)
        y_pred = algorithm.labels_.astype(int)

        plot_voronoi_diagram(X, y_true, y_pred)

        algorithm = cluster.KMeans(n_clusters=worst_case)
        algorithm.fit(X)
        y_pred = algorithm.labels_.astype(int)

        plot_voronoi_diagram(X, y_true, y_pred)


def run_fourth_task():
    for a in range(6):
        path = f"project_1/{int(a / 3) + 1}_{a % 3 + 1}.csv"
        X = load(path)[0]
        y_true = load(path)[1]
        worst, best = runScorePlot(X,y_true,0.05,0.5,0.05,"")
        plot_voronoi_diagram(X, y_true, worst)
        plot_voronoi_diagram(X, y_true, best)


def run_sixth_task():
    flowers = load_iris().data
    flower_labels = load_iris().target
    flowers = StandardScaler().fit_transform(flowers)

    wine = load_wine().data
    wine_labels = load_wine().target
    wine = StandardScaler().fit_transform(wine)

    breast = load_breast_cancer().data
    breast_labels = load_breast_cancer().target
    breast = StandardScaler().fit_transform(breast)

    min_clusters = 2
    max_clusters = 10
    cluster_range = range(min_clusters, max_clusters+1)
    silhouette_scores_flowers = []
    silhouette_scores_wine = []
    silhouette_scores_breast = []

    for n_clusters in cluster_range:
        # Fit KMeans clustering algorithm
        cluster_labels_flowers = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit_predict(flowers)
        cluster_labels_wine = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit_predict(wine)
        cluster_labels_breast = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit_predict(breast)

        silhouette_scores_flowers.append(silhouette_score(flowers, cluster_labels_flowers))
        silhouette_scores_wine.append(silhouette_score(wine, cluster_labels_wine))
        silhouette_scores_breast.append(silhouette_score(breast, cluster_labels_breast))

    plt.plot(cluster_range, silhouette_scores_flowers, marker='', label='iris')
    plt.plot(cluster_range, silhouette_scores_wine, marker='', label='wine')
    plt.plot(cluster_range, silhouette_scores_breast, marker='', label='breast cancer')
    plt.xlabel('n_clusters')
    plt.ylabel('silhouette score')
    plt.title('Silhouette scores of different datasets')
    plt.grid(True)
    plt.legend()
    plt.show()

    runScorePlot(flowers,flower_labels,0.3,1,0.1, "Scores of iris dataset")
    runScorePlot(wine, wine_labels,1.5,2.5,0.1, "Scores score of wine dataset")
    runScorePlot(breast, breast_labels,1.5,3,0.1, "Scores of breast cancer dataset")

    runScatterPlot(flowers, flower_labels, 'Scatter plot - Iris dataset')
    runScatterPlot(wine, wine_labels, 'Scatter plot - Wine dataset')
    runScatterPlot(breast, breast_labels, 'Scatter plot - Breast dataset')


if __name__ == '__main__':

    pass
    #run_first_task()
    #run_second_task()
    #run_third_task()
    #run_fourth_task()
    #run_sixth_task()

