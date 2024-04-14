import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

from sklearn.metrics import silhouette_score,adjusted_rand_score,homogeneity_score,completeness_score,v_measure_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from sklearn.svm import SVC

from warmUp.warmup import plot_voronoi_diagram,load

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
    for index, (points_data, labels_data) in enumerate(zip(data, labels)):
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))
        clf.fit(points_data, labels_data)
        predicted_labels = clf.predict(points_data)
        plot_voronoi_diagram(points_data,labels_data,predicted_labels)


if __name__ == '__main__':

    pageOne()
