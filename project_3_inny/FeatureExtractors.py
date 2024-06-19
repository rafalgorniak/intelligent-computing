import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from skimage.feature import hog
import numpy as np


def apply_lda(dataset):
    X = dataset.data.numpy().reshape(len(dataset), -1)
    y = dataset.targets.numpy()
    lda = LDA(n_components=2)
    X_transformed = lda.fit_transform(X, y)
    return torch.tensor(X_transformed, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def apply_hog(dataset):
    X = dataset.data.numpy()
    hog_features = []
    for img in X:
        img = img.reshape((28, 28))
        hog_feat = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(hog_feat)
    X_transformed = np.array(hog_features)
    y = dataset.targets.numpy()
    return torch.tensor(X_transformed, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def apply_pca(dataset):
    X = dataset.data.numpy().reshape(len(dataset), -1)
    y = dataset.targets.numpy()
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X, y)
    return torch.tensor(X_transformed, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

