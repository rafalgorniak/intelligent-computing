import numpy as np
import torch
from skimage.feature import hog
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from extraction_type import ExtractionType


class MNISTCustomDataset(Dataset):
    def __init__(self, train: bool, feature_extraction: str, pca=None, lda=None, limit: int = None):
        self.mnist_dataset = datasets.MNIST(root='./data', train=train, transform=transforms.ToTensor(), download=True)
        if limit:
            self.mnist_dataset = Subset(self.mnist_dataset, range(limit))
        self.feature_extraction = feature_extraction
        self.pca = pca
        self.lda = lda

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        img = img.view(-1)

        if self.feature_extraction == ExtractionType.HOG:
            img = self.extract_hog(img)
        elif self.feature_extraction == ExtractionType.LDA and self.lda is not None:
            img = self.lda.transform(img.numpy().reshape(1, -1)).flatten()
        elif self.feature_extraction == ExtractionType.PCA and self.pca is not None:  # Check for PCA
            img = self.pca.transform(img.numpy().reshape(1, -1)).flatten()

        return torch.tensor(img, dtype=torch.float32), label

    @staticmethod
    def extract_hog(img: torch.Tensor) -> np.ndarray:
        img_np = img.numpy().reshape(28, 28)
        hog_features = hog(img_np, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
        return hog_features
