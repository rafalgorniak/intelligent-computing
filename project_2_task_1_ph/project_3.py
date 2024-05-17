import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
)
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.spatial import Voronoi, voronoi_plot_2d
import os

from custom_dataset import CustomDataset
from custom_mlp import CustomMLP
from extraction_type import ExtractionType
from mnist_custom_dataset import MNISTCustomDataset
from torchvision import datasets
from torchvision.transforms import transforms
from sklearn.svm import SVC
from sklearn.utils import resample


def train_model(model, train_loader, test_loader, loss_function, optimizer, device, num_epochs):
    model.to(device)
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device).float(), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train_samples += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_train_predictions / total_train_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f} - Train Accuracy: {train_accuracy:.4f}')

        train_accuracies.append(train_accuracy)

        _, _, test_accuracy = test_model(model, test_loader, device)
        test_accuracies.append(test_accuracy / 100)

    # Ensure the original return values are preserved
    evaluation_results = evaluate_model(model, train_loader, test_loader, device)
    return train_accuracies, test_accuracies, evaluation_results


def test_model(model, dataloader, device):
    model.eval()
    total_samples = 0
    correct_predictions = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device).float(), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = (correct_predictions / total_samples) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

    return all_labels, all_predictions, accuracy


def scorePlot(epoch_range, train_accuracies, test_accuracies, x_label='Epoch'):
    plt.plot(epoch_range, train_accuracies, marker='', label='Train Accuracy')
    plt.plot(epoch_range, test_accuracies, marker='', label='Test Accuracy')

    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='lower right')

    plt.show()


def initialize_model(input_size, hidden_size, output_size, device):
    model = CustomMLP(input_size, hidden_size, output_size)
    model.to(device)
    return model


def evaluate_model(model, train_loader, test_loader, device):
    model.eval()
    with torch.no_grad():
        train_features, train_labels, train_predictions = [], [], []
        test_features, test_labels, test_predictions = [], [], []

        for images, labels in train_loader:
            images, labels = images.to(device).float(), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            train_features.append(images.cpu().numpy())
            train_labels.append(labels.cpu().numpy())
            train_predictions.append(predicted.cpu().numpy())

        train_features = np.concatenate(train_features)
        train_labels = np.concatenate(train_labels)
        train_predictions = np.concatenate(train_predictions)

        for images, labels in test_loader:
            images, labels = images.to(device).float(), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_features.append(images.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_predictions.append(predicted.cpu().numpy())

        test_features = np.concatenate(test_features)
        test_labels = np.concatenate(test_labels)
        test_predictions = np.concatenate(test_predictions)

        adjusted_rand = adjusted_rand_score(test_labels, test_predictions)
        homogeneity = homogeneity_score(test_labels, test_predictions)
        completeness = completeness_score(test_labels, test_predictions)
        v_measure = v_measure_score(test_labels, test_predictions)

        print(f'Adjusted Rand Score: {adjusted_rand:.2f}')
        print(f'Homogeneity Score: {homogeneity:.2f}')
        print(f'Completeness Score: {completeness:.2f}')
        print(f'V-Measure Score: {v_measure:.2f}')

    return train_labels, train_predictions, test_labels, test_predictions, train_features, test_features


def plot_voronoi_diagram(X: np.ndarray, y_true_labels: np.ndarray, dataset_name: str, extraction_method: str, title: str = 'Voronoi Diagram', num_points: int = 5000) -> None:
    # Downsample the dataset if it contains more points than specified
    if len(X) > num_points:
        X, y_true_labels = resample(X, y_true_labels, n_samples=num_points, random_state=42)

    plot_margin = 0.1
    x1_max = X[:, 0].max() + plot_margin
    x1_min = X[:, 0].min() - plot_margin
    x2_max = X[:, 1].max() + plot_margin
    x2_min = X[:, 1].min() - plot_margin

    border_points = [
        [999, 999],
        [-999, 999],
        [999, -999],
        [-999, -999]
    ]

    X = np.append(X, border_points, axis=0)
    y_true_labels = np.append(y_true_labels, [0, 0, 0, 0])

    group_colors = sns.color_palette("hsv", len(np.unique(y_true_labels)))

    voronoi = Voronoi(X)
    fig, ax = plt.subplots()
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_width=1, line_alpha=0.4)

    for region_index in range(len(voronoi.point_region)):
        region = voronoi.regions[voronoi.point_region[region_index]]
        if not -1 in region:
            polygon = [voronoi.vertices[i] for i in region]
            point_label = y_true_labels[region_index]
            color = group_colors[point_label % len(group_colors)]
            ax.fill(*zip(*polygon), color=color, alpha=0.4)

    unique_true_labels = np.unique(y_true_labels)

    for i, label in enumerate(unique_true_labels):
        idx = np.where(y_true_labels == label)[0]
        color = group_colors[i % len(group_colors)]
        ax.plot(X[idx, 0], X[idx, 1], 'o', color=color, markersize=5, alpha=1, label=str(label))

    plt.xlim([x1_min, x1_max])
    plt.ylim([x2_min, x2_max])
    plt.title(f'{title} - {dataset_name} - {extraction_method}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Classes")
    plt.show()


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, dataset_name: str, extraction_method: str, title: str = 'Decision Boundary', num_points: int =5000) -> None:

    if len(X) > num_points:
        X, y = resample(X, y, n_samples=num_points, random_state=42)

    # Fit SVM classifier
    clf = SVC(kernel='linear')
    clf.fit(X, y)

    # Plot decision boundary
    plot_margin = 0.1
    x1_max = X[:, 0].max() + plot_margin
    x1_min = X[:, 0].min() - plot_margin
    x2_max = X[:, 1].max() + plot_margin
    x2_min = X[:, 1].min() - plot_margin

    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)

    # Plot data points
    group_colors = sns.color_palette("hsv", len(np.unique(y)))
    for i, label in enumerate(np.unique(y)):
        idx = np.where(y == label)[0]
        color = group_colors[i % len(group_colors)]
        plt.scatter(X[idx, 0], X[idx, 1], color=color, label=str(label))

    plt.xlim([x1_min, x1_max])
    plt.ylim([x2_min, x2_max])
    plt.title(f'{title} - {dataset_name} - {extraction_method}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Classes")
    plt.show()


def display_plots(train_features, train_labels, test_features, test_labels, train_predictions, test_predictions, dataset_name, extraction_method, lda=None):
    conf_matrix_train = confusion_matrix(train_labels, train_predictions)
    conf_matrix_test = confusion_matrix(test_labels, test_predictions)

    display_train = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_train, display_labels=np.unique(train_labels))
    display_test = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=np.unique(test_labels))

    display_train.plot(cmap=plt.cm.Blues)
    plt.title(f"Train Confusion Matrix - {dataset_name} - {extraction_method}")
    plt.show()

    display_test.plot(cmap=plt.cm.Blues)
    plt.title(f"Test Confusion Matrix - {dataset_name} - {extraction_method}")
    plt.show()

    if train_features.shape[1] == 2:
        plot_voronoi_diagram(train_features, train_labels, dataset_name, extraction_method, title='Voronoi Diagram - Train')
        plot_voronoi_diagram(test_features, test_labels, dataset_name, extraction_method, title='Voronoi Diagram - Test')
        plot_decision_boundary(train_features, train_labels, dataset_name, extraction_method,
                             title='Decision Boundary - Train')
        plot_decision_boundary(test_features, test_labels, dataset_name, extraction_method,
                             title='Decision Boundary - Test')



def getTSNETrain():
    mnist_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_images = mnist_data.data.numpy().reshape(-1, 28 * 28)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_representation = tsne.fit_transform(train_images)
    return torch.tensor(tsne_representation.astype('float32')), mnist_data.targets


def getTSNETest():
    mnist_data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    test_images = mnist_data.data.numpy().reshape(-1, 28 * 28)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_representation = tsne.fit_transform(test_images)
    return torch.tensor(tsne_representation.astype('float32')), mnist_data.targets


def load_datasets(extraction_method, batch_size, dataset_name):
    lda = None

    if dataset_name == 'MNIST':
        train_dataset = MNISTCustomDataset(train=True, feature_extraction=ExtractionType.FLATTEN)

        if extraction_method == ExtractionType.LDA:
            flattened_imgs = [img.numpy() for img, _ in train_dataset]
            labels = [label for _, label in train_dataset]
            lda = LDA(n_components=2)
            lda.fit(flattened_imgs, labels)

            train_dataset = MNISTCustomDataset(train=True, feature_extraction=extraction_method, lda=lda)
            test_dataset = MNISTCustomDataset(train=False, feature_extraction=extraction_method, lda=lda)

        elif extraction_method == ExtractionType.TSNE:
            train_data, train_labels = getTSNETrain()
            test_data, test_labels = getTSNETest()

            train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
            test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

        elif extraction_method == ExtractionType.PCA:
            flattened_imgs = [img.numpy() for img, _ in train_dataset]
            labels = [label for _, label in train_dataset]
            pca = PCA(n_components=2)  # Initialize PCA
            pca.fit(flattened_imgs, labels)

            train_dataset = MNISTCustomDataset(train=True, feature_extraction=extraction_method, pca=pca)
            test_dataset = MNISTCustomDataset(train=False, feature_extraction=extraction_method, pca=pca)

        else:
            train_dataset = MNISTCustomDataset(train=True, feature_extraction=extraction_method)
            test_dataset = MNISTCustomDataset(train=False, feature_extraction=extraction_method)

        input_size = train_dataset[0][0].shape[0]
        output_size = 10

    else:
        if dataset_name == 'Iris':
            data = load_iris()
        elif dataset_name == 'Wine':
            data = load_wine()
        elif dataset_name == 'Breast Cancer':
            data = load_breast_cancer()

        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        if extraction_method == ExtractionType.LDA:
            n_components = min(X_train.shape[1], len(np.unique(y_train)) - 1)
            lda = LDA(n_components=n_components)
            X_train = lda.fit_transform(X_train, y_train)
            X_test = lda.transform(X_test)

        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)

        input_size = X_train.shape[1]
        output_size = len(np.unique(y_train))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, input_size, output_size, lda


def save_model(model, filename):
    torch.save(model.state_dict(), f'data/{filename}')
    print(f'Model saved to {filename}')


def load_model_if_exists(model, filename):
    if os.path.exists(f'data/{filename}'):
        model.load_state_dict(torch.load(f'data/{filename}'))
        model.eval()
        print(f'Model loaded from {filename}')

        return True

    return False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.01
hidden_size = 100
batch_size = 32

#datasets_list = ['MNIST', 'Iris', 'Wine', 'Breast Cancer']
#feature_extraction_methods = [ExtractionType.FLATTEN, ExtractionType.LDA, ExtractionType.HOG, ExtractionType.TSNE, ExtractionType.PCA]
datasets_list = ['MNIST']
feature_extraction_methods = [ExtractionType.PCA, ExtractionType.TSNE]

epochs_dict = {
    'MNIST': 10,
    'Iris': 20,
    'Wine': 20,
    'Breast Cancer': 20
}

for dataset_name in datasets_list:
    for extraction_method in feature_extraction_methods:
        print(f'\nDataset: {dataset_name}, Feature Extraction Method: {extraction_method}')

        train_loader, test_loader, input_size, output_size, lda = load_datasets(extraction_method, batch_size, dataset_name=dataset_name)
        print(f'\nNumber of training examples: {len(train_loader)}')
        model = initialize_model(input_size, hidden_size, output_size, device)
        model_filename = f'model_{dataset_name}_{extraction_method}.pth'

        num_epochs = epochs_dict[dataset_name]

        if not load_model_if_exists(model, model_filename):
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            train_accuracies, test_accuracies, train_labels, train_predictions, test_labels, test_predictions, train_features, test_features = train_model(
                model, train_loader, test_loader, loss_function, optimizer, device, num_epochs)

            scorePlot(range(1, num_epochs + 1), train_accuracies, test_accuracies, 'Epoch')

            print(f'Testing model with feature extraction method: {extraction_method}')
            display_plots(train_features, train_labels, test_features, test_labels, train_predictions, test_predictions, dataset_name, extraction_method, lda)

            save_model(model, model_filename)
        else:
            print(f'Model {model_filename} has been loaded.')

            train_labels, train_predictions, test_labels, test_predictions, train_features, test_features = evaluate_model(
                model, train_loader, test_loader, device)

            display_plots(train_features, train_labels, test_features, test_labels, train_predictions, test_predictions, dataset_name, extraction_method, lda)
