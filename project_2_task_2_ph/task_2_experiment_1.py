import os
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from torch import optim, nn
from torchvision.transforms import transforms
import seaborn as sns

from models import LEARN_RATE, BATCH_SIZE, NUM_EPOCHS, MNISTCNN2, Cifar10CNN2, DatasetName, MNISTCNN, Cifar10CNN

def prepare_data_experiment_one(dataset=DatasetName.MNIST, two_features=False):
    if dataset == DatasetName.MNIST:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

        if two_features:
            model_path = 'data/best_mnist_two_features_model.pth'
        else:
            model_path = 'data/best_mnist_model.pth'

    elif dataset == DatasetName.CIFAR10:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        if two_features:
            model_path = 'data/best_cifar10_two_features_model.pth'
        else:
            model_path = 'data/best_cifar10_model.pth'

    return train_loader, test_loader, model_path


def train_model(model, criterion, optimizer, train_loader, test_loader):
    train_accuracy = []
    test_accuracy = []
    best_accuracy = 0.0
    best_model = None

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_accuracy = correct_train / total_train
        train_accuracy.append(epoch_train_accuracy)

        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

            epoch_test_accuracy = correct_test / total_test
            test_accuracy.append(epoch_test_accuracy)

            if epoch_test_accuracy > best_accuracy:
                best_accuracy = epoch_test_accuracy
                best_model = model.state_dict()

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(train_loader)}, '
              f'Train Accuracy: {epoch_train_accuracy}, Test Accuracy: {epoch_test_accuracy}')

    print(f'Best Test Accuracy: {best_accuracy}')
    return train_accuracy, test_accuracy, best_model


def evaluate_model(model: nn.Module, train_loader, test_loader, model_path, dataset_name, extraction_method, with_feature_scatter=False, visualised_sample_size=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded model from file.")
    else:
        train_accuracy, test_accuracy, best_model = train_model(model, criterion, optimizer, train_loader, test_loader)
        torch.save(best_model, model_path)
        print("Saved trained model to file.")

        plt.plot(train_accuracy, label='Train Accuracy')
        plt.plot(test_accuracy, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Accuracy scores - {dataset_name} {extraction_method}')
        plt.show()

    model.eval()

    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            pred_labels.extend(predicted.numpy())
            true_labels.extend(labels.numpy())

    cm = confusion_matrix(true_labels, pred_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(true_labels))
    display.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {dataset_name} {extraction_method}')
    plt.show()

    if with_feature_scatter:
        visualize_feature_scatter(model, test_loader, dataset_name, extraction_method, visualised_sample_size)


def visualize_feature_scatter(model, loader, dataset_name, extraction_method, sample_size=None):
    if not hasattr(model, 'forward_conv'):
        print("The model does not support feature scatter visualization.")
        return

    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, target in loader:
            output = model.forward_conv(inputs)
            features.extend(output.numpy())
            labels.extend(target.numpy())

    features = np.array(features)
    labels = np.array(labels)

    if sample_size is None:
        sample_size = len(features)

    sample_size = min(sample_size, len(features))
    sample_indices = np.random.choice(len(features), sample_size, replace=False)
    features_sampled = features[sample_indices]
    labels_sampled = labels[sample_indices]

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_sampled)

    plot_decision_boundary(features_2d, labels_sampled, dataset_name, extraction_method)


def plot_decision_boundary(X: np.ndarray, y_true_labels: np.ndarray, dataset_name: str, extraction_method: str, title: str = 'Decision Boundary') -> None:
    plot_margin: float = 0.1
    grid_step_size: float = 0.05

    x_min: float = X[:, 0].min() - plot_margin
    x_max: float = X[:, 0].max() + plot_margin
    y_min: float = X[:, 1].min() - plot_margin
    y_max: float = X[:, 1].max() + plot_margin

    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step_size), np.arange(y_min, y_max, grid_step_size))
    grid = np.c_[xx.ravel(), yy.ravel()]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y_true_labels)
    Z = knn.predict(grid)
    Z = Z.reshape(xx.shape)

    group_colors = sns.color_palette("hsv", len(np.unique(y_true_labels)))
    colormap = LinearSegmentedColormap.from_list('colormap', group_colors, N=256)

    plt.contourf(xx, yy, Z, cmap=colormap, alpha=0.4)
    plt.contour(xx, yy, Z, colors='k', linewidths=1)

    if y_true_labels is not None:
        true_label_colors = [group_colors[label] for label in y_true_labels]
        plt.scatter(X[:, 0], X[:, 1], c=true_label_colors, edgecolors='k')

    plt.title(f'{title} - {dataset_name} - {extraction_method}')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


if __name__ == '__main__':

    train_loader1, test_loader1, model_path1 = prepare_data_experiment_one(DatasetName.MNIST, two_features=True)
    train_loader2, test_loader2, model_path2 = prepare_data_experiment_one(DatasetName.CIFAR10, two_features=True)

    mnist_model_2 = MNISTCNN2()
    cifar10_model_2 = Cifar10CNN2()

    evaluate_model(mnist_model_2, train_loader1, test_loader1, model_path1, DatasetName.MNIST, "Two Features", with_feature_scatter=True, visualised_sample_size=None)
    evaluate_model(cifar10_model_2, train_loader2, test_loader2, model_path2, DatasetName.CIFAR10, "Two Features", with_feature_scatter=True, visualised_sample_size=None)

    train_loader3, test_loader3, model_path3 = prepare_data_experiment_one(DatasetName.MNIST, two_features=False)
    train_loader4, test_loader4, model_path4 = prepare_data_experiment_one(DatasetName.CIFAR10, two_features=False)

    mnist_model = MNISTCNN()
    cifar10_model = Cifar10CNN()

    evaluate_model(mnist_model, train_loader3, test_loader3, model_path3, DatasetName.MNIST, "", visualised_sample_size=None)
    evaluate_model(cifar10_model, train_loader4, test_loader4, model_path4, DatasetName.CIFAR10, "", visualised_sample_size=None)