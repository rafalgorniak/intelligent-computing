import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_accuracy(train_accuracies, test_accuracies, title):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_augmentations(dataset, augmentation, num_samples=5):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(num_samples):
        sample, _ = dataset[i]
        axes[0, i].imshow(sample.permute(1, 2, 0))
        axes[0, i].axis('off')

        augmented_sample = augmentation(sample)
        axes[1, i].imshow(augmented_sample.permute(1, 2, 0))
        axes[1, i].axis('off')

    plt.show()


def plot_feature_distribution(dataset, model, title='Feature Distribution'):
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for data, label in dataset:
            data = data.unsqueeze(0).to(device)
            feature = model(data)
            features.append(feature.cpu().numpy())
            labels.append(label)

    features = np.concatenate(features)
    labels = np.array(labels)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette='tab10')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()


def plot_confusion_matrix(true_labels, pred_labels, classes, title):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()