import os
import random
import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms, RandomRotation, ColorJitter
from sklearn.decomposition import PCA

from models import DatasetName, BATCH_SIZE, LEARN_RATE, Cifar10CNN2, MNISTCNN2, Cifar10CNN, MNISTCNN, NUM_EPOCHS

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


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


def ensure_class_balance(indices, targets, num_samples_per_class):
    balanced_indices = []

    for i in range(10):
        class_indices = [idx for idx in indices if targets[idx] == i]
        balanced_indices.extend(random.sample(class_indices, min(num_samples_per_class, len(class_indices))))

    return balanced_indices


def get_transform(augment):
    transform_list = [transforms.ToTensor()]

    if augment == "rotation":
        transform_list = [RandomRotation(60)] + transform_list
    elif augment == "color_jitter":
        transform_list = [ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)] + transform_list

    transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    return transforms.Compose(transform_list)


def prepare_dataset(dataset, augment, num_samples, two_features):
    print(f"Preparing dataset: {dataset}, Augmentation: {augment}, Num samples: {num_samples}, Two features: {two_features}")

    transform = get_transform(augment)
    transform_original = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if dataset == DatasetName.MNIST:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        trainset_original = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_original)
    elif dataset == DatasetName.CIFAR10:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainset_original = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_original)

    if num_samples:
        print(f"Balancing classes for {num_samples} samples.")

        indices = ensure_class_balance(list(range(len(trainset))), trainset.targets, num_samples // 10)
        trainset = Subset(trainset, indices)
        trainset_original = Subset(trainset_original, indices)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    train_loader_original = DataLoader(trainset_original, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    dataset_name = dataset.lower()
    model_path = f"data/best_{dataset_name}_{'two_features_model' if two_features else 'model'}.pth"

    return train_loader, test_loader, train_loader_original, model_path


def evaluate_model_experiment_two(model: nn.Module, train_loader, test_loader, model_path, num_runs=10):
    print(f"Evaluating model: {model.__class__.__name__} with {num_runs} runs.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    accuracies = []

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}.")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"Training model.")
            train_accuracy, test_accuracy, best_model = train_model(model, criterion, optimizer, train_loader, test_loader)
            model.load_state_dict(best_model)
            torch.save(best_model, model_path)

        model.eval()

        pred_labels, true_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                pred_labels.extend(predicted.numpy())
                true_labels.extend(labels.numpy())

        test_accuracy = np.mean(np.array(pred_labels) == np.array(true_labels))
        accuracies.append(test_accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"Mean accuracy: {mean_accuracy}, Std accuracy: {std_accuracy}")

    return mean_accuracy, std_accuracy


def visualize_augmentations(train_loader_original, augmentation_method, dataset_name):

    print(f"Visualizing augmentations for {dataset_name} with method {augmentation_method}")

    dataiter = iter(train_loader_original)
    images, labels = next(dataiter)

    class_images = {i: None for i in range(10)}

    for img, label in zip(images, labels):
        if class_images[label.item()] is None:
            class_images[label.item()] = img
        if all(v is not None for v in class_images.values()):
            break

    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 6))

    if augmentation_method == "rotation":
        transform = RandomRotation(10)
    elif augmentation_method == "color_jitter":
        transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    else:
        transform = None

    for i in range(10):
        original_image = class_images[i]
        augmented_image = transform(original_image.unsqueeze(0)).squeeze(0) if transform else original_image

        original_image_np = original_image.permute(1, 2, 0).numpy()
        augmented_image_np = augmented_image.permute(1, 2, 0).numpy()

        if original_image_np.shape[2] == 1:
            original_image_np = original_image_np.squeeze()
            augmented_image_np = augmented_image_np.squeeze()

        axes[0, i].imshow(original_image_np, cmap='gray' if original_image_np.ndim == 2 else None)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Original {i}")

        axes[1, i].imshow(augmented_image_np, cmap='gray' if augmented_image_np.ndim == 2 else None)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Augmented {i}")

    plt.suptitle(f'Augmentation: {augmentation_method} for {dataset_name}')
    plt.show()


def visualize_distribution(train_loader, augmentation_method, num_samples, dataset_name, two_features=False, model=None):
    if two_features and model:

        print(f"Visualizing distribution for {dataset_name} with {augmentation_method}, {num_samples} samples.")

        features, labels = [], []

        with torch.no_grad():
            for inputs, target in train_loader:
                output = model.forward_conv(inputs)
                features.extend(output.numpy())
                labels.extend(target.numpy())

        features, labels = np.array(features), np.array(labels)
        sample_size = min(num_samples, len(features)) if num_samples is not None else len(features)
        sample_indices = np.random.choice(len(features), sample_size, replace=False)
        features_sampled, labels_sampled = features[sample_indices], labels[sample_indices]

        if features_sampled.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_sampled)
        else:
            features_2d = features_sampled

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_sampled, cmap='tab10', alpha=0.6)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Classes')
        plt.title(f'Scatter Plot {dataset_name} Two Features, (Augmentation: {augmentation_method}, {num_samples} samples)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()


def main():
    augmentations = [
        {"name": "rotation", "transform": "rotation"},
        {"name": "color_jitter", "transform": "color_jitter"}
    ]

    sample_sizes = [100, 200, 1000, None]

    architectures = [
        {"name": "Cifar10CNN2", "model_class": Cifar10CNN2, "dataset": DatasetName.CIFAR10, "two_features": True},
        {"name": "MNISTCNN2", "model_class": MNISTCNN2, "dataset": DatasetName.MNIST, "two_features": True},
        {"name": "Cifar10CNN", "model_class": Cifar10CNN, "dataset": DatasetName.CIFAR10, "two_features": False},
        {"name": "MNISTCNN", "model_class": MNISTCNN, "dataset": DatasetName.MNIST, "two_features": False},
    ]

    results = []

    for arch in architectures:
        for aug in augmentations:
            for num_samples in sample_sizes:

                print(f"Processing Architecture: {arch['name']}, Augmentation: {aug['name']}, Num Samples: {num_samples}")

                train_loader, test_loader, train_loader_original, model_path = prepare_dataset(
                    dataset=arch["dataset"],
                    augment=aug["transform"],
                    num_samples=num_samples,
                    two_features=arch["two_features"]
                )

                model = arch["model_class"]()

                mean_accuracy, std_accuracy = evaluate_model_experiment_two(
                    model,
                    train_loader,
                    test_loader,
                    model_path
                )

                results.append({
                    "Architecture": arch["name"],
                    "Augmentation": aug["name"],
                    "Num Samples": num_samples if num_samples else "All",
                    "Mean Test Accuracy": mean_accuracy,
                    "Std Test Accuracy": std_accuracy
                })

                print(f"Architecture: {arch['name']}, Augmentation: {aug['name']}, Num Samples: {num_samples}, Mean Test Accuracy: {mean_accuracy}, Std Test Accuracy: {std_accuracy}")

                if aug["transform"]:
                    visualize_augmentations(train_loader_original, aug["name"], arch["dataset"])

                visualize_distribution(train_loader, aug["name"], num_samples, arch["dataset"], arch["two_features"], model)

    df_results = pd.DataFrame(results)
    df_results.to_csv("experiment_two_results.csv", index=False)

    print("Results saved to experiment_two_results.csv")
    print(df_results)


if __name__ == "__main__":
    main()
