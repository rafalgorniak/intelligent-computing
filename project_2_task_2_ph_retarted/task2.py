import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_lightning import Trainer

from models import MNISTCNN, MNISTCNN2, CIFAR10CNN, CIFAR10CNN2
from custom_dataset import CustomDataset, DatasetName
from LitModel import LitModel
from plots import visualize_augmentations, plot_feature_distribution, plot_confusion_matrix
from file_helper import save_model, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_augmentation_methods():
    augmentations = {
        "no_augmentation": transforms.Compose([
            transforms.ToTensor(),
        ]),
        "augmentation_1": transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        "augmentation_2": transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
    }
    return augmentations


def get_subset(dataset, num_samples_per_class):
    original_dataset = dataset.dataset
    if isinstance(original_dataset, Subset):
        original_dataset = original_dataset.dataset
    targets = np.array(original_dataset.targets)
    indices = np.hstack([np.where(targets == i)[0][:num_samples_per_class] for i in range(len(np.unique(targets)))])
    subset = Subset(original_dataset, indices)
    return subset


def train_and_evaluate_model(model_class, dataset_name, num_samples, augmentation_name, device):
    model_path = f'data/{dataset_name}_{model_class.__name__}_{num_samples}_{augmentation_name}.pth'

    if os.path.isfile(model_path):
        print(f"Model file {model_path} already exists. Skipping training.")
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model, optimizer, _ = load_model(model, optimizer, model_path)
    else:
        train_dataset = CustomDataset(dataset_name, train=True,
                                      use_augmentation=(augmentation_name != "no_augmentation"))
        test_dataset = CustomDataset(dataset_name, train=False, use_augmentation=False)

        if num_samples:
            train_dataset = get_subset(train_dataset, num_samples)

        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        lit_model = LitModel(model)

        trainer = Trainer(max_epochs=10, devices=1 if torch.cuda.is_available() else 1,
                          accelerator="gpu" if torch.cuda.is_available() else "cpu")

        trainer.fit(lit_model, train_dataloaders=trainloader, val_dataloaders=testloader)

        save_model(model, optimizer, trainer.current_epoch, model_path)

    return model


def evaluate_augmentation_impact(model_class, dataset_name, num_samples, augmentations):
    results = {}

    for augmentation_name, augmentation in augmentations.items():
        print(
            f"Evaluating augmentation {augmentation_name} with {num_samples if num_samples else 'all'} samples for {dataset_name}")

        accuracies = []

        for run in range(10):
            model = train_and_evaluate_model(model_class, dataset_name, num_samples, augmentation_name, device)

            test_dataset = CustomDataset(dataset_name, train=False, use_augmentation=False)
            testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

            lit_model = LitModel(model)
            trainer = Trainer(devices=1 if torch.cuda.is_available() else 1,
                              accelerator="gpu" if torch.cuda.is_available() else "cpu")
            val_metrics = trainer.validate(lit_model, dataloaders=testloader)

            val_accuracy = val_metrics[0].get('val_acc', None)
            if val_accuracy is not None:
                val_accuracy = val_accuracy.cpu().numpy() if hasattr(val_accuracy, 'cpu') else val_accuracy
            accuracies.append(val_accuracy)

        results[augmentation_name] = accuracies

    return results


def evaluate_training_data_impact(model_class, dataset_name, augmentations, num_samples_list):
    results = {}

    for num_samples in num_samples_list:
        print(f"Evaluating training data size with {num_samples if num_samples else 'all'} samples for {dataset_name}")

        for augmentation_name, augmentation in augmentations.items():
            accuracies = []

            for run in range(10):
                model = train_and_evaluate_model(model_class, dataset_name, num_samples, augmentation_name, device)

                test_dataset = CustomDataset(dataset_name, train=False, use_augmentation=False)
                testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

                lit_model = LitModel(model)
                trainer = Trainer(devices=1 if torch.cuda.is_available() else 1,
                                  accelerator="gpu" if torch.cuda.is_available() else "cpu")
                val_metrics = trainer.validate(lit_model, dataloaders=testloader)

                val_accuracy = val_metrics[0].get('val_acc', None)
                if val_accuracy is not None:
                    val_accuracy = val_accuracy.cpu().numpy() if hasattr(val_accuracy, 'cpu') else val_accuracy
                accuracies.append(val_accuracy)

            results[(num_samples, augmentation_name)] = accuracies

    return results


def find_best_model(model_classes, dataset_names, num_samples_list, augmentations):
    best_model = None
    best_accuracy = 0

    for model_class in model_classes:
        for dataset_name in dataset_names:
            if dataset_name == DatasetName.MNIST and model_class not in [MNISTCNN, MNISTCNN2]:
                continue
            if dataset_name == DatasetName.CIFAR10 and model_class not in [CIFAR10CNN, CIFAR10CNN2]:
                continue

            for num_samples in num_samples_list:
                for augmentation_name, augmentation in augmentations.items():
                    accuracies = []

                    for run in range(10):
                        model = train_and_evaluate_model(model_class, dataset_name, num_samples, augmentation_name,
                                                         device)

                        test_dataset = CustomDataset(dataset_name, train=False, use_augmentation=False)
                        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

                        lit_model = LitModel(model)
                        trainer = Trainer(devices=1 if torch.cuda.is_available() else 1,
                                          accelerator="gpu" if torch.cuda.is_available() else "cpu")
                        val_metrics = trainer.validate(lit_model, dataloaders=testloader)

                        val_accuracy = val_metrics[0].get('val_acc', None)
                        if val_accuracy is not None:
                            val_accuracy = val_accuracy.cpu().numpy() if hasattr(val_accuracy, 'cpu') else val_accuracy
                        accuracies.append(val_accuracy)

                    mean_accuracy = np.mean(accuracies)
                    std_accuracy = np.std(accuracies)
                    print(
                        f"Model: {model_class.__name__}, Dataset: {dataset_name}, Augmentation: {augmentation_name}, Num Samples: {num_samples if num_samples else 'all'}, Mean Accuracy: {mean_accuracy:.4f}, Std Accuracy: {std_accuracy:.4f}")

                    if mean_accuracy > best_accuracy:
                        best_accuracy = mean_accuracy
                        best_model = (model_class, dataset_name, augmentation_name, num_samples)

    return best_model, best_accuracy


def visualize_best_model(best_model, device, augmentations):
    model_class, dataset_name, augmentation_name, num_samples = best_model
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_path = f'data/{dataset_name}_{model_class.__name__}_{num_samples}_{augmentation_name}.pth'
    model, optimizer, _ = load_model(model, optimizer, model_path)

    test_dataset = CustomDataset(dataset_name, train=False, use_augmentation=False)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    true_labels = []
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for batch in testloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    classes = list(range(10))
    plot_confusion_matrix(true_labels, pred_labels, classes,
                          f'Confusion Matrix for {best_model[0].__name__} on {best_model[1]}')

    train_dataset = CustomDataset(dataset_name, train=True, use_augmentation=(augmentation_name != "no_augmentation"))
    if num_samples:
        train_dataset = get_subset(train_dataset, num_samples)
    visualize_augmentations(train_dataset, augmentations[augmentation_name])

    if model_class == MNISTCNN2 or model_class == CIFAR10CNN2:
        plot_feature_distribution(train_dataset, model)


def main():
    num_samples_list = [None, 100, 200, 1000]
    augmentations = get_augmentation_methods()

    model_classes = [MNISTCNN, MNISTCNN2, CIFAR10CNN, CIFAR10CNN2]
    dataset_names = [DatasetName.MNIST, DatasetName.CIFAR10]

    for model_class in model_classes:
        for dataset_name in dataset_names:
            if dataset_name == DatasetName.MNIST and model_class not in [MNISTCNN, MNISTCNN2]:
                continue
            if dataset_name == DatasetName.CIFAR10 and model_class not in [CIFAR10CNN, CIFAR10CNN2]:
                continue

            print(f"\nEvaluating model {model_class.__name__} on dataset {dataset_name} with various augmentations")
            augmentation_results = evaluate_augmentation_impact(model_class, dataset_name, None, augmentations)
            for augmentation_name, accuracies in augmentation_results.items():
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                print(
                    f"Augmentation: {augmentation_name}, Mean Accuracy: {mean_accuracy:.4f}, Std Accuracy: {std_accuracy:.4f}")

    for model_class in model_classes:
        for dataset_name in dataset_names:
            if dataset_name == DatasetName.MNIST and model_class not in [MNISTCNN, MNISTCNN2]:
                continue
            if dataset_name == DatasetName.CIFAR10 and model_class not in [CIFAR10CNN, CIFAR10CNN2]:
                continue

            print(
                f"\nEvaluating model {model_class.__name__} on dataset {dataset_name} with various training data sizes")
            training_data_results = evaluate_training_data_impact(model_class, dataset_name, augmentations,
                                                                  num_samples_list)
            for (num_samples, augmentation_name), accuracies in training_data_results.items():
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                print(
                    f"Num Samples: {num_samples if num_samples else 'all'}, Augmentation: {augmentation_name}, Mean Accuracy: {mean_accuracy:.4f}, Std Accuracy: {std_accuracy:.4f}")

    print("\nFinding the best model configuration")
    best_model, best_accuracy = find_best_model(model_classes, dataset_names, num_samples_list, augmentations)
    print(f"Best Model: {best_model}, Best Accuracy: {best_accuracy:.4f}")

    visualize_best_model(best_model, device, augmentations)


if __name__ == "__main__":
    main()
