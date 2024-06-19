import os
import torch
import torch.optim as optim
import numpy as np
from torchvision.transforms import transforms
from torch import nn
from sklearn.metrics import classification_report, accuracy_score

from captum.attr import Lime, Saliency, IntegratedGradients, FeatureAblation
import matplotlib.pyplot as plt
from captum.attr import visualization as viz

from Datasets import (load_and_prepare_iris_data, load_and_prepare_wine_data, load_and_prepare_breast_cancer_data,
                      load_and_prepare_mnist_data, load_and_prepare_cifar10_data)
from CustomMLP import CustomMLP
from CustomCNN import CIFAR10CNN, CIFAR10CNN2, MNISTCNN, MNISTCNN2


def attributionExplain(model, data_loader, method_name="lime", is_cnn=False, num_samples=50, feature_names=None):
    model.eval()

    if method_name == "lime":
        method = Lime(model)
    elif method_name == "saliency":
        method = Saliency(model)
    elif method_name == "integrated_gradients":
        method = IntegratedGradients(model)
    elif method_name == "feature_ablation":
        method = FeatureAblation(model)
    else:
        raise ValueError("Unknown explanation method")

    # Get a batch of data
    for data, target in data_loader:
        if not is_cnn:  # Flatten data if not using CNN
            data = data.view(data.size(0), -1)

        # Use the first sample from the batch
        input_sample = data[0].unsqueeze(0)
        target_sample = target[0].unsqueeze(0)

        print(input_sample)

        # Get prediction for the sample
        output = model(input_sample)
        _, predicted = torch.max(output.data, 1)

        # Generate explanation
        if method_name in ["lime", "feature_ablation"]:
            attr = method.attribute(input_sample, target=predicted, n_samples=num_samples)
        else:
            attr = method.attribute(input_sample, target=predicted)

        # Convert the attribution to numpy
        attr = attr.cpu().detach().numpy().squeeze()  # Remove the batch dimension
        input_sample_np = input_sample.cpu().detach().numpy().squeeze()  # Remove the batch dimension

        # Debug prints
        print("attr shape:", attr.shape)
        print("input_sample shape:", input_sample_np.shape)

        # Visualize the explanation
        if is_cnn or input_sample_np.ndim == 3:  # Check if the input sample is image-like
            if input_sample_np.ndim == 2:  # Handle 2D grayscale images like MNIST
                plt.imshow(input_sample_np, cmap='gray')
                plt.title("Current Image")
                plt.axis('off')
                plt.show()

                # Expand dimensions of attr for visualization
                attr = np.expand_dims(attr, axis=-1)
                viz.visualize_image_attr(attr, input_sample_np, method="heat_map", sign="all", show_colorbar=True,
                                         title=f"{method_name.capitalize()} Explanation")
            else:
                if input_sample_np.shape[0] == 3:
                    input_sample_np = np.transpose(input_sample_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
                if attr.ndim == 3:  # Check if the attr has 3 dimensions (for CNN)
                    attr = np.transpose(attr, (1, 2, 0))  # Move channels to the last dimension
                    viz.visualize_image_attr(attr, input_sample_np, method="heat_map", sign="all", show_colorbar=True,
                                             title=f"{method_name.capitalize()} Explanation")
                else:  # For MLP or non-3D attr
                    plt.imshow(input_sample_np)
                    plt.title("Current Image")
                    plt.axis('off')
                    plt.show()

        else:  # For tabular data, we use a bar plot
            attr = attr.reshape(-1)
            input_sample_np = input_sample_np.reshape(-1)
            plt.figure(figsize=(10, 6))
            if feature_names is not None:
                plt.barh(feature_names, attr)
                plt.xlabel("Feature Importance")
                plt.ylabel("Features")
                plt.title(f"{method_name.capitalize()} Explanation")
            else:
                plt.barh(range(len(attr)), attr)
                plt.xlabel("Feature Importance")
                plt.ylabel("Feature Index")
                plt.title(f"{method_name.capitalize()} Explanation")
            plt.show()

        break  # Explain only the first sample for now



#def counterfactsExplain


def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001, is_cnn=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for data, target in train_loader:
            if not is_cnn:  # Flatten data if not using CNN
                data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Calculate training accuracy
        train_accuracy = correct / total

        # Evaluate on test set
        test_accuracy, _ = evaluate_model(model, test_loader, is_cnn)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, '
              f'Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%')

    print('Training finished.')


def evaluate_model(model, test_loader, is_cnn=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            if not is_cnn:  # Flatten data if not using CNN
                data = data.view(data.size(0), -1)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=1)
    return accuracy, report

def save_model(model, model_name):
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/{model_name}.pth')


def load_model(model, model_name):
    model.load_state_dict(torch.load(f'models/{model_name}.pth'))
    model.eval()
    return model


if __name__ == "__main__":

    augmentation_first = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    augmentation_second = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    augmentation_none = transforms.Compose([
        transforms.ToTensor(),
    ])

    datasets = [
        #("Iris Dataset", load_and_prepare_iris_data, 4, 3, 'iris_mlp', False),
        #("Wine Dataset", load_and_prepare_wine_data, 13, 3, 'wine_mlp', False),
        ("Breast Cancer Dataset", load_and_prepare_breast_cancer_data, 30, 2, 'breast_cancer_mlp', False),
        #("MNIST Dataset (Org. MLP)", lambda: load_and_prepare_mnist_data(method='none'), 28*28, 10, 'mnist_mlp', False),
        #("MNIST Dataset (Org. CNN)", lambda: load_and_prepare_mnist_data(method='none'), 28*28, 10, 'mnist_cnn', True),
        #("MNIST Dataset (Org. CNN2)", lambda: load_and_prepare_mnist_data(method='none'), 2, 10, 'mnist_cnn2', True),
        #("MNIST Dataset (LDA)", lambda: load_and_prepare_mnist_data(method='lda'), 2, 10, 'mnist_lda_mlp', False),
        #("MNIST Dataset (HOG)", lambda: load_and_prepare_mnist_data(method='hog'), 144, 10, 'mnist_hog_mlp', False),
        #("MNIST Dataset (PCA)", lambda: load_and_prepare_mnist_data(method='pca'), 2, 10, 'mnist_pca_mlp', False),
        #("CIFAR-10 Dataset CNN", load_and_prepare_cifar10_data, 32*32, 10, 'cifar10_cnn', True)
        #("CIFAR-10 Dataset CNN2", load_and_prepare_cifar10_data, 2, 10, 'cifar10_cnn2', True)
    ]

    for dataset_name, data_loader, input_size, output_size, model_name, is_cnn in datasets:
        print(f"\n{dataset_name}:")
        train_loader, test_loader = data_loader()

        if is_cnn:
            if dataset_name == "MNIST Dataset (Org. CNN)":
                model = MNISTCNN()
            elif dataset_name == "MNIST Dataset (Org. CNN2)":
                model = MNISTCNN2()
            elif dataset_name == "CIFAR-10 Dataset CNN":
                model = CIFAR10CNN()
            elif dataset_name == "CIFAR-10 Dataset CNN2":
                model = CIFAR10CNN2()
        else:
            model = CustomMLP(input_size=input_size, hidden_size=128, output_size=output_size)

        model_path = f'models/{model_name}.pth'
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model = load_model(model, model_name)
        else:
            print("Training new model")
            train_model(model, train_loader, test_loader, epochs=10, is_cnn=is_cnn)
            save_model(model, model_name)

        accuracy, report = evaluate_model(model, test_loader, is_cnn=is_cnn)
        print(report)
        print(f'Accuracy: {str(accuracy * 100)[:5]}%')

        attributionExplain(model, test_loader, method_name="lime", is_cnn=is_cnn)
        attributionExplain(model, test_loader, method_name="saliency", is_cnn=is_cnn)
        attributionExplain(model, test_loader, method_name="integrated_gradients", is_cnn=is_cnn)
        attributionExplain(model, test_loader, method_name="feature_ablation", is_cnn=is_cnn)