import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from skimage.feature import hog
import numpy as np
import os
import torch.optim as optim
from torchvision.transforms import transforms
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from captum.attr import Lime, Saliency, IntegratedGradients, FeatureAblation
import matplotlib

from CustomCNN import CIFAR10CNN, MNISTCNN, MNISTCNN2
from CustomMLP import CustomMLP

matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
from captum.attr import visualization as viz

from Datasets import (load_and_prepare_iris_data, load_and_prepare_wine_data, load_and_prepare_breast_cancer_data,
                      load_and_prepare_mnist_data, load_and_prepare_cifar10_data)

def apply_lda(dataset):
    X = dataset.data.numpy().reshape(len(dataset), -1)
    y = dataset.targets.numpy()
    lda = LDA(n_components=2)
    X_transformed = lda.fit_transform(X, y)
    return TensorDataset(torch.tensor(X_transformed, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

def apply_hog(dataset):
    X = dataset.data.numpy()
    hog_features = []
    for img in X:
        img = img.reshape((28, 28))
        hog_feat = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(hog_feat)
    X_transformed = np.array(hog_features)
    y = dataset.targets.numpy()
    return TensorDataset(torch.tensor(X_transformed, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

def apply_pca(dataset):
    X = dataset.data.numpy().reshape(len(dataset), -1)
    y = dataset.targets.numpy()
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X, y)
    return TensorDataset(torch.tensor(X_transformed, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

def attributionExplain_combined(model, data_loader, methods, dataset_name, is_cnn=False, num_samples=50, max_plots=2):
    model.eval()
    method_dict = {
        "lime": Lime(model),
        "saliency": Saliency(model),
        "integrated_gradients": IntegratedGradients(model),
        "feature_ablation": FeatureAblation(model)
    }

    samples = {}
    for data, target in data_loader:
        if not is_cnn:
            data = data.view(data.size(0), -1)
        data.requires_grad = True  # Ensure gradients are enabled for the data

        for i in range(len(target)):
            label = target[i].item()
            if label not in samples:
                samples[label] = (data[i].unsqueeze(0), target[i].unsqueeze(0))
            if len(samples) == len(set(target.cpu().numpy())):
                break
        if len(samples) == len(set(target.cpu().numpy())):
            break

    fig, axes = plt.subplots(max_plots, len(methods), figsize=(15, 10))

    for i, (class_idx, (input_sample, target_sample)) in enumerate(list(samples.items())[:max_plots]):
        output = model(input_sample)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(probabilities.data, 1)

        for j, method_name in enumerate(methods):
            method = method_dict[method_name]

            if method_name in ["lime", "feature_ablation"]:
                attr = method.attribute(input_sample, target=predicted, n_samples=num_samples)
            else:
                attr = method.attribute(input_sample, target=predicted)

            attr = attr.cpu().detach().numpy().squeeze()
            input_sample_np = input_sample.cpu().detach().numpy().squeeze()

            if 'MNIST' in dataset_name:  # Image datasets
                if 'LDA' in dataset_name or 'PCA' in dataset_name or 'HOG' in dataset_name:
                    attr = attr.reshape(-1)
                    input_sample_np = input_sample_np.reshape(-1)
                    axes[i, j].barh(range(len(attr)), attr)
                    axes[i, j].set_title(f'Pred: {predicted.item()}, Orig: {target_sample.item()}')
                else:
                    input_sample_np = input_sample_np.reshape(28, 28)
                    attr = attr.reshape(28, 28)

                    # Expand dimensions if necessary for visualization
                    if attr.ndim == 2:
                        attr = np.expand_dims(attr, axis=-1)
                    if input_sample_np.ndim == 2:
                        input_sample_np = np.expand_dims(input_sample_np, axis=-1)

                    axes[i, j].imshow(input_sample_np, cmap='gray')
                    viz.visualize_image_attr(attr, input_sample_np, method="blended_heat_map", sign="all", show_colorbar=True, plt_fig_axis=(fig, axes[i, j]))
                    axes[i, j].set_title(f'Pred: {predicted.item()}, Orig: {target_sample.item()}')

            elif 'CIFAR-10' in dataset_name:
                # Normalize and clip the input sample to valid range for imshow
                input_sample_np = (input_sample_np - input_sample_np.min()) / (input_sample_np.max() - input_sample_np.min())
                input_sample_np = np.clip(input_sample_np, 0, 1)

                if attr.ndim == 3 and attr.shape[0] == 3:  # Ensure the attribute map has 3 channels
                    attr = np.transpose(attr, (1, 2, 0))
                if input_sample_np.ndim == 3 and input_sample_np.shape[0] == 3:
                    input_sample_np = np.transpose(input_sample_np, (1, 2, 0))

                axes[i, j].imshow(input_sample_np)
                viz.visualize_image_attr(attr, input_sample_np, method="blended_heat_map", sign="all", show_colorbar=True, plt_fig_axis=(fig, axes[i, j]))
                axes[i, j].set_title(f'Pred: {predicted.item()}, Orig: {target_sample.item()}')

            elif dataset_name in ["Iris Dataset", "Wine Dataset", "Breast Cancer Dataset"]:  # Non-image datasets
                attr = attr.reshape(-1)
                input_sample_np = input_sample_np.reshape(-1)
                axes[i, j].barh(range(len(attr)), attr)
                axes[i, j].set_title(f'Pred: {predicted.item()}, Orig: {target_sample.item()}')

            else:
                attr = attr.reshape(-1)
                input_sample_np = input_sample_np.reshape(-1)
                axes[i, j].barh(range(len(attr)), attr)
                axes[i, j].set_title(f'Pred: {predicted.item()}, Orig: {target_sample.item()}')

    plt.suptitle(f"xAI for {dataset_name} using Methods: {', '.join(methods)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"xai_{dataset_name.replace(' ', '_')}.png")
    plt.close()

def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001, is_cnn=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            if not is_cnn:
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

        train_accuracy = correct / total

        test_accuracy, _ = evaluate_model(model, test_loader, is_cnn)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%')

    print('Training finished.')

def evaluate_model(model, test_loader, is_cnn=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            if not is_cnn:
                data = data.view(data.size(0), -1)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities.data, 1)
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
    # Define the transforms for data augmentation
    augmentation_none = transforms.Compose([
        transforms.ToTensor(),
    ])

    datasets = [
        ("Iris Dataset", load_and_prepare_iris_data, 4, 3, 'iris_mlp', False),
        ("Wine Dataset", load_and_prepare_wine_data, 13, 3, 'wine_mlp', False),
        ("Breast Cancer Dataset", load_and_prepare_breast_cancer_data, 30, 2, 'breast_cancer_mlp', False),
        ("MNIST Dataset (Original MLP)", lambda: load_and_prepare_mnist_data(method='none'), 28*28, 10, 'mnist_mlp', False),
        ("MNIST Dataset (Original CNN)", lambda: load_and_prepare_mnist_data(method='none'), 28*28, 10, 'mnist_cnn', True),
        ("MNIST Dataset (Original CNN2)", lambda: load_and_prepare_mnist_data(method='none'), 28*28, 10, 'mnist_cnn2', True),
        ("MNIST Dataset (LDA)", lambda: apply_lda(load_and_prepare_mnist_data(method='none')[0].dataset), 2, 10, 'mnist_lda_mlp', False),
        ("MNIST Dataset (HOG)", lambda: apply_hog(load_and_prepare_mnist_data(method='none')[0].dataset), 144, 10, 'mnist_hog_mlp', False),
        ("MNIST Dataset (PCA)", lambda: apply_pca(load_and_prepare_mnist_data(method='none')[0].dataset), 2, 10, 'mnist_pca_mlp', False),
        ("CIFAR-10 Dataset CNN", lambda: load_and_prepare_cifar10_data(), 32*32*3, 10, 'cifar10_cnn', True),
        ("CIFAR-10 Dataset CNN2", lambda: load_and_prepare_cifar10_data(), 32*32*3, 10, 'cifar10_cnn2', True)
    ]

    for dataset_name, data_loader, input_size, output_size, model_name, is_cnn in datasets:
        print(f"\n{dataset_name}:")
        if callable(data_loader):
            if "MNIST" in dataset_name and ("LDA" in dataset_name or "HOG" in dataset_name or "PCA" in dataset_name):
                transformed_dataset = data_loader()
                train_loader = DataLoader(transformed_dataset, batch_size=64, shuffle=True)
                test_loader = DataLoader(transformed_dataset, batch_size=64, shuffle=False)
            else:
                train_loader, test_loader = data_loader()
        else:
            train_loader, test_loader = data_loader

        if is_cnn:
            if "MNIST" in dataset_name:
                model = MNISTCNN() if dataset_name == "MNIST Dataset (Original CNN)" else MNISTCNN2()
            else:
                model = CIFAR10CNN()  # Ensure CIFAR-10 uses a model with 3 input channels
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

        # Visualization for image datasets and non-image datasets
        methods = ["lime", "saliency", "integrated_gradients", "feature_ablation"]
        attributionExplain_combined(model, test_loader, methods, dataset_name, is_cnn=is_cnn, max_plots=2)
