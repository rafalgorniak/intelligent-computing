import os
import torch
import torch.optim as optim
from torchvision.transforms import transforms
from torch import nn
from sklearn.metrics import classification_report, accuracy_score
from torchvision.datasets import CIFAR10

from Datasets import (load_and_prepare_iris_data, load_and_prepare_wine_data, load_and_prepare_breast_cancer_data,
                      load_and_prepare_mnist_data, load_and_prepare_cifar10_data)
from CustomMLP import CustomMLP
from CustomCNN import CIFAR10CNN, CIFAR10CNN2, MNISTCNN, MNISTCNN2


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
        #("Breast Cancer Dataset", load_and_prepare_breast_cancer_data, 30, 2, 'breast_cancer_mlp', False),
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