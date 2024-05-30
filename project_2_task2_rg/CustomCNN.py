from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from torch import optim, nn
from torchvision.transforms import transforms
import seaborn as sns
import statistics

from project_2_task2_rg.models import CIFAR10CNN, CIFAR10CNN2

subsets = [100, 200, 1000]


def create_subset(dataset, n):
    indices = np.random.choice(len(dataset), n, replace=False)
    subset = Subset(dataset, indices)
    return subset


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_accuracy = []
    test_accuracy = []
    best_accuracy = 0.0
    best_model = None

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"Beginning of epoch {epoch + 1}")

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
        epoch_train_loss = running_loss / len(train_loader)

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

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_train_loss}, '
              f'Train Accuracy: {epoch_train_accuracy}, Test Accuracy: {epoch_test_accuracy}')

    print(f'Best Test Accuracy: {best_accuracy}')
    return train_accuracy, test_accuracy, best_model


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total} %')


def plot_accuracies(train_accuracy, test_accuracy, dataset_name, extraction_method):
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Accuracy scores - {dataset_name} {extraction_method}')
    plt.show()


def plot_confusion_matrix(model, test_loader, dataset_name, extraction_method):
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


def plot_decision_boundary(model, test_loader, dataset_name, extraction_method, title='Decision Boundary'):
    model.eval()
    features = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            extracted_features = model.extract_features(inputs)
            features.extend(extracted_features.numpy())
            true_labels.extend(labels.numpy())

    features = np.array(features)
    true_labels = np.array(true_labels)

    plot_margin = 0.1
    grid_step_size = 0.05

    x_min = features[:, 0].min() - plot_margin
    x_max = features[:, 0].max() + plot_margin
    y_min = features[:, 1].min() - plot_margin
    y_max = features[:, 1].max() + plot_margin

    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step_size), np.arange(y_min, y_max, grid_step_size))
    grid = np.c_[xx.ravel(), yy.ravel()]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features, true_labels)
    Z = knn.predict(grid)
    Z = Z.reshape(xx.shape)

    group_colors = sns.color_palette("hsv", len(np.unique(true_labels)))
    colormap = LinearSegmentedColormap.from_list('colormap', group_colors, N=256)

    plt.contourf(xx, yy, Z, cmap=colormap, alpha=0.4)
    plt.contour(xx, yy, Z, colors='k', linewidths=1)

    if true_labels is not None:
        true_label_colors = [group_colors[label] for label in true_labels]
        plt.scatter(features[:, 0], features[:, 1], c=true_label_colors, edgecolors='k')

    plt.title(f'{title} - {dataset_name} - {extraction_method}')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def show_cifar10_samples(test_dataset_cifar, title):
    # Dictionary to store 10 images for each class
    class_images = {i: [] for i in range(10)}

    # Counter to keep track of the number of images for each class
    class_counter = {i: 0 for i in range(10)}

    # Iterate through the test dataset to select 10 images for each class
    for img, label in test_dataset_cifar:
        if class_counter[label] < 10:
            class_images[label].append(img)
            class_counter[label] += 1

        # Check if 10 images have been collected for each class
        if all(count == 10 for count in class_counter.values()):
            break

    # Display the selected images
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(np.transpose(class_images[i][j], (1, 2, 0)))
            axes[i, j].axis('off')
    plt.suptitle(title)  # Set the title for the entire figure
    plt.show()


def first_exp():
    cifar_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    mnist_train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    mnist_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    criterion = nn.CrossEntropyLoss()

    train_dataset_cifar = CIFAR10(root='./data', train=True, download=False, transform=cifar_train_transform)
    test_dataset_cifar = CIFAR10(root='./data', train=False, download=False, transform=cifar_test_transform)

    train_loader = DataLoader(train_dataset_cifar, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset_cifar, batch_size=64, shuffle=False, num_workers=2)

    model = CIFAR10CNN2()

    """
    train_dataset_mnist = MNIST(root='./data', train=True, download=False, transform=mnist_train_transform)
    test_dataset_mnist = MNIST(root='./data', train=False, download=False, transform=mnist_test_transform)

    train_loader = DataLoader(train_dataset_mnist, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset_mnist, batch_size=64, shuffle=False, num_workers=2)

    model = MNISTCNN2()
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    train_accuracy, test_accuracy, best_model = train(model, criterion, optimizer, train_loader,
                                                      test_loader, epochs=10)
    """
    print("Testing...")
    test(model, test_loader_cifar)
    """

    print("Plotting accuracies...")
    plot_accuracies(train_accuracy, test_accuracy, "CIFAR10", "Two features")

    print("Plotting confusion matrix...")
    plot_confusion_matrix(model, test_loader, "CIFAR10", "Two features")

    print("Plotting decision boundary...")
    plot_decision_boundary(model, test_loader, "CIFAR10", "Two features")


def second_exp():
    cifar_train_transform_mine = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_train_transform_alternative = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    criterion = nn.CrossEntropyLoss()
    models = [CIFAR10CNN2, CIFAR10CNN]
    subsets = [100, 200, 1000, 10000]
    augmentations = [cifar_test_transform, cifar_train_transform_mine, cifar_train_transform_alternative]
    LOOP_ITER = 5


    test_dataset_cifar = CIFAR10(root='./data', train=False, download=False, transform=cifar_test_transform)
    test_loader = DataLoader(test_dataset_cifar, batch_size=64, shuffle=False, num_workers=2)

    for augment in augmentations:
        """
        train_dataset_cifar = CIFAR10(root='./data', train=True, download=False, transform=cifar_test_transform)
        show_cifar10_samples(train_dataset_cifar, "CIFAR10 training dataset for no augmentation")

        train_dataset_cifar = CIFAR10(root='./data', train=True, download=False, transform=cifar_train_transform_mine)
        show_cifar10_samples(train_dataset_cifar, "CIFAR10 training dataset for 1st person augmentation")

        train_dataset_cifar = CIFAR10(root='./data', train=True, download=False, transform=cifar_train_transform_alternative)
        show_cifar10_samples(train_dataset_cifar, "CIFAR10 training dataset for 2st person augmentation")
        """
        train_dataset_cifar = CIFAR10(root='./data', train=True, download=False,
                                      transform=augment)

        for subset in subsets:
            #if(subset == None):
                #train_loader = DataLoader(train_dataset_cifar, batch_size=64, shuffle=True, num_workers=2)
            #else:
            train_loader = DataLoader(create_subset(train_dataset_cifar, subset), batch_size=64, shuffle=True, num_workers=2)

            for model_cls  in models:

                accuracy = []

                for i in range(LOOP_ITER):
                    current_model = model_cls()
                    optimizer = optim.Adam(current_model.parameters(), lr=0.001)

                    print("Training...")
                    train_accuracy, test_accuracy, best_model = train(current_model, criterion, optimizer, train_loader,
                                                                      test_loader, epochs=5)
                    accuracy.append(test_accuracy[-1])

                    #if(model==CIFAR10CNN2()):
                        #print("Plotting decision boundary...")
                        #plot_decision_boundary(model, test_loader, "CIFAR10CNN2", "Two features")

                # Calculate the average value (mean)
                average_accuracy = statistics.mean(accuracy)

                # Calculate the standard deviation
                standard_deviation = statistics.stdev(accuracy)

                model_name = model_cls.__name__
                augmentation_name = augment.transforms[0].__class__.__name__ if augment == cifar_test_transform else (
                    "1st user augmentation" if augment == cifar_train_transform_mine else "2nd user augmentation")

                print(f"Average Accuracy for {model_name}, with {augmentation_name}, for {subset} images:",
                      average_accuracy)
                print(f"Standard Deviation for {model_name}, with {augmentation_name}, for {subset} images:",
                      standard_deviation)


if __name__ == "__main__":

    #first_exp()
    second_exp()
