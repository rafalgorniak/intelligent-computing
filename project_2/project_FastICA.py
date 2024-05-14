import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

num_epochs = 5
hidden_size = 128
output_size = 10

transform = transforms.Compose([
    transforms.ToTensor()
])

criterion = nn.CrossEntropyLoss()


def getMnistTrain():
    mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_images = mnist_data.data.numpy().reshape(-1, 28 * 28)
    ica = FastICA(n_components=2, random_state=42)
    ica_representation = ica.fit_transform(train_images)
    return torch.tensor(ica_representation.astype('float32')), mnist_data.targets


def getMnistTest():
    mnist_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_images = mnist_data.data.numpy().reshape(-1, 28 * 28)
    ica = FastICA(n_components=2, random_state=42)
    ica_representation = ica.fit_transform(test_images)
    return torch.tensor(ica_representation.astype('float32')), mnist_data.targets


class customMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(customMLP, self).__init__()
        self.function1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.function2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.function1(x)
        x = self.relu(x)
        x = self.function2(x)
        return x


def train(model, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, running_loss / len(train_loader)))


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy on the test set: %.2f %%' % accuracy)


if __name__ == "__main__":
    train_data, train_targets = getMnistTrain()
    test_data, test_targets = getMnistTest()

    # Train data loader
    train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_data, train_targets),
                                               batch_size=64, shuffle=True)

    # Test data loader
    test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(test_data, test_targets),
                                              batch_size=64, shuffle=False)

    # Visualization of FastICA transformation for training data
    plt.figure(figsize=(8, 6))
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_targets, cmap='viridis', s=10)
    plt.colorbar(label='Digit')
    plt.title('FastICA Visualization of MNIST Training Dataset')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    # Train your model with FastICA transformed data
    model = customMLP(2, hidden_size, output_size)  # Input size changed to reflect FastICA components
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train(model, optimizer, train_loader, num_epochs)

    # Test your model
    test(model, test_loader)
