import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import tensorflow as tf
import numpy as np

'''
# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, _), (_, _) = mnist.load_data()

# Normalize pixel values
train_images = train_images / 255.0

# Flatten images and count white and black pixels
threshold = 0.5
num_white_pixels = np.sum(train_images > threshold, axis=(1, 2))
num_black_pixels = np.sum(train_images <= threshold, axis=(1, 2))

# Create 2-element vectors
vectors = np.column_stack((num_white_pixels, num_black_pixels))

# Print number of black and white pixels for the first 10 vectors
for i in range(10):
    print("Vector", i+1, "- Number of White Pixels:", vectors[i][0], "- Number of Black Pixels:", vectors[i][1])

'''

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

def train(model, criterion, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(-1, 28*28)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss/len(train_loader)))

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy on the test set: %.2f %%' % accuracy)

if __name__ == "__main__":
    # Przygotowanie danych
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=50, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=50, shuffle=False)

    # Definicja modelu
    input_size = 28*28
    hidden_size = 128
    output_size = 10

    model = customMLP(input_size, hidden_size, output_size)

    # Definicja funkcji straty i optymalizatora
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Trening modelu
    num_epochs = 10
    train(model, criterion, optimizer, train_loader, num_epochs)

    # Testowanie modelu
    test(model, test_loader)
