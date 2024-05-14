import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

num_epochs = 5
image_size = 28 * 28
hidden_size = 128
output_size = 10

'''
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
'''

transform = transforms.Compose([
        transforms.ToTensor()
    ])

criterion = nn.CrossEntropyLoss()

def getMnistTrain():
    return datasets.MNIST(root='./data', train=True, transform=transform, download=True)



def getMnistTest():
    return datasets.MNIST(root='./data', train=False, transform=transform)

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
        for images, labels in train_loader:
            images = images.view(-1, image_size)
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

    train_loader = torch.utils.data.DataLoader(dataset=getMnistTrain(), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=getMnistTest(), batch_size=64, shuffle=False)

    model = customMLP(image_size, hidden_size, output_size)

    # Definicja funkcji straty i optymalizatora
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Trening modelu
    train(model, optimizer, train_loader, num_epochs)

    # Testowanie modelu
    test(model, test_loader)
