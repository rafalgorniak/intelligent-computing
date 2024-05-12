import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
'''
num_epochs = 5
image_size = 28 * 28
hidden_size = 128
output_size = 10

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

criterion = nn.CrossEntropyLoss()

def getMnistTrain():
    return datasets.MNIST(root='./data', train=True, transform=transform, download=True)


def getTrainSetByColor():
    data = getMnistTrain()


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
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define transformation to convert image to 2-element vector
# Define transformation to convert image to 2-element vector
# Define transformation to convert image to 2-element vector
# Define transformation to convert image to 2-element vector
def transform_to_contour_avg(image):
    # Convert torch tensor to numpy array
    image_np = image.numpy()

    # Convert image to binary format (bright pixels: 1, black pixels: 0)
    binary_image = (image_np > 0).astype(int)

    # Define kernel for checking neighbors
    kernel = torch.tensor([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

    # Get dimensions of the image
    rows, cols = binary_image.shape[:2]

    # Initialize contour length and sum of pixel values
    contour_length = 0
    sum_pixel_values = 0

    # Iterate through each pixel
    for i in range(rows):
        for j in range(cols):
            # Check if current pixel is bright (non-black)
            if binary_image[i, j] == 1:
                is_contour = False
                # Check neighbors
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        # Check if neighbor is within bounds and is black
                        if 0 <= ni < rows and 0 <= nj < cols and kernel[di + 1, dj + 1] == 0 and binary_image[ni, nj] == 0:
                            is_contour = True
                            break  # Exit inner loop if a black neighbor is found
                    if is_contour:
                        break  # Exit outer loop if a black neighbor is found
                if is_contour:
                    contour_length += 1

                # Add pixel value to sum
                sum_pixel_values += image_np[i, j]

    # Compute average pixel value
    avg_pixel_value = sum_pixel_values / (torch.sum(image > 0).float())

    return torch.tensor([contour_length, avg_pixel_value], dtype=torch.float32)


# Define the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transform_to_contour_avg
])

# Define the MNIST train dataset with the new transformation
def getMnistTrain():
    return datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Define the MNIST test dataset with the new transformation
def getMnistTest():
    return datasets.MNIST(root='./data', train=False, transform=transform)

# Define the custom MLP model
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

# Define the training function
def train(model, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss/len(train_loader)))

# Define the testing function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy on the test set: %.2f %%' % accuracy)

if __name__ == "__main__":
    num_epochs = 50
    input_size = 2  # Number of elements in transformed vector
    hidden_size = 128
    output_size = 10

    train_loader = torch.utils.data.DataLoader(dataset=getMnistTrain(), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=getMnistTest(), batch_size=64, shuffle=False)

    model = customMLP(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train(model, optimizer, train_loader, num_epochs)
    test(model, test_loader)
