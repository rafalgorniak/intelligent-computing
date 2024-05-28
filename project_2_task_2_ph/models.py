import torch
from torch import nn

LEARN_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 64


class DatasetName:
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"


class MNISTCNN2(nn.Module):
    def __init__(self):
        super(MNISTCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Reduced filters
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Reduced filters
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  # Adjusted input size
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return x


class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Reduced filters
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Reduced filters
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  # Adjusted input size
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Cifar10CNN(nn.Module):
    def __init__(self):
        super(Cifar10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Increased filters
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Increased filters
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Added layer
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)  # Added layer

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 2 * 2, 512)  # Adjusted input size
        self.fc2 = nn.Linear(512, 10)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)

        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)

        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)

        x = torch.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(-1, 128 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Cifar10CNN2(nn.Module):
    def __init__(self):
        super(Cifar10CNN2, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Increased filters
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Increased filters
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Added layer
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)  # Added layer

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 2 * 2, 512)  # Adjusted input size
        self.fc2 = nn.Linear(512, 10)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)

        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)

        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)

        x = torch.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(-1, 128 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def forward_conv(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)

        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)

        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)

        x = torch.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(-1, 128 * 2 * 2)
        x = torch.relu(self.fc1(x))

        return x