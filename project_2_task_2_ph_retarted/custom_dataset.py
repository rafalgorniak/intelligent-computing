import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

class DatasetName:
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"

class CustomDataset(Dataset):
    def __init__(self, dataset_name, train=True, use_augmentation=False, limit=None):
        self.dataset_name = dataset_name
        self.train = train
        self.use_augmentation = use_augmentation
        self.transform = self.get_transform()

        if dataset_name == DatasetName.MNIST:
            self.dataset = torchvision.datasets.MNIST(root='./data', train=train, transform=self.transform, download=True)
        elif dataset_name == DatasetName.CIFAR10:
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train, transform=self.transform, download=True)
        else:
            raise ValueError("Unsupported dataset. Please use 'MNIST' or 'CIFAR10'.")

        if limit:
            self.dataset = Subset(self.dataset, range(limit))

    def get_transform(self):
        if self.dataset_name == DatasetName.MNIST:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.dataset_name == DatasetName.CIFAR10:
            if self.use_augmentation:
                return transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label
