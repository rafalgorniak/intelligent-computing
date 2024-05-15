import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = torch.tensor(target, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.target[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label