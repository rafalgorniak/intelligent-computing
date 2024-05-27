import pytorch_lightning as pl
import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LitModel(pl.LightningModule):
    def __init__(self, model, lr=0.001):
        super(LitModel, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = self(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = self(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer