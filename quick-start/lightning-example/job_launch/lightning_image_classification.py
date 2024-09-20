import lightning as L
import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

import numpy as np


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    #     def forward(self, batch):
    #         inputs, target = batch
    #         return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True, sync_dist=True, prog_bar=True)

    def predict_step(self, batch):
        inputs, _ = batch
        return self.model(inputs)


class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["labels"]

        if self.transform:
            image = self.transform(image)

        return image, label


def run():
    ds = load_dataset("Bingsu/Cat_and_Dog")

    batch_size = 256
    num_workers = 12

    ds = ds.shuffle()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = HuggingFaceDataset(ds["train"], transform=transform)
    val_dataset = HuggingFaceDataset(ds["test"], transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    classes = ["cat", "dog"]

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Linear(2048, 2)
    lit_resnet = LitModel(resnet)
    trainer = L.Trainer(max_epochs=10, strategy="ddp", accelerator="gpu")
    trainer.fit(
        model=lit_resnet, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    run()
