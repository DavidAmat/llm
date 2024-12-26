import os

import lightning as L
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient


class MNISTModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


if __name__ == "__main__":

    # -----------------------
    # 1) Tell MLflow to talk to our server
    # -----------------------
    mlflow.set_tracking_uri("http://localhost:5000")
    # Optionally set an experiment name if desired
    mlflow.set_experiment("NewExperiment")

    # -----------------------
    # 2) Initialize our model and dataset
    # -----------------------
    mnist_model = MNISTModel()

    train_ds = MNIST(
        os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
    )

    indices = torch.arange(32)
    train_ds = Subset(train_ds, indices)
    train_loader = DataLoader(train_ds, batch_size=8)

    # -----------------------
    # 3) Initialize Lightning Trainer
    # -----------------------
    trainer = L.Trainer(max_epochs=3)

    # -----------------------
    # 4) Enable auto-logging
    # -----------------------
    mlflow.pytorch.autolog()

    # -----------------------
    # 5) Train the model
    # -----------------------
    with mlflow.start_run() as run:
        trainer.fit(mnist_model, train_loader)

    # -----------------------
    # 6) Print out what was auto-logged
    # -----------------------
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
