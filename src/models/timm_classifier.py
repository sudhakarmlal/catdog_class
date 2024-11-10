import lightning as L
import timm
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy


class TimmClassifier(L.LightningModule):
    def __init__(
        self,
        base_model: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained model
        self.model = timm.create_model(
            base_model, pretrained=pretrained, num_classes=num_classes
        )

        # Multi-class accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.val_acc(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.factor,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }
