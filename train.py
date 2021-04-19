import torch
import torchvision.models as models
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning import LightningModule, Trainer


class LiftModel(LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        
    def step(self, batch, stage):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def train_dataloader(self, *args, **kwargs):
        ds = CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        return torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
             
    def val_dataloader(self, *args, **kwargs):
        ds = CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        return torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)


if __name__ == "__main__":
    model = LiftModel(models.resnet50(pretrained=True))
    trainer = Trainer(max_epochs=10, gpus=2, profiler="pytorch", accelerator="ddp")
    trainer.fit(model)
    trainer.validate(model)