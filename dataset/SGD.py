import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch


class SGDDataModule(pl.LightningDataModule):
    def __init__(self, num_of_steps_each_epoch=10):
        super().__init__()
        self.data = DataLoader(TensorDataset(torch.rand(num_of_steps_each_epoch)))

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.data
