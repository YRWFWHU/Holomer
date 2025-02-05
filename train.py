from distutils.command.config import config

from pytorch_lightning.cli import LightningCLI

from algorithms.SGD import SGD
from algorithms.DoubleStage import DoubleStage
from algorithms.End2End import End2End
from dataset.SGD import SGDDataModule
from dataset.DualNet import DIV2K


"""
Available Model: SGD, DoubleStage, End2End
--------------
Available DataModule: DIV2K
"""

cli = LightningCLI(model_class=DoubleStage, datamodule_class=DIV2K)
# cli = LightningCLI(model_class=SGD, datamodule_class=SGDDataModule)
