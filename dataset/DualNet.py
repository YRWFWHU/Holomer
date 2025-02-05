import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import torch
import os
from PIL import Image


class Div2k(Dataset):
    def __init__(self,
                 data_dir,
                 transform=transforms.Compose([
                     transforms.RandomResizedCrop(size=(512, 512), scale=(0.4, 1.0)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor()
                 ]),
                 color_channel=None
                 ):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)
        self.color_channel = color_channel

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        img_name = os.path.join(self.data_dir, self.file_list[item])
        pil_img = Image.open(img_name)
        image = self.transform(pil_img).float()
        if self.color_channel == 'rgb':
            return image
        elif self.color_channel == 'red':
            return image[0:1, ...]
        elif self.color_channel == 'green':
            return image[1:2, ...]
        elif self.color_channel == 'blue':
            return image[2:3, ...]
        else:
            raise ValueError('choose color channel from red/green/blue/rgb')


class DIV2K(pl.LightningDataModule):
    def __init__(self, data_dir: str, color_channel: str, train_batch_size=4, val_batch_size=4, pred_batch_size=4,
                 resolution: list = (1024, 2048), num_workers=1):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.pred_batch_size = pred_batch_size
        self.data_dir = data_dir
        self.num_of_workers = num_workers

        if color_channel == 'gray':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((214, 214), antialias=True),
                transforms.Pad(padding=((1024 - 214) // 2, (1024 - 214) // 2, (1024 - 214) // 2, (1024 - 214) // 2),
                               fill=0),
            ])

            self.train_set = Div2k(data_dir=self.data_dir + '/train', transform=self.transform,
                                   color_channel='rgb')
            self.val_set = Div2k(data_dir=self.data_dir + '/val', transform=self.transform, color_channel='rgb')
            self.test_set = Div2k(data_dir=self.data_dir + '/test', transform=self.transform,
                                  color_channel='rgb')
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor()
            ])

            self.train_set = Div2k(data_dir=self.data_dir + '/train', transform=self.transform,
                                   color_channel=color_channel)
            self.val_set = Div2k(data_dir=self.data_dir + '/val', transform=self.transform, color_channel=color_channel)
            self.test_set = Div2k(data_dir=self.data_dir + '/test', transform=self.transform,
                                  color_channel=color_channel)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True,
                          num_workers=self.num_of_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_of_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.pred_batch_size, shuffle=False,
                          num_workers=self.num_of_workers)
