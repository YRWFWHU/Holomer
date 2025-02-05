import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Div2k(Dataset):
    def __init__(self,
                 data_dir,
                 transform=transforms.Compose([
                     transforms.RandomResizedCrop(size=(512, 512), scale=(0.4, 1.0)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor()
                 ]),
                 ):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = os.path.join(self.data_dir, self.file_list[index])
        pil_img = Image.open(img_name)

        image = self.transform(pil_img).float()

        return image
