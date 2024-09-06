import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine import utils


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = utils.find_images(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = utils.read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return img_path, image


class ImageLoader:
    def __init__(self, image_directory, batch_size=32):

        self.dataset = ImageDataset(image_directory)
        self.img_paths = self.dataset.img_paths
        self.batch_size = batch_size

    def _custom_collate_fn(self, batch):
        img_paths, images = zip(*batch)
        return list(img_paths), list(images)

    def init_dataloader(self):
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._custom_collate_fn,
        )
        return self.dataloader
