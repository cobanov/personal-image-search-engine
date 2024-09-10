from typing import List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from engine import utils


class ImageDataset(Dataset):
    """
    A dataset class for loading images from a directory.

    Parameters
    ----------
    img_dir : str
        Directory containing the images.
    transform : Optional[callable], optional
        Optional transform to apply to images (default is None).

    Methods
    -------
    __len__() -> int
        Returns the number of images in the dataset.
    __getitem__(idx: int) -> Tuple[str, any]
        Returns the image path and image at the specified index.
    """

    def __init__(self, img_dir: str, transform: Optional[callable] = None):
        """Initializes ImageDataset with the given directory and optional transform."""
        self.img_paths: List[str] = utils.find_images(img_dir)
        self.transform: Optional[callable] = transform

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[str, any]:
        """Returns the image path and the image at the specified index."""
        img_path = self.img_paths[idx]
        image = utils.read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return img_path, image

        return img_path, image


class ImageLoader:
    """
    Loads images from a directory and manages batching.

    Parameters
    ----------
    image_directory : str
        Directory containing the images.
    batch_size : int, optional
        Number of images per batch (default is 32).

    Methods
    -------
    init_dataloader() -> DataLoader
        Initializes the DataLoader for the image dataset.
    """

    def __init__(self, image_directory: str, batch_size: int = 32):
        """Initializes ImageLoader with the given directory and batch size."""
        self.dataset = ImageDataset(image_directory)
        self.img_paths: List[str] = self.dataset.img_paths
        self.batch_size: int = batch_size

    def _custom_collate_fn(
        self, batch: List[Tuple[str, any]]
    ) -> Tuple[List[str], List[any]]:
        """Custom function for batching image paths and images."""
        img_paths, images = zip(*batch)
        return list(img_paths), list(images)

    def init_dataloader(self) -> DataLoader:
        """Initializes and returns the DataLoader."""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._custom_collate_fn,
        )
        return self.dataloader
