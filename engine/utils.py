import glob
import logging
import os
from multiprocessing import Pool, cpu_count
from typing import List, Optional

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def read_images_from_directory(image_directory: str) -> List[str]:
    """
    Reads image file paths from a directory.

    :param image_directory: Directory to search for images.
    :return: List of image file paths.
    """
    list_of_images = []
    image_extensions = ("*.png", "*.jpg", "*.jpeg")
    for ext in image_extensions:
        list_of_images.extend(glob.glob(os.path.join(image_directory, ext)))
    logging.info(f"Images found: {len(list_of_images)}")
    return list_of_images


def process_image(img_path: str) -> Optional[Image.Image]:
    """
    Process a single image: open and convert to RGB.

    :param img_path: Path to the image.
    :return: Processed PIL Image or None if failed.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        return img
    except Exception as e:
        logging.warning(f"Failed to process {img_path}: {e}")
        return None


def read_with_pil(
    list_of_images: List[str], multiprocess: bool = False
) -> List[Image.Image]:
    """
    Reads images using PIL, with optional multiprocessing.

    :param list_of_images: List of image file paths.
    :param multiprocess: Whether to use multiprocessing.
    :return: List of PIL Image objects.
    """
    if multiprocess:
        with Pool(cpu_count()) as pool:
            pil_images = list(
                tqdm(
                    pool.imap(process_image, list_of_images),
                    total=len(list_of_images),
                    desc="Reading images",
                )
            )
    else:
        pil_images = []
        for img_path in tqdm(list_of_images, desc="Reading images"):
            img = process_image(img_path)
            if img:
                pil_images.append(img)

    pil_images = [
        img for img in pil_images if img is not None
    ]  # Filter out failed images
    logging.info(f"Total images read: {len(pil_images)}")
    return pil_images


def read_image(image_path: str) -> Optional[Image.Image]:
    """
    Reads a single image from a file path.

    :param image_path: Path to the image file.
    :return: PIL Image object or None if failed.
    """
    return process_image(image_path)
