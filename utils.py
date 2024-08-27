import glob
import logging
import os

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def read_images_from_directory(image_directory: str) -> list:
    list_of_images = []
    image_extensions = ("*.png", "*.jpg", "*.jpeg")
    for ext in image_extensions:
        list_of_images.extend(glob.glob(os.path.join(image_directory, ext)))
    logging.info(f"Images found: {len(list_of_images)}")
    return list_of_images


def read_with_pil(list_of_images: list, resize: bool = True) -> list:
    pil_images = []
    for img_path in tqdm(list_of_images, desc="Reading images"):
        try:
            img = Image.open(img_path).convert("RGB")
            if resize:
                img.thumbnail((512, 512))
            pil_images.append(img)
        except Exception as e:
            logging.warning(f"Failed to process {img_path}: {e}")
    logging.info(f"Total images read: {len(pil_images)}")
    return pil_images


def read_image(image_path: str) -> Image.Image:
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Failed to read image at {image_path}: {e}")
        return None
