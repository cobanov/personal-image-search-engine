import glob
import logging
import os
from multiprocessing import Pool, cpu_count
from typing import List, Optional

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def find_images(image_directory: str, extensions: List[str]) -> List[str]:
    """
    Find image files in a directory based on a list of extensions.

    :param image_directory: Directory to search for images.
    :param extensions: List of image file extensions.
    :return: List of image file paths.
    """
    return [
        img
        for ext in extensions
        for img in glob.glob(os.path.join(image_directory, ext))
    ]


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


def read_images_with_pil(
    image_paths: List[str], multiprocess: bool = False
) -> List[Image.Image]:
    """
    Reads images using PIL, with optional multiprocessing.

    :param image_paths: List of image file paths.
    :param multiprocess: Whether to use multiprocessing.
    :return: List of PIL Image objects.
    """
    if multiprocess:
        with Pool(cpu_count()) as pool:
            pil_images = list(
                tqdm(
                    pool.imap(process_image, image_paths),
                    total=len(image_paths),
                    desc="Reading images",
                )
            )
    else:
        pil_images = [
            process_image(img_path)
            for img_path in tqdm(image_paths, desc="Reading images")
            if process_image(img_path) is not None
        ]

    pil_images = [
        img for img in pil_images if img is not None
    ]  # Filter out failed images
    logging.info(f"Total images read: {len(pil_images)}")
    return pil_images


def read_images_from_directory(
    image_directory: str, multiprocess: bool = False
) -> List[str]:
    """
    Reads image file paths from a directory, with optional multiprocessing.

    :param image_directory: Directory to search for images.
    :param multiprocess: Whether to use multiprocessing.
    :return: List of image file paths.
    """
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]

    if multiprocess:
        with Pool(cpu_count()) as pool:
            image_paths = [
                img
                for sublist in tqdm(
                    pool.starmap(
                        find_images,
                        [(image_directory, [ext]) for ext in image_extensions],
                    ),
                    total=len(image_extensions),
                    desc="Finding images",
                )
                for img in sublist
            ]
    else:
        image_paths = find_images(image_directory, image_extensions)

    logging.info(f"Images found: {len(image_paths)}")
    return image_paths


# Example usage
if __name__ == "__main__":
    directory = "path/to/your/images"

    # Get the list of image paths
    image_paths = read_images_from_directory(directory, multiprocess=True)

    # Convert the image paths to PIL images
    pil_images = read_images_with_pil(image_paths, multiprocess=True)
