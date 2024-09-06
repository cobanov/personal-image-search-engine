import glob
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
from engine.logging_config import log


def find_images(image_directory: str) -> List[str]:
    """
    Find image files in a directory based on a list of extensions.

    :param image_directory: Directory to search for images.
    :return: List of image file paths.
    """
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = [
        img
        for ext in extensions
        for img in glob.glob(os.path.join(image_directory, ext))
    ]
    log.info(f"Images found: {len(image_paths)}")
    return image_paths


def save_filelist(filelist: List[str], filelist_path: str) -> None:
    """
    Save a list of file paths to a text file.

    :param filelist: List of file paths.
    :param filelist_path: Path to save the file list.
    """
    try:
        with open(filelist_path, "w") as f:
            for path in filelist:
                f.write(f"{path}\n")
        log.info(f"File list saved to {filelist_path}")
    except Exception as e:
        log.error(f"Failed to save file list to {filelist_path}: {e}")


# def load_from_filelist(filelist_path: str) -> List[str]:
#     """
#     Load a list of file paths from a text file.

#     :param filelist_path: Path to the file list.
#     :return: List of file paths.
#     """
#     try:
#         with open(filelist_path, "r") as f:
#             filelist = f.read().splitlines()
#         return filelist
#     except Exception as e:
#         log.error(f"Failed to load file list from {filelist_path}: {e}")
#         return []


def read_image(image_path: str) -> Optional[Image.Image]:
    """
    Read an image file and return as a PIL Image.

    :param image_path: Path to the image file.
    :return: PIL Image object or None if failed.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        norm_path = os.path.normpath(image_path)
        log.warning(f"Failed to read {norm_path}: {e}")
        return None


def load_image_as_dict(
    img_path: str, as_array: bool = False
) -> Optional[Dict[str, Image.Image]]:
    """
    Process a single image: open and convert to RGB, return as a dictionary.
    Optionally convert the image to a NumPy array.

    :param img_path: Path to the image.
    :param as_array: Convert image to NumPy array instead of returning as PIL Image.
    :return: Dictionary with the image path and processed image (PIL or NumPy array) or None if failed.
    """
    img = read_image(img_path)
    if img is None:
        return None

    norm_path = os.path.normpath(img_path)
    if as_array:
        img = np.array(img)  # Convert image to NumPy array

    return {"img_path": norm_path, "img": img}


def process_images(
    image_paths: List[str], as_array: bool = False
) -> List[Dict[str, Image.Image]]:
    """
    Process a list of images sequentially.

    :param image_paths: List of image file paths.
    :param as_array: Convert images to NumPy arrays instead of returning PIL Images.
    :return: List of dictionaries containing image paths and processed images.
    """
    results = []
    for img_path in tqdm(image_paths):
        result = load_image_as_dict(img_path, as_array)
        if result is not None:
            results.append(result)

    log.info(f"Images processed successfully: {len(results)}")
    return results


# def process_images_multiprocessing(
#     image_paths: List[str], num_workers: Optional[int] = None, as_array: bool = False
# ) -> List[Dict[str, Image.Image]]:
#     """
#     Process a list of images using multiprocessing.

#     :param image_paths: List of image file paths.
#     :param num_workers: Number of worker processes. Defaults to the number of CPU cores.
#     :param as_array: Convert images to NumPy arrays instead of returning PIL Images.
#     :return: List of dictionaries containing image paths and processed images.
#     """
#     if num_workers is None:
#         num_workers = max(1, cpu_count() - 2)  # Ensure at least 1 worker

#     log.info(f"Using {num_workers} worker processes.")

#     results = []
#     try:
#         with Pool(num_workers) as pool:
#             for result in tqdm(
#                 pool.imap(lambda path: load_image_as_dict(path, as_array), image_paths),
#                 total=len(image_paths),
#             ):
#                 if result is not None:
#                     results.append(result)
#     except Exception as e:
#         log.error(f"Error during multiprocessing: {e}")
#     finally:
#         log.info(f"Images processed successfully: {len(results)}")

#     return results
