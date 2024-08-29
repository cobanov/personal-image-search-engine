import logging
import os
from typing import List

import faiss
import numpy as np
import pandas as pd


def save_image_paths_as_csv(processed_images: List[str], save_path: str) -> None:
    if not isinstance(save_path, str) or not save_path.endswith(".csv"):
        raise ValueError("save_path must be a string ending with '.csv'")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_path = [img["img_path"] for img in processed_images]
    pd.DataFrame(img_path, columns=["img_path"]).to_csv(save_path, index=False)
    logging.info(f"Image paths saved: {len(img_path)} to {os.path.abspath(save_path)}")


def save_image_embeddings_as_npy(embeddings: np.ndarray, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(embeddings))
    logging.info(
        f"Embeddings of shape {embeddings.shape} saved to {os.path.abspath(save_path)}.npy"
    )


def save_image_embeddings_as_faiss(embeddings: np.ndarray, save_path: str) -> None:
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    logging.info(f"Total number of embeddings indexed: {index.ntotal}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)
    logging.info(f"FAISS index saved to {os.path.abspath(save_path)}")
