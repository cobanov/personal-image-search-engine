import os

import pandas as pd

import embedding_model
import utils
from plotting import plot_list_of_images

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize the embedding model
query_engine = embedding_model.QueryEngine()

# Load the faiss index
query_engine.load_faiss_index("./embeddings/faiss_index.index")

# Query an image
source_image = r"dataset\random\2023-04-02_15-15-53_UTC.jpg"
file_paths = "embeddings/file_paths.csv"

# Load the file paths
file_paths = pd.read_csv(file_paths)["file_path"]
img = utils.read_image(source_image)

# Search for the nearest neighbors
distances, indices = query_engine.search_images(img, 4)
nearest_neighbors_paths = [file_paths[i] for i in indices[0]]

# Plot the source image and the nearest neighbors
plot_list_of_images(source_image, nearest_neighbors_paths)
