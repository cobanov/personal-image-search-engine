import os

import pandas as pd
import segaf.segaf as segaf
import segaf.utils as utils
from segaf.plotting import plot_list_of_images

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize the embedding model
query_engine = segaf.QueryEngine()

# Load the faiss index
query_engine.load_faiss_index("./embeddings/faiss_index.index")

# Query an image
text_query = r"a puppy dog"
file_paths = "embeddings/file_paths.csv"

# Load the file paths
file_paths = pd.read_csv(file_paths)["file_path"]

# Search for the nearest neighbors
distances, indices = query_engine.search_text(text_query, 4)
nearest_neighbors_paths = [file_paths[i] for i in indices[0]]

# Plot the source image and the nearest neighbors
plot_list_of_images(None, nearest_neighbors_paths)