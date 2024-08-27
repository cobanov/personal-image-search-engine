import os

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import generate_embeddings
import utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ? Fix this with a better solution


source_image = "./mertcobanov/2021-06-25_19-15-06_UTC_4.jpg"
index_file = "embeddings/faiss_index.index"
embedding_file = "embeddings/embeddings.npy"
file_paths = "embeddings/file_paths.csv"


index = faiss.read_index(index_file)
embeddings = np.load(embedding_file)
file_paths = pd.read_csv(file_paths)


embeddings.shape
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

model, tokenizer, preprocess = generate_embeddings.get_model_and_tokenizer()

image = utils.read_image(source_image)
image_embedding = generate_embeddings.generate_clip_embedding_from_image(
    image, model, preprocess
)
image_query_vector = np.expand_dims(
    image_embedding, axis=0
)  # Shape should be (1, embedding_dim)


image_query_vector.shape

image_query_vector.squeeze().shape

k = 4  # Number of nearest neighbors to retrieve
distances, indices = index.search(image_query_vector, k)
