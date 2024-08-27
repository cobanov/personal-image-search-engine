import segaf.utils as utils
from segaf.segaf import EmbeddingGenerator

# Initialize the model
eg = EmbeddingGenerator()

# Generate embeddings from folder
image_folder = "./dataset/random"
image_paths = utils.read_images_from_directory(image_folder)
imgs = utils.read_with_pil(image_paths)

# Generate embeddings from images
img_embeddings = eg.generate_image_embeddings(imgs)

# Save the embeddings, file paths and faiss index
eg.save_embeddings_as_npy(img_embeddings, "embeddings/random_images_embeddings.npy")
eg.save_file_paths(image_paths, "embeddings/random_images_paths.csv")
eg.save_faiss_index(img_embeddings, "embeddings/random_images_faiss.index")
