import logging

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class EmbeddingModel:
    def __init__(self, model_name="ViT-B-32", pretrained_model="laion2b_s34b_b79k"):
        self.model, self.tokenizer, self.preprocess = self._load_model_and_tokenizer(
            model_name, pretrained_model
        )

    def _load_model_and_tokenizer(self, model_name, pretrained_model):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_model
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, tokenizer, preprocess

    @staticmethod
    def save_embeddings_as_npy(embeddings, save_path):
        print(embeddings.shape)
        np.save(save_path, np.array(embeddings))
        logging.info(f"Embeddings saved to {save_path}.npy")

    @staticmethod
    def save_file_paths(file_paths, save_path):
        df = pd.DataFrame({"file_path": file_paths})
        df.to_csv(save_path, index=True)
        logging.info(f"File paths saved to {save_path}.csv")

    @staticmethod
    def save_faiss_index(embeddings, save_path):
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)
        print(f"Total number of embeddings indexed: {index.ntotal}")
        faiss.write_index(index, save_path)
        logging.info(f"FAISS index saved to {save_path}")


class EmbeddingGenerator(EmbeddingModel):
    def __init__(self, model_name="ViT-B-32", pretrained_model="laion2b_s34b_b79k"):
        super().__init__(model_name, pretrained_model)

    def generate_text_embedding(self, text):
        text_input = self.tokenizer([text])
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.squeeze(0)
        return text_features.cpu().numpy()

    def generate_image_embedding(self, image):
        image_input = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.squeeze(0)
        return image_features.cpu().numpy()

    def generate_image_embeddings(self, images):
        embeddings = []
        with torch.no_grad():
            for image in tqdm(images, desc="Generating embeddings"):
                image_input = self.preprocess(image).unsqueeze(0)
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.squeeze(0)
                embeddings.append(image_features.cpu().numpy())
        return np.array(embeddings)

    def calculate_probabilities(self, image_features, text_features):  #! Not Sure!
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        image_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        return text_probs, image_probs


class QueryEngine(EmbeddingGenerator):
    def __init__(self, model_name="ViT-B-32", pretrained_model="laion2b_s34b_b79k"):
        super().__init__(model_name, pretrained_model)

    def load_faiss_index(self, index_path):
        self.index = faiss.read_index(index_path)

    def search_images(self, image, k=5):
        image_embedding = self.generate_image_embedding(image)
        image_query_vector = np.expand_dims(image_embedding, axis=0)
        distances, indices = self.index.search(image_query_vector, k)
        return distances, indices

    def search_text(self, text, index, k=5):
        text_embedding = self.generate_text_embedding(text)
        text_query_vector = np.expand_dims(text_embedding, axis=0)
        distances, indices = index.search(text_query_vector, k)
        return distances, indices


# Usage Example:
# clip_generator = EmbeddingGenerator()
# text_embedding = clip_generator.generate_text_embedding("example text")
# image_embedding = clip_generator.generate_image_embedding(image)
# image_embeddings = clip_generator.generate_image_embeddings(images)
# clip_generator.save_embeddings_as_npy(image_embeddings, "embeddings_path")
# clip_generator.save_faiss_index(image_embeddings, "faiss_index_path")
