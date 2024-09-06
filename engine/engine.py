import logging

import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

from engine import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class EmbeddingModel:
    def __init__(
        self, model_name: str = "ViT-B-32", pretrained_model: str = "laion2b_s34b_b79k"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, self.preprocess = self._load_model_and_tokenizer(
            model_name, pretrained_model
        )

    def _load_model_and_tokenizer(self, model_name: str, pretrained_model: str):
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_model
            )
            model = model.to(self.device)
            model.eval()
            tokenizer = open_clip.get_tokenizer(model_name)
            logging.info(
                f"Successfully loaded model {model_name} with pretrained weights {pretrained_model}"
            )
            return model, tokenizer, preprocess
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    @staticmethod
    def save_embeddings_as_npy(embeddings: np.ndarray, save_path: str) -> None:
        np.save(save_path, np.array(embeddings))
        logging.info(f"Embeddings saved to {save_path}.npy")

    @staticmethod
    def save_file_paths(file_paths: list, save_path: str) -> None:
        if not save_path.endswith(".csv"):
            save_path += ".csv"
        df = pd.DataFrame({"file_path": file_paths})
        df.to_csv(save_path, index=True)
        logging.info(f"File paths saved to {save_path}.csv")

    # @staticmethod
    # def save_faiss_index(embeddings: np.ndarray, save_path: str) -> None:
    #     embedding_dim = embeddings.shape[1]
    #     index = faiss.IndexFlatL2(embedding_dim)
    #     index.add(embeddings)
    #     logging.info(f"Total number of embeddings indexed: {index.ntotal}")
    #     faiss.write_index(index, save_path)
    #     logging.info(f"FAISS index saved to {save_path}")


class EmbeddingGenerator(EmbeddingModel):
    def __init__(
        self, model_name: str = "ViT-B-32", pretrained_model: str = "laion2b_s34b_b79k"
    ):
        super().__init__(model_name, pretrained_model)

    def generate_text_embedding(self, text: str) -> np.ndarray:
        text_input = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.squeeze(0)
        return text_features.cpu().numpy()

    def generate_image_embedding(self, image) -> np.ndarray:
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.squeeze(0)
        return image_features.cpu().numpy()

    def generate_image_embedding_from_path(self, image_path: str) -> np.ndarray:
        try:
            image = utils.read_image(image_path)
            return self.generate_image_embedding(image)
        except FileNotFoundError:
            logging.error(f"Image file {image_path} not found.")
            raise
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            raise

    def generate_image_embeddings(
        self, images: list, batch_size: int = 32
    ) -> np.ndarray:
        embeddings = []
        num_images = len(images)
        num_batches = (num_images + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Generating embeddings"):
                batch_images = images[i * batch_size : (i + 1) * batch_size]
                try:
                    batch_inputs = torch.stack(
                        [self.preprocess(image) for image in batch_images]
                    ).to(self.device)
                    batch_features = self.model.encode_image(batch_inputs)
                    batch_features /= batch_features.norm(dim=-1, keepdim=True)
                    embeddings.extend(batch_features.cpu().numpy())
                except RuntimeError as e:
                    logging.error(
                        f"Error processing batch {i}, trying smaller batch size: {e}"
                    )
                    batch_size = max(1, batch_size // 2)  # Reduce the batch size
                    continue

        logging.info(f"Total embeddings generated: {len(embeddings)}")
        return np.array(embeddings)

    def calculate_probabilities(
        self, image_features: np.ndarray, text_features: np.ndarray
    ) -> tuple:
        image_features = torch.tensor(image_features).to(self.device)
        text_features = torch.tensor(text_features).to(self.device)
        text_probs = torch.nn.functional.softmax(
            100.0 * image_features @ text_features.T, dim=-1
        )
        image_probs = torch.nn.functional.softmax(
            100.0 * text_features @ image_features.T, dim=-1
        )
        return text_probs.cpu().numpy(), image_probs.cpu().numpy()

    @staticmethod
    def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return np.dot(embedding_a, embedding_b) / (
            np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
        )


# class QueryEngine(EmbeddingGenerator):
#     def __init__(
#         self, model_name: str = "ViT-B-32", pretrained_model: str = "laion2b_s34b_b79k"
#     ):
#         super().__init__(model_name, pretrained_model)
#         self.index = None

#     def load_faiss_index(self, index_path: str) -> None:
#         try:
#             self.index = faiss.read_index(index_path)
#             logging.info(f"FAISS index loaded from {index_path}")
#         except Exception as e:
#             logging.error(f"Error loading FAISS index: {e}")
#             raise

#     def search_images(self, image, k: int = 5) -> tuple:
#         if self.index is None:
#             raise ValueError(
#                 "FAISS index has not been loaded. Please load an index first."
#             )
#         image_embedding = self.generate_image_embedding(image)
#         image_query_vector = np.expand_dims(image_embedding, axis=0)
#         distances, indices = self.index.search(image_query_vector, k)
#         return distances, indices

#     def search_text(self, text: str, k: int = 5) -> tuple:
#         if self.index is None:
#             raise ValueError(
#                 "FAISS index has not been loaded. Please load an index first."
#             )
#         text_embedding = self.generate_text_embedding(text)
#         text_query_vector = np.expand_dims(text_embedding, axis=0)
#         distances, indices = self.index.search(text_query_vector, k)
#         return distances, indices


# Usage Example:
# clip_generator = EmbeddingGenerator()
# text_embedding = clip_generator.generate_text_embedding("example text")
# image_embedding = clip_generator.generate_image_embedding(image)
# image_embeddings = clip_generator.generate_image_embeddings(images)
# clip_generator.save_embeddings_as_npy(image_embeddings, "embeddings_path")
# clip_generator.save_faiss_index(image_embeddings, "faiss_index_path")
