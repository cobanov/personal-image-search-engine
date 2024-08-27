import logging

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def get_model_and_tokenizer(model_name="ViT-B-32"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, preprocess


def generate_clip_embedding_from_text(text, model, tokenizer):
    text_input = tokenizer([text])
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.squeeze(0)
    return text_features.cpu().numpy()


def generate_clip_embedding_from_image(image, model, preprocess):
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.squeeze(0)
    return image_features.cpu().numpy()


def generate_clip_embeddings_from_images(images, model, preprocess):
    embeddings = []
    with torch.no_grad():
        for image in tqdm(images, desc="Generating embeddings"):
            image_input = preprocess(image).unsqueeze(0)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.squeeze(0)
            embeddings.append(image_features.cpu().numpy())
    return np.array(embeddings)


def calculate_prob(image_features, text_features):  #! Not Sure!
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    image_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    return text_probs, image_probs


def save_embeddings_as_npy(embeddings, save_path):
    print(embeddings.shape)
    np.save(save_path, np.array(embeddings))
    logging.info(f"Embeddings saved to {save_path}.npy")


def save_file_paths(file_paths, save_path):
    df = pd.DataFrame({"file_path": file_paths})
    df.to_csv(save_path, index=True)
    logging.info(f"File paths saved to {save_path}.csv")


def save_faiss_index(embeddings, save_path):
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    print(f"Total number of embeddings indexed: {index.ntotal}")
    faiss.write_index(index, save_path)
    logging.info(f"FAISS index saved to {save_path}")
