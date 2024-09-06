from engine import dataset
from tqdm import tqdm
from engine import engine
import logging
from engine import database

logging.basicConfig(
    filename="faulty_images.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

il = dataset.ImageLoader("./dataset/fauna_images")
dl = il.init_dataloader()
eg = engine.EmbeddingGenerator()
db = database.Database("data/fauna-images", "fauna_4k", 512)
tbl = db.get_table()


for img_paths, images in tqdm(dl, desc="Processing Batches", leave=False, ncols=100):
    for img_path, image in zip(img_paths, images):

        try:
            embedding = eg.generate_image_embedding(image)
        except Exception as e:
            logging.error(f"Error generating embedding for image {img_path}: {str(e)}")
            continue  # Skip this image and move to the next one

        try:
            data = [{"vector": embedding, "img_path": img_path}]
            tbl.add(data)
        except Exception as e:
            logging.error(
                f"Error writing data to the database for image {img_path}: {str(e)}"
            )
