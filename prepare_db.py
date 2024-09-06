from engine import dataset
from engine import engine
from engine import database
from engine.logging_config import log

from tqdm.rich import trange, tqdm


il = dataset.ImageLoader("./dataset/flora_images", batch_size=64)
dl = il.init_dataloader()
eg = engine.EmbeddingGenerator(
    model_name="ViT-B-32", pretrained_model="laion2b_s34b_b79k"
)
db = database.Database("database/nature", "flora_20k", 512)
tbl = db.get_table()  #! This unnecesary!


for img_paths, images in tqdm(dl, desc="Processing Batches"):
    for img_path, image in zip(img_paths, images):

        try:
            embedding = eg.generate_image_embedding(image)
        except Exception as e:
            log.error(f"Error generating embedding for image {img_path}: {str(e)}")
            continue  # Skip this image and move to the next one

        try:
            data = [{"vector": embedding, "img_path": img_path}]
            tbl.add(data)
        except Exception as e:
            log.error(
                f"Error writing data to the database for image {img_path}: {str(e)}"
            )
