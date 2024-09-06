import os
import sys

import lancedb
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../engine")))
from engine import EmbeddingGenerator  # Import the EmbeddingGenerator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

dataset_dir = "C:/Users/hope/Desktop/developer/personal-image-search-engine/dataset"
app.mount("/dataset", StaticFiles(directory=dataset_dir), name="dataset")

templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


db = lancedb.connect("./database/nature/")
tbl = db.open_table("flora_20k_multi")

eg = EmbeddingGenerator(model_name="ViT-B-32", pretrained_model="laion2b_s34b_b79k")


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_location = f"uploaded_images/{file.filename}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    image_embedding = eg.generate_image_embedding_from_path(file_location)
    neighbors = tbl.search(image_embedding).limit(24).to_pandas()
    neighbors_distances = neighbors["_distance"].to_list()
    neighbors_paths = neighbors["img_path"].to_list()

    return {"image_paths": neighbors_paths}


@app.post("/search/")
async def search(query: str = Form(...)):
    text_embedding = eg.generate_text_embedding(query)
    neighbors = tbl.search(text_embedding).limit(24).to_pandas()
    neighbors_distances = neighbors["_distance"].to_list()
    neighbors_paths = neighbors["img_path"].to_list()

    return {"image_paths": neighbors_paths}
