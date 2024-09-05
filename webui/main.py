import os
import sys

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

sys.path.append(os.path.abspath("../engine"))

from engine import EmbeddingGenerator, QueryEngine
from utils import read_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount(
    "/dataset",
    StaticFiles(
        directory="C:/Users/hope/Desktop/developer/personal-image-search-engine/dataset"
    ),
    name="dataset",
)

templates = Jinja2Templates(directory="templates")


index_file_path = r"C:\Users\hope\Desktop\developer\personal-image-search-engine\embeddings\flora_sample.index"
file_paths_path = r"C:\Users\hope\Desktop\developer\personal-image-search-engine\embeddings\flora_sample.csv"

query_engine = QueryEngine()

query_engine.load_faiss_index(index_file_path)
file_paths = pd.read_csv(file_paths_path)["img_path"]


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_location = f"uploaded_images/{file.filename}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    img = read_image(file_location)
    distances, indices = query_engine.search_images(img, 24)
    nearest_neighbors_paths = [file_paths[i] for i in indices[0]]

    return {"image_paths": nearest_neighbors_paths}


@app.post("/search/")
async def search(query: str = Form(...)):
    distances, indices = query_engine.search_text(query, 24)
    nearest_neighbors_paths = [file_paths[i] for i in indices[0]]

    return {"image_paths": nearest_neighbors_paths}
