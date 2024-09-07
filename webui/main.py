import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from webui.routes import router

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

# Static files directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Dataset directory
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset")
app.mount("/dataset", StaticFiles(directory=dataset_dir), name="dataset")

app.include_router(router)
