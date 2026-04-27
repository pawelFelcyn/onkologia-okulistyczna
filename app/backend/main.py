from enum import Enum
from io import BytesIO

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference_service import (
    InferenceService,
    MissingDependencyError,
    ModelUnavailableError,
)


app = FastAPI()

origins = [
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelEnum(str, Enum):
    YOLO = "yolo"
    UNET = "unet"


inference_service = InferenceService()


@app.post("/inference")
async def infer(file: UploadFile = File(...), model: ModelEnum = Form(ModelEnum.YOLO)):
    img_bytes = await file.read()
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")

    try:
        result = inference_service.infer(model.value, pil_img)
        return {"detections": result.detections}
    except ModelUnavailableError as e:
        return {"detections": [], "error": str(e)}
    except MissingDependencyError as e:
        return {"detections": [], "error": str(e)}
    except Exception as e:
        print(f"[WARN] Inference failed: {e}")
        return {"detections": [], "error": "Inference failed"}


@app.post("/volume")
async def calculcate_volume(files: list[UploadFile] = File(...)):
    images = []
    for file in files:
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes))
        images.append(img)

    import random

    volume = random.uniform(1, 16)
    return {"volume": volume}
