from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

model = YOLO(r"models/yolo-weights.pt")
app = FastAPI()
origins = [
    "http://localhost:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/inference")
async def infer(file: UploadFile = File(...)):
    img_bytes = await file.read()
    # img = Image.open(BytesIO(img_bytes)).convert(
    # "RGB")  # Ensure 3 channels for YOLO
    img = Image.open(BytesIO(img_bytes))

    results = model(img)
    r = results[0]

    detections = []
    for i in range(len(r.boxes)):
        det = {
            "class": r.names[int(r.boxes[i].cls)],
            "conf": float(r.boxes[i].conf),
            "box": r.boxes[i].xyxy[0].tolist(),
        }
        # Add mask coordinates if available
        if r.masks is not None:
            det["segments"] = r.masks.xyn[i].tolist()  # Normalized coordinates

        detections.append(det)

    return {"detections": detections}


def run_inference(img):
    return model(img)


@app.post("/volume")
async def calculcate_volume(files: list[UploadFile] = File(...)):
    images = []
    for file in files:
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes))
        images.append(img)
    import random
    volume = random.uniform(1000, 5000)

    return {"volume": volume}
