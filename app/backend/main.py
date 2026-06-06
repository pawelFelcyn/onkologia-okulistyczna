from datetime import date
from enum import Enum
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from inference_service import ( 
    InferenceService,
    MissingDependencyError,
    ModelUnavailableError,
)
from repository import AppRepository, ProcessedScanInput


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
repository = AppRepository()
repository.initialize()
app.mount("/storage", StaticFiles(directory=str(repository.uploads_dir)), name="storage")


class PatientCreate(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


def _normalize_name(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail=f"{field_name} cannot be empty")
    return normalized


def _validate_study_date(study_date: str) -> str:
    try:
        return date.fromisoformat(study_date).isoformat()
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="study_date must use YYYY-MM-DD format",
        ) from exc


def _polygon_area(points: list[list[float]] | list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) / 2.0


def _get_detection_polygon_px(detection: dict[str, Any]) -> list[tuple[float, float]]:
    pixel_segments = detection.get("segments_px") or []
    if len(pixel_segments) >= 3:
        return [
            (float(point[0]), float(point[1]))
            for point in pixel_segments
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]

    normalized_segments = detection.get("segments") or []
    mask_size = detection.get("mask_size") or []
    if len(normalized_segments) >= 3 and len(mask_size) == 2:
        width, height = float(mask_size[0]), float(mask_size[1])
        return [
            (float(point[0]) * width, float(point[1]) * height)
            for point in normalized_segments
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]

    return []


def _estimate_volume_from_detections(study_scans: list[ProcessedScanInput]) -> float:
    tumor_areas: list[float] = []

    for scan in study_scans:
        for detection in scan.detections:
            if detection.get("class") != "tumor":
                continue

            segments_px = _get_detection_polygon_px(detection)
            if len(segments_px) >= 3:
                tumor_areas.append(_polygon_area(segments_px))
                continue

            x1, y1, x2, y2 = detection.get("box", [0, 0, 0, 0])
            tumor_areas.append(max(0.0, (x2 - x1) * (y2 - y1)))

    if not tumor_areas:
        return 0.0

    average_area = sum(tumor_areas) / len(tumor_areas)
    scaled_volume = average_area * len(study_scans) * 35.0
    return round(scaled_volume, 2)


def _process_scan(file_bytes: bytes, model: ModelEnum) -> list[dict[str, Any]]:
    pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
    result = inference_service.infer(model.value, pil_img)
    return result.detections


@app.get("/patients")
def list_patients():
    return {"patients": repository.list_patients()}


@app.post("/patients", status_code=201)
def create_patient(payload: PatientCreate):
    first_name = _normalize_name(payload.first_name, "first_name")
    last_name = _normalize_name(payload.last_name, "last_name")
    patient = repository.create_patient(first_name=first_name, last_name=last_name)
    return patient


@app.get("/patients/{patient_id}")
def get_patient(patient_id: int):
    patient = repository.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/patients/{patient_id}/studies")
def list_patient_studies(patient_id: int):
    patient = repository.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"studies": repository.list_studies_for_patient(patient_id)}


@app.get("/patients/{patient_id}/volume-trend")
def get_patient_volume_trend(patient_id: int):
    patient = repository.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"trend": repository.list_volume_trend(patient_id)}


@app.post("/patients/{patient_id}/studies", status_code=201)
async def create_patient_study(
    patient_id: int,
    study_date: str = Form(...),
    model: ModelEnum = Form(ModelEnum.YOLO),
    files: list[UploadFile] = File(...),
):
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="At least 3 scans are required")

    if repository.get_patient(patient_id) is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    validated_study_date = _validate_study_date(study_date)
    processed_scans: list[ProcessedScanInput] = []

    try:
        for upload in files:
            file_bytes = await upload.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="One of the uploaded files is empty")

            detections = _process_scan(file_bytes, model)
            processed_scans.append(
                ProcessedScanInput(
                    filename=upload.filename or f"scan_{len(processed_scans) + 1}.png",
                    content=file_bytes,
                    detections=detections,
                )
            )
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MissingDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded scan: {exc}") from exc

    volume_mm3 = _estimate_volume_from_detections(processed_scans)

    try:
        study = repository.create_study(
            patient_id=patient_id,
            study_date=validated_study_date,
            selected_model=model.value,
            scans=processed_scans,
            volume_mm3=volume_mm3,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return study


@app.get("/studies/{study_id}")
def get_study(study_id: int):
    study = repository.get_study(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return study


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
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="At least 3 scans are required")

    processed_scans: list[ProcessedScanInput] = []
    for file in files:
        img_bytes = await file.read()
        detections = _process_scan(img_bytes, ModelEnum.YOLO)
        processed_scans.append(
            ProcessedScanInput(
                filename=file.filename or "scan.png",
                content=img_bytes,
                detections=detections,
            )
        )

    volume = _estimate_volume_from_detections(processed_scans)
    return {"volume": volume}
