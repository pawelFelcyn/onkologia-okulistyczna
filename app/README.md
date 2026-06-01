# AI Eye - OCT Scan Analysis Demo

This application is a clinical tool prototype for analyzing Ophthalmic Computed Tomography (OCT) scans. It now supports a persistent patient registry, dated studies per patient, saved scan sequences, stored segmentation masks, and longitudinal tumor volume tracking.

## Technologies Used

### Backend
*   Python
*   FastAPI
*   Ultralytics (YOLOv8)
*   Uvicorn
*   Pillow
*   Pydantic

### Frontend
*   React
*   TypeScript
*   Vite
*   TailwindCSS
*   Lucide React
*   Canvas API

---

## How to Run

### Model weights (required)

For the application to work, the backend needs YOLO weights and optionally U-Net weights.

The backend now looks for them in the following places, in this order:

- backend/models/...
- repository-level models/...
- repository-level base_models/...

In this repository, the current fallbacks already match existing files:

- models/weights.pt or base_models/yolov8n-seg.pt for YOLO
- models/unet/weights.pth for U-Net

If you want to use different filenames/paths, update the defaults and fallbacks in app/backend/inference_service.py.

### 1. Run Everything with Docker Compose

Prerequisites:
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/)

From the `app/` directory:

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- Backend data persistence: Docker volume mounted at backend/data

### 2. Manual Setup (Backend + Frontend)

Prerequisites:
*   [Python 3.12+](https://www.python.org/)
*   [Node.js 20+](https://nodejs.org/)
*   [uv](https://docs.astral.sh/uv/) (recommended) or `pip`

#### Backend

```bash
cd backend
uv sync  # or: pip install -r pyproject.toml
python run.py
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Features
1.  **Patient Registry**: Create patients using first name and last name.
2.  **Studies Per Patient**: Upload OCT scan sequences for a selected patient and assign a study date.
3.  **Persistent Storage**: Save scans, segmentation results and computed tumor volume with each study.
4.  **Study Browser**: Browse patients and inspect all saved studies for each patient.
5.  **Tumor Volume Trend**: View a longitudinal chart of the algorithm-estimated tumor volume over time.
6.  **Tumor Segmentation (YOLOv8 / U-Net)**: Open any saved scan to inspect stored segmentation masks.
7.  **Comparison View**: Use the interactive slider to compare raw scans with saved AI segmentation overlays.

## Current Workflow
1. Create a patient in the left-side registry.
2. Select that patient and create a new study by providing the study date, model and at least 3 scans.
3. The backend stores the uploaded scans, runs segmentation, computes the current mock tumor volume and saves everything as one dated study.
4. Browse older studies for the patient and open any saved scan in the segmentation viewer.
5. Use the volume chart to compare tumor size changes across time.

## Data Storage

The backend persists application data under backend/data:

- app.db: SQLite database with patients, studies and scan metadata
- uploads/: saved scan files grouped by patient and study

With Docker Compose, this directory is stored in a named volume so the data survives container recreation.
