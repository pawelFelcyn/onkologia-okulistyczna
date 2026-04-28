# AI Eye - OCT Scan Analysis Demo

This application is a clinical tool prototype for analyzing Ophthalmic Computed Tomography (OCT) scans. It features automated tumor segmentation (model can be selected: YOLOv8 or U-Net) and volumetric estimation for scan sequences.

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

For the application to work, the backend must have a models/ directory with the following files:

- backend/models/yolo-weights.pt (required; backend will fail to start without it)
- backend/models/unet.pth (required only if you want to use U-Net; if missing, selecting U-Net will return “model unavailable”)

If you want to use different filenames/paths, update the defaults in
[app/backend/inference_service.py](http://_vscodecontentref_/3) (and how it’s instantiated in
[app/backend/main.py](http://_vscodecontentref_/4)).

### 1. Run Everything with Docker Compose

Prerequisites:
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/)

From the `app/` directory:

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

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
1.  **Sequence Upload**: Upload multiple OCT scans simultaneously.
2.  **Tumor Segmentation (YOLOv8 / U-Net)**: Choose the model and click "Analyze" on any scan to view AI segmentation masks.
3.  **Volume Estimation**: Select 3 or more scans in a sequence to calculate estimated tumor volume (mm³).
4.  **Comparison View**: Use the interactive slider to compare raw scans with AI-segmented results.
