# AI Eye - OCT Scan Analysis Demo

This application is a clinical tool prototype for analyzing Ophthalmic Computed Tomography (OCT) scans. It features automated tumor segmentation using YOLOv8 and volumetric estimation for scan sequences.

## 🛠 Technologies Used

### Backend
*   **Python 3.12**
*   **FastAPI 0.128.0**: High-performance web framework for the API.
*   **Ultralytics 8.4.6**: Used for YOLOv8 computer vision tasks (segmentation).
*   **Uvicorn 0.40.0**: ASGI server for running the FastAPI application.
*   **Pillow 12.1.0**: Image processing library.
*   **Pydantic 2.12.5**: Data validation and settings management.

### Frontend
*   **React 19.2.0**: UI library.
*   **TypeScript 5.9.3**: Static typing for enhanced developer experience.
*   **Vite 7.2.4**: Modern frontend build tool and dev server.
*   **TailwindCSS 4.1.18**: Utility-first CSS framework for styling.
*   **Lucide React 0.562.0**: Icon library.
*   **Canvas API**: Used for rendering clinical masks and overlays on OCT scans.

---

## 🚀 How to Run

### Prerequisites
*   [Python 3.12+](https://www.python.org/)
*   [Node.js 20+](https://nodejs.org/)
*   [uv](https://docs.astral.sh/uv/) (Recommended for backend management) or `pip`

### 1. Run the Backend
The backend handles AI inference and volume calculations.

```bash
cd backend
# Install dependencies
uv sync  # or: pip install -r pyproject.toml

# Start the server
python run.py
# The API will be available at http://localhost:8000
```

### 2. Run the Frontend
The frontend provides the interactive clinical interface.

```bash
cd frontend
# Install dependencies
npm install

# Start the development server
npm run dev
# The app will be available at http://localhost:5173
```

## 🔍 Features
1.  **Sequence Upload**: Upload multiple OCT scans simultaneously.
2.  **Tumor Segmentation**: Click "Analyze" on any scan to view real-time AI segmentation masks.
3.  **Volume Estimation**: Select 3 or more scans in a sequence to calculate estimated tumor volume (mm³).
4.  **Comparison View**: Use the interactive slider to compare raw scans with AI-segmented results.
