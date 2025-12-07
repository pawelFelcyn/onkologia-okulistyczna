# Local setup for Label Studio + HTTP image server

Short guide to run Label Studio locally and serve your dataset without copying files.
Workflow: (1) start venv → (2) start CORS-enabled static server → (3) generate tasks JSON → (4) import tasks into Label Studio.

---

## 1. Prepare & activate your virtual environment

Make sure `label-studio` and required scripts are installed in the environment you will use.

PowerShell (example with uv-created env):

```powershell
# activate the uv/venv environment (adjust path to your env)
& "C:\your-path-to-venv\myenv\Scripts\Activate.ps1"
```

Windows CMD (alternative):

```cmd
C:\path\to\venv\Scripts\activate.bat
```

Install Label Studio if needed:

```bash
pip install label-studio
```

---

## 2. Start the image server (CORS-enabled)

Run the provided `serve.py` script so Label Studio (running in the browser) can load images via HTTP. The server adds CORS headers and auto-restarts on crash.

Example:

```bash
python serve.py --serve_dir "C:/some-path/Ophthalmic_Scans/raw" --port 8000
```

Notes:

* Use forward slashes `E:/...` or escaped backslashes `E:\\...` when passing Windows paths on the CLI.
* The script normalizes paths, but double-check the folder exists.
* Stop the server cleanly with `Ctrl+C`.

Test a file in the browser:

```
http://localhost:8000/sub-.../ses-.../color/.../original_images/1.png
```

---

## 3. Generate Label Studio JSON tasks

Use `generate_label_studio_json.py` to create the tasks file. The script walks your dataset structure and reads per-image metadata JSONs.

Usage:

```bash
python generate_label_studio_json.py \
  --base_path "C:/some-path/Ophthalmic_Scans/raw" \
  --url_prefix "http://127.0.0.1:8000" \
  --output label_studio_tasks.json
```

Flags:

* `--base_path` — root of your dataset (where `sub-*` folders live). **Required.**
* `--url_prefix` — HTTP prefix used in `image` fields (default `http://localhost:8000`). Make sure it matches the server host/port.
* `--output` — output JSON filename.

Each task will look like:

```json
{
  "image": "http://127.0.0.1:8000/sub-1231556165.../ses-12345667/.../3.jpg",
  "folder_patient_id": "a1b2c3d4e5f6",
  "patient_id": "a1b2c3d4e5f6",
  "image_id": "12314565_Color_R_007.png",
  "diagnosis": "unknown",
  "date": "2022-12-15 00:00:00",
  "area": "lesion",
  "reference_eye": false,
  "image_type": "color",
  "laterality": "R"
}
```

---

## 4. Start Label Studio and import tasks

From the same activated venv run:

```bash
label-studio
```

In the Label Studio UI:

1. Create a new project.
2. Go to **Import Data**.
3. Upload the generated `label_studio_tasks.json` (or paste its content).
4. Make sure your Label config (interface) matches the `image` field type (Image tag, etc.).

Because images are served by `serve.py` with CORS headers, Label Studio will load them via HTTP.

---

## 5. Quick troubleshooting

* If images do not load but the URL works in a browser, check CORS: use the serve script provided (it sends `Access-Control-Allow-Origin: *`).
* If `generate_label_studio_json.py` creates an empty file:

  * confirm `--base_path` is correct (path typos are common).
  * prefer forward slashes: `E:/...`.
* If `serve.py` complains about directory not found, test the path in PowerShell:

  ```powershell
  Test-Path "C:\some-path\Ophthalmic_Scans\raw"
  ```
* If port is busy, change `--port` to another unused port (e.g. `9000`) and update `--url_prefix` accordingly.
* Stop the server with `Ctrl+C`. Label Studio can be stopped similarly.

---

## 6. Minimal command sequence (example)

```powershell
# 1. activate env
& "C:\your-path-to-venv\myenv\Scripts\Activate.ps1"

# 2. start server in same window (or another)
python serve.py --serve_dir "C:/.../raw" --port 8000

# 3. (in another terminal in same venv) generate JSON
python generate_tasks.py --base_path "C:/.../raw" --url_prefix "http://127.0.0.1:8000" --output tasks.json

# 4. start label studio in the activated venv
label-studio
# 5. Import tasks.json via the UI
# 6. use provided labelling interafce from labelling_interface.xml file 
```
