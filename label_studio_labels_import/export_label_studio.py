import time
import requests
import json
from pathlib import Path

def filter_by_image_type(path: str, type: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = [item for item in data if item["data"]["image_type"] == type]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

def export_label_studio_data(
    base_url: str,
    api_key: str,
    project_id: int,
    output_file: str = "export.json",
    export_type: str | None = None,
    payload: dict | None = None,
    poll_interval: int = 2,
    timeout: int = 300
) -> str:
    """
    Create an export in Label Studio with optional filtering and format conversion,
    wait until it's completed, and download the result.

    Args:
        filters (dict): Label Studio filter definition (same as UI filters)
    """

    headers = {
        "Authorization": f"Token {api_key}"
    }

    # Step 1: Create export snapshot (with optional filters)
    create_url = f"{base_url}/api/projects/{project_id}/exports/"

    r = requests.post(create_url, headers=headers, json=payload)
    r.raise_for_status()

    export_id = r.json()["id"]
    print(f"[+] Export created: {export_id}")

    # Step 2: Wait for snapshot
    status_url = f"{base_url}/api/projects/{project_id}/exports/{export_id}"
    start_time = time.time()

    while True:
        r = requests.get(status_url, headers=headers)
        r.raise_for_status()

        status = r.json()["status"]
        print(f"[...] status: {status}")

        if status == "completed":
            break
        if status == "failed":
            raise RuntimeError("Export failed")

        if time.time() - start_time > timeout:
            raise TimeoutError("Export timed out")

        time.sleep(poll_interval)

    # Step 3: Convert format (optional)
    if export_type:
        convert_url = f"{base_url}/api/projects/{project_id}/exports/{export_id}/convert"

        r = requests.post(
            convert_url,
            headers=headers,
            json={"export_type": export_type}
        )
        r.raise_for_status()

        print(f"[+] Conversion started: {export_type}")

        start_time = time.time()
        while True:
            r = requests.get(status_url, headers=headers)
            r.raise_for_status()

            status = r.json()["status"]
            print(f"[...] conversion status: {status}")

            if status == "completed":
                export_id = r.json()["id"]
                break
            if status == "failed":
                raise RuntimeError("Conversion failed")

            if time.time() - start_time > timeout:
                raise TimeoutError("Conversion timed out")

            time.sleep(poll_interval)
            
    download_url = f"{base_url}/api/projects/{project_id}/exports/{export_id}/download"
    download_params = {}
    if export_type:
        download_params["export_type"] = export_type
    r = requests.get(download_url, headers=headers, params=download_params)
    r.raise_for_status()
    with open(output_file, "wb") as f:
        f.write(r.content)
    print(f"[+] Saved to {output_file}")
    return export_id

def convert_label_studio_export(
    base_url: str,
    api_key: str,
    project_id: int,
    export_id: int,
    export_type: str,
    output_path: str,
    poll_interval: int = 2,
    timeout: int = 300
) -> None:
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    # 🔹 1. Start conversion
    convert_url = f"{base_url}/api/projects/{project_id}/exports/{export_id}/convert"

    response = requests.post(
        convert_url,
        headers=headers,
        json={
            "export_type": export_type,
            "download_resources": True  # opcjonalnie
        }
    )
    response.raise_for_status()

    data = response.json()
    converted_format_id = data.get("converted_format")

    print(f"[INFO] Conversion started: format={export_type}, id={converted_format_id}")

    # 🔹 2. Poll status
    start_time = time.time()

    while True:
        status_url = f"{base_url}/api/projects/{project_id}/exports/{export_id}"
        r = requests.get(status_url, headers=headers)
        r.raise_for_status()

        export_data = r.json()

        converted_formats = export_data.get("converted_formats", [])

        current = None
        for cf in converted_formats:
            if (
                (converted_format_id and cf.get("id") == converted_format_id)
                or cf.get("export_type") == export_type
            ):
                current = cf
                break

        status = current.get("status") if current else None

        print(f"[INFO] Status: {status}")

        if status in ("completed", "failed"):
            break

        if time.time() - start_time > timeout:
            raise TimeoutError("Conversion timed out")

        time.sleep(poll_interval)

    if status == "failed":
        raise RuntimeError("Conversion failed")

    # 🔹 3. Download file
    download_url = f"{base_url}/api/projects/{project_id}/exports/{export_id}/download"

    params = {
        "exportType": export_type
    }

    with requests.get(download_url, headers=headers, params=params, stream=True) as r:
        r.raise_for_status()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"[INFO] File downloaded to: {output_path}")
    