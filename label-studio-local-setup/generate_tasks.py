import json
import argparse
from pathlib import Path


def generate_tasks(base_path: Path, url_prefix: str, output_json: Path):
    """
    Generate Label Studio tasks from folder structure and metadata files.
    """

    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    tasks = []

    print(f"Scanning base directory: {base_path}")

    for patient_folder in base_path.glob("sub-*"):
        patient_id = patient_folder.name.replace("sub-", "")

        for session_folder in patient_folder.glob("ses-*"):
            color_path = session_folder / "color"

            for area_folder in color_path.glob("*"):
                for eye_folder in area_folder.glob("*"):
                    image_folder = eye_folder / "original_images"
                    metadata_folder = eye_folder / "metadata"

                    for img_file in image_folder.glob("*.*"):
                        metadata_file = metadata_folder / f"{img_file.stem}.json"

                        if not metadata_file.exists():
                            print(f"Missing metadata for image: {img_file}")
                            continue

                        with open(metadata_file, encoding="utf-8") as f:
                            meta = json.load(f)

                        image_rel_path = img_file.relative_to(base_path).as_posix()

                        task = {
                            "image": f"{url_prefix}/{image_rel_path}",
                            "file_name": img_file.name,
                            "patient_id": meta.get("patient_id"),
                            "image_id": meta.get("image_id"),
                            "diagnosis": meta.get("diagnosis"),
                            "date": meta.get("date"),
                            "area": meta.get("area"),
                            "reference_eye": meta.get("reference_eye"),
                            "image_type": meta.get("image_type"),
                            "laterality": meta.get("laterality"),
                        }

                        tasks.append(task)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4, ensure_ascii=False)

    print(f"\nSuccessfully saved {len(tasks)} tasks to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Label Studio JSON task file from a structured dataset."
    )

    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the dataset root folder (e.g. ../Ophthalmic_Scans/raw)"
    )

    parser.add_argument(
        "--url_prefix",
        type=str,
        default="http://localhost:8000",
        help="URL prefix for image serving (default: http://localhost:8000)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="label_studio_tasks.json",
        help="Output JSON filename"
    )

    args = parser.parse_args()

    generate_tasks(
        base_path=Path(args.base_path),
        url_prefix=args.url_prefix.rstrip("/"),
        output_json=Path(args.output)
    )
