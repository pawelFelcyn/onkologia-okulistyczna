import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageTk


DEFAULT_SPLIT_DIR = Path("Ophthalmic_Scans") / "splits" / "tumor_and_fluid_segmentation_oct3"
REVIEW_FIELD = "segmentation_mask_review"
CANVAS_SIZE = (420, 420)
FLUID_LABEL = 0
TUMOR_LABEL = 1


@dataclass
class ReviewRecord:
    image_path: Path
    label_path: Path
    metadata_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review OCT test masks reconstructed from YOLO labels and save ok/not_ok into local metadata JSON files."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory containing test.csv.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start review from this zero-based record index.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def metadata_path_for_image(repo_root: Path, image_path: str) -> Path:
    image_rel = Path(image_path)
    metadata_rel = Path(*image_rel.parts[:-2]) / "metadata" / f"{image_rel.stem}.json"
    return repo_root / "Ophthalmic_Scans" / metadata_rel


def load_test_records(repo_root: Path, split_dir: Path) -> list[ReviewRecord]:
    csv_path = split_dir / "test.csv"
    if not csv_path.is_file():
        raise ValueError(f"Missing test.csv in split directory: {split_dir}")

    df = pd.read_csv(csv_path)
    records: list[ReviewRecord] = []
    for row in df.itertuples(index=False):
        image_path = repo_root / "Ophthalmic_Scans" / Path(row.image_path)
        label_path = repo_root / "Ophthalmic_Scans" / Path(row.label_path)
        metadata_path = metadata_path_for_image(repo_root, row.image_path)
        records.append(
            ReviewRecord(
                image_path=image_path,
                label_path=label_path,
                metadata_path=metadata_path,
            )
        )

    if not records:
        raise ValueError(f"No records found in test.csv: {csv_path}")
    return records


def read_metadata(metadata_path: Path) -> dict:
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def write_review_status(metadata_path: Path, status: str) -> None:
    metadata = read_metadata(metadata_path)
    metadata[REVIEW_FIELD] = status
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)
        file.write("\n")


def build_preview(image: Image.Image, canvas_size: tuple[int, int], grayscale: bool = False) -> Image.Image:
    preview_source = image.copy()
    if grayscale:
        preview_source = ImageOps.grayscale(preview_source)
    preview_source.thumbnail(canvas_size, Image.Resampling.LANCZOS)

    preview = Image.new("RGB", canvas_size, color="white")
    offset_x = (canvas_size[0] - preview_source.width) // 2
    offset_y = (canvas_size[1] - preview_source.height) // 2
    if preview_source.mode != "RGB":
        preview_source = preview_source.convert("RGB")
    preview.paste(preview_source, (offset_x, offset_y))
    return preview


def load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def masks_from_yolo_label(label_path: Path, width: int, height: int) -> tuple[Image.Image, Image.Image]:
    tumor_mask = np.zeros((height, width), dtype=np.uint8)
    fluid_mask = np.zeros((height, width), dtype=np.uint8)

    if not label_path.is_file():
        return Image.fromarray(tumor_mask), Image.fromarray(fluid_mask)

    with label_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            values = line.split()
            class_id = int(values[0])
            coordinates = list(map(float, values[1:]))
            if len(coordinates) < 6 or len(coordinates) % 2 != 0:
                continue

            points = np.array(coordinates, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= width
            points[:, 1] *= height
            polygon = np.round(points).astype(np.int32)

            if class_id == TUMOR_LABEL:
                cv2.fillPoly(tumor_mask, [polygon], 255)
            elif class_id == FLUID_LABEL:
                cv2.fillPoly(fluid_mask, [polygon], 255)

    return Image.fromarray(tumor_mask), Image.fromarray(fluid_mask)


class ReviewApp:
    def __init__(self, root: tk.Tk, records: list[ReviewRecord], start_index: int):
        self.root = root
        self.records = records
        self.index = max(0, min(start_index, len(records) - 1))
        self.tk_images: list[ImageTk.PhotoImage] = []

        self.root.title("OCT Test YOLO Label Review")
        self.root.geometry("1380x760")

        self.status_var = tk.StringVar()
        self.path_var = tk.StringVar()
        self.review_var = tk.StringVar()

        self._build_ui()
        self._bind_shortcuts()
        self.refresh()

    def _build_ui(self) -> None:
        top_frame = tk.Frame(self.root, padx=12, pady=12)
        top_frame.pack(fill="x")

        tk.Label(top_frame, textvariable=self.status_var, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(top_frame, textvariable=self.path_var, justify="left", anchor="w", wraplength=1320).pack(anchor="w", pady=(6, 0))
        tk.Label(top_frame, textvariable=self.review_var, justify="left", anchor="w").pack(anchor="w", pady=(6, 0))

        image_frame = tk.Frame(self.root, padx=12, pady=12)
        image_frame.pack(fill="both", expand=True)

        self.image_labels = []
        for title in ("OCT image", "Tumor mask from YOLO", "Fluid mask from YOLO"):
            panel = tk.Frame(image_frame)
            panel.pack(side="left", fill="both", expand=True, padx=8)
            tk.Label(panel, text=title, font=("Segoe UI", 11, "bold")).pack(pady=(0, 8))
            image_label = tk.Label(panel, bd=1, relief="solid", bg="white")
            image_label.pack(fill="both", expand=True)
            self.image_labels.append(image_label)

        button_frame = tk.Frame(self.root, padx=12, pady=12)
        button_frame.pack(fill="x")

        tk.Button(button_frame, text="Prev", width=12, command=self.prev_record).pack(side="left")
        tk.Button(button_frame, text="OK (G)", width=18, command=lambda: self.mark_and_advance("ok")).pack(side="left", padx=8)
        tk.Button(button_frame, text="Not OK (B)", width=18, command=lambda: self.mark_and_advance("not_ok")).pack(side="left")
        tk.Button(button_frame, text="Next", width=12, command=self.next_record).pack(side="left", padx=8)
        tk.Button(button_frame, text="Quit", width=12, command=self.root.destroy).pack(side="right")

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Left>", lambda _event: self.prev_record())
        self.root.bind("<Right>", lambda _event: self.next_record())
        self.root.bind("g", lambda _event: self.mark_and_advance("ok"))
        self.root.bind("G", lambda _event: self.mark_and_advance("ok"))
        self.root.bind("b", lambda _event: self.mark_and_advance("not_ok"))
        self.root.bind("B", lambda _event: self.mark_and_advance("not_ok"))

    def current_record(self) -> ReviewRecord:
        return self.records[self.index]

    def refresh(self) -> None:
        record = self.current_record()
        metadata = read_metadata(record.metadata_path)
        current_review = metadata.get(REVIEW_FIELD, "<not reviewed>")
        width, height = load_image_size(record.image_path)
        tumor_mask, fluid_mask = masks_from_yolo_label(record.label_path, width, height)

        self.status_var.set(f"Test record {self.index + 1}/{len(self.records)}")
        self.path_var.set(
            "\n".join(
                [
                    f"image: {record.image_path}",
                    f"label: {record.label_path}",
                    f"metadata: {record.metadata_path}",
                ]
            )
        )
        self.review_var.set(f"Current metadata field {REVIEW_FIELD}: {current_review}")

        with Image.open(record.image_path) as image:
            previews = [
                build_preview(image, CANVAS_SIZE),
                build_preview(tumor_mask, CANVAS_SIZE, grayscale=True),
                build_preview(fluid_mask, CANVAS_SIZE, grayscale=True),
            ]

        self.tk_images = [ImageTk.PhotoImage(image) for image in previews]
        for label, image in zip(self.image_labels, self.tk_images):
            label.configure(image=image)

    def mark_and_advance(self, status: str) -> None:
        record = self.current_record()
        try:
            write_review_status(record.metadata_path, status)
        except Exception as error:
            messagebox.showerror("Save failed", str(error))
            return

        if self.index < len(self.records) - 1:
            self.index += 1
        self.refresh()

    def prev_record(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.refresh()

    def next_record(self) -> None:
        if self.index < len(self.records) - 1:
            self.index += 1
            self.refresh()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    split_dir = args.split_dir
    if not split_dir.is_absolute():
        split_dir = repo_root / split_dir

    records = load_test_records(repo_root, split_dir)

    root = tk.Tk()
    ReviewApp(root, records, args.start_index)
    root.minsize(1180, 680)
    root.mainloop()


if __name__ == "__main__":
    main()