import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_SPLIT_DIR = Path("Ophthalmic_Scans") / "splits" / "tumor_and_fluid_segmentation_oct2"
DEFAULT_OUTPUT_SPLIT_DIR = Path("Ophthalmic_Scans") / "splits" / "tumor_and_fluid_segmentation_oct3"
REVIEW_FIELD = "segmentation_mask_review"
BAD_STATUS = "not_ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create tumor_and_fluid_segmentation_oct3 by removing bad raw samples and processed descendants of bad raw samples."
    )
    parser.add_argument(
        "--input-split-dir",
        type=Path,
        default=DEFAULT_INPUT_SPLIT_DIR,
        help="Directory containing source train.csv, val.csv and test.csv.",
    )
    parser.add_argument(
        "--output-split-dir",
        type=Path,
        default=DEFAULT_OUTPUT_SPLIT_DIR,
        help="Directory where filtered train.csv, val.csv and test.csv will be written.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_rel_path(path_str: str) -> Path:
    return Path(path_str.replace("\\", "/"))


def read_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metadata_path_from_image_rel(image_rel: Path) -> Path:
    return Path(*image_rel.parts[:-2]) / "metadata" / f"{image_rel.stem}.json"


def metadata_path_from_original_rel(original_rel: Path) -> Path:
    return Path(*original_rel.parts[:-2]) / "metadata" / f"{original_rel.stem}.json"


def source_metadata_rel_for_row(repo_root: Path, image_path: str) -> Path:
    image_rel = normalize_rel_path(image_path)
    if image_rel.parts[0] == "raw":
        return metadata_path_from_image_rel(image_rel)

    if image_rel.parts[0] != "processed":
        raise ValueError(f"Unsupported image root for row: {image_path}")

    processed_metadata_rel = metadata_path_from_image_rel(image_rel)
    processed_metadata_abs = repo_root / "Ophthalmic_Scans" / processed_metadata_rel
    processed_metadata = read_metadata(processed_metadata_abs)
    raw_source = processed_metadata.get("raw_source")
    if not raw_source:
        raise ValueError(f"Missing raw_source in processed metadata: {processed_metadata_abs}")

    raw_source_rel = normalize_rel_path(raw_source)
    if raw_source_rel.parts[0] != "raw":
        raise ValueError(f"Expected raw_source to point into raw, got: {raw_source}")

    return metadata_path_from_original_rel(raw_source_rel)


def load_review_status(metadata_abs: Path) -> str | None:
    if not metadata_abs.is_file():
        return None
    metadata = read_metadata(metadata_abs)
    return metadata.get(REVIEW_FIELD)


def filter_split_dataframe(repo_root: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    keep_rows = []
    stats = Counter()

    for row in df.itertuples(index=False):
        source_metadata_rel = source_metadata_rel_for_row(repo_root, row.image_path)
        source_metadata_abs = repo_root / "Ophthalmic_Scans" / source_metadata_rel
        review_status = load_review_status(source_metadata_abs)
        image_rel = normalize_rel_path(row.image_path)

        if review_status == BAD_STATUS:
            stats["removed_total"] += 1
            if image_rel.parts[0] == "raw":
                stats["removed_raw"] += 1
            else:
                stats["removed_processed"] += 1
            continue

        keep_rows.append(row._asdict())
        stats["kept_total"] += 1
        if image_rel.parts[0] == "raw":
            stats["kept_raw"] += 1
        else:
            stats["kept_processed"] += 1

    return pd.DataFrame(keep_rows, columns=df.columns), stats


def process_split_file(repo_root: Path, input_csv: Path, output_csv: Path) -> Counter:
    df = pd.read_csv(input_csv)
    filtered_df, stats = filter_split_dataframe(repo_root, df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)
    stats["input_rows"] = len(df)
    stats["output_rows"] = len(filtered_df)
    return stats


def print_summary(summary_by_split: dict[str, Counter]) -> None:
    print("Filtered split summary:")
    total = Counter()
    for split_name in ("train", "val", "test"):
        stats = summary_by_split.get(split_name, Counter())
        total.update(stats)
        print(f"\nSplit: {split_name}")
        print(f"  input rows: {stats['input_rows']}")
        print(f"  output rows: {stats['output_rows']}")
        print(f"  removed raw: {stats['removed_raw']}")
        print(f"  removed processed: {stats['removed_processed']}")
        print(f"  kept raw: {stats['kept_raw']}")
        print(f"  kept processed: {stats['kept_processed']}")

    print("\nTotal:")
    print(f"  input rows: {total['input_rows']}")
    print(f"  output rows: {total['output_rows']}")
    print(f"  removed raw: {total['removed_raw']}")
    print(f"  removed processed: {total['removed_processed']}")


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()

    input_split_dir = args.input_split_dir
    if not input_split_dir.is_absolute():
        input_split_dir = repo_root / input_split_dir

    output_split_dir = args.output_split_dir
    if not output_split_dir.is_absolute():
        output_split_dir = repo_root / output_split_dir

    summary_by_split: dict[str, Counter] = {}
    for split_name in ("train", "val", "test"):
        input_csv = input_split_dir / f"{split_name}.csv"
        if not input_csv.is_file():
            continue
        output_csv = output_split_dir / f"{split_name}.csv"
        summary_by_split[split_name] = process_split_file(repo_root, input_csv, output_csv)

    if not summary_by_split:
        raise ValueError(f"No split CSV files found in {input_split_dir}")

    print_summary(summary_by_split)


if __name__ == "__main__":
    main()