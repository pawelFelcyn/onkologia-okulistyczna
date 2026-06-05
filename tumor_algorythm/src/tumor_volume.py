import os
import sys
import glob
import argparse

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


IMAGE_WIDTH = 1024                       
IMAGE_HEIGHT = 295                       
AXIS_OF_ROTATION_X = IMAGE_WIDTH / 2.0   
PIXEL_SIZE_UM = 1.0                      
PIXEL_SIZE_MM = 0.001                    
DELTA_THETA_DEG = 15.0                   
DELTA_THETA_RAD = np.pi / 12.0           

NUM_SCANS = 12                           
NUM_INTEGRATION_POINTS = 25    


_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def load_yolo_polygon(txt_path):
    polygons = []

    if os.path.getsize(txt_path) == 0:
        return Polygon()

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            coords = list(map(float, parts[1:]))
            points = []
            for i in range(0, len(coords) - 1, 2):
                x_px = coords[i] * IMAGE_WIDTH
                y_px = coords[i + 1] * IMAGE_HEIGHT
                points.append((x_px, y_px))

            if len(points) < 3:
                continue

            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if (not poly.is_empty) and poly.area > 0.0:
                polygons.append(poly)

    if not polygons:
        return Polygon()
    if len(polygons) == 1:
        return polygons[0]
    return unary_union(polygons)


def split_along_axis(polygon):
    if polygon.is_empty:
        return Polygon(), Polygon()

    left_clip = box(-1.0, -1.0, AXIS_OF_ROTATION_X, IMAGE_HEIGHT + 1.0)
    right_clip = box(AXIS_OF_ROTATION_X, -1.0, IMAGE_WIDTH + 1.0, IMAGE_HEIGHT + 1.0)

    left_half = polygon.intersection(left_clip)
    right_half = polygon.intersection(right_clip)
    return left_half, right_half


def area_and_centroid_x(geom):
    if geom is None or geom.is_empty:
        return 0.0, AXIS_OF_ROTATION_X

    area = geom.area
    if area <= 0.0:
        return 0.0, AXIS_OF_ROTATION_X

    centroid = geom.centroid
    if centroid.is_empty:
        return 0.0, AXIS_OF_ROTATION_X

    return area, centroid.x


def list_label_files(labels_dir):

    files = glob.glob(os.path.join(labels_dir, "*.txt"))

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return (0, int(stem))      
        except ValueError:
            return (1, stem)           

    return sorted(files, key=sort_key)


def compute_tumor_volume(labels_dir, verbose=True):
    files = list_label_files(labels_dir)
    if not files:
        raise FileNotFoundError(f"No .txt files in folder: {labels_dir}")

    if verbose:
        print("=" * 78)
        print("EYE TUMOR VOLUMETRY - radial OCT scans, Pappus-Guldin + trapezoid")
        print("=" * 78)
        print(f"\nLabels folder : {labels_dir}")
        print(f"Detected {len(files)} input files in scan order:")
        for idx, path in enumerate(files, start=1):
            print(f"  Scan {idx:2d}: {os.path.basename(path)}")

    if len(files) != NUM_SCANS:
        print(
            f"\n[WARNING] Expected {NUM_SCANS} scans, found "
            f"{len(files)}. Result may be incorrect."
        )

    left_halves = []
    right_halves = []
    for path in files:
        polygon = load_yolo_polygon(path)
        left_half, right_half = split_along_axis(polygon)
        left_halves.append(left_half)
        right_halves.append(right_half)

    geometries = []
    geometries.extend(left_halves)        
    geometries.extend(right_halves)       
    geometries.append(left_halves[0])       

    assert len(geometries) == NUM_INTEGRATION_POINTS, (
        f"Integration vector has {len(geometries)} elements, "
        f"expected {NUM_INTEGRATION_POINTS}."
    )

    rows = []
    moments = np.zeros(NUM_INTEGRATION_POINTS, dtype=np.float64)
    for k, geom in enumerate(geometries):
        theta_deg = k * DELTA_THETA_DEG

        area_px, centroid_x = area_and_centroid_x(geom)
        r_bar_px = abs(centroid_x - AXIS_OF_ROTATION_X)

        area_mm2 = area_px * (PIXEL_SIZE_MM ** 2)
        r_bar_mm = r_bar_px * PIXEL_SIZE_MM

        moment = area_mm2 * r_bar_mm
        moments[k] = moment

        rows.append(
            {
                "k": k,
                "theta_deg": theta_deg,
                "area_mm2": area_mm2,
                "r_bar_mm": r_bar_mm,
                "moment": moment,
            }
        )

    volume_mm3 = float(_trapezoid(moments, dx=DELTA_THETA_RAD))

    if verbose:
        _print_report(rows, volume_mm3)

    return {
        "files": files,
        "rows": rows,
        "M_array": moments,
        "volume_mm3": volume_mm3,
    }



def _print_report(rows, volume_mm3):
    print("\n" + "-" * 78)
    print("Static moments for 25 theta angles")
    print("-" * 78)
    header = (
        f"{'#':>3} | {'theta [deg]':>11} | {'Area [mm^2]':>14} | "
        f"{'Centroid r [mm]':>16} | {'Moment [mm^3]':>15}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['k']:>3} | {row['theta_deg']:>11.1f} | "
            f"{row['area_mm2']:>14.6e} | {row['r_bar_mm']:>16.6f} | "
            f"{row['moment']:>15.6e}"
        )

    print("-" * 78)
    print(f"Total Tumor Volume: {volume_mm3:.6e} mm^3")
    print(f"                    {volume_mm3:.8f} mm^3")
    print("=" * 78)


def default_labels_dir():
    """Default: data/segmentation_masks/patient_1/yolo_labels."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)        
    return os.path.join(
        project_root, "data", "segmentation_masks", "patient_1", "yolo_labels"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Eye tumor volumetry from radial OCT scans, YOLO masks.",
    )
    parser.add_argument(
        "labels_dir",
        nargs="?",
        default=default_labels_dir(),
        help="Folder with 12 YOLO .txt label files. "
        "Default: data/segmentation_masks/patient_1/yolo_labels",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.labels_dir):
        print(f"[ERROR] Folder does not exist: {args.labels_dir}", file=sys.stderr)
        return 1

    compute_tumor_volume(args.labels_dir, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
