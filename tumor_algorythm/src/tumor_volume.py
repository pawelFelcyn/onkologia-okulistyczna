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
    """
    Wczytuje plik etykiety YOLO i zwraca geometrię shapely w przestrzeni
    pikselowej obrazu (denormalizacja współrzędnych).

    Plik może zawierać wiele linii (wiele konturów) - są one łączone w jedną
    geometrię (unia). Puste pliki (brak guza) zwracają pusty wielokąt.
    """
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
    """
    Tnie wielokąt wzdłuż wirtualnej linii X = AXIS_OF_ROTATION_X (512.0) i
    zwraca krotkę (lewa_polowa, prawa_polowa).

    Lewa  połowa: X <= 512.0
    Prawa połowa: X >= 512.0

    Jeśli guz leży w całości po jednej stronie osi, druga połowa będzie pustą
    geometrią (pole = 0) - jest to obsługiwane bezpiecznie.
    """
    if polygon.is_empty:
        return Polygon(), Polygon()

    # Prostokąty cięcia z zapasem, obejmujące całą przestrzeń obrazu po danej
    # stronie osi obrotu.
    left_clip = box(-1.0, -1.0, AXIS_OF_ROTATION_X, IMAGE_HEIGHT + 1.0)
    right_clip = box(AXIS_OF_ROTATION_X, -1.0, IMAGE_WIDTH + 1.0, IMAGE_HEIGHT + 1.0)

    left_half = polygon.intersection(left_clip)
    right_half = polygon.intersection(right_clip)
    return left_half, right_half


def area_and_centroid_x(geom):
    """
    Zwraca (pole_w_px^2, centroid_x). Dla pustej / zerowej geometrii zwraca
    (0.0, AXIS_OF_ROTATION_X), aby r_bar = 0 i moment = 0.

    Geometria może być Polygon, MultiPolygon lub GeometryCollection (gdy
    cięcie utworzy zdegenerowane fragmenty) - liczone jest tylko pole części
    powierzchniowych, a centroid jest średnią ważoną polem.
    """
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
    """
    Zwraca posortowaną listę plików .txt z folderu etykiet.

    Sortowanie jest NUMERYCZNE wg nazwy (1, 2, ..., 12), a nie leksykalne
    (które dałoby 1, 10, 11, 12, 2, ...), co jest krytyczne dla poprawnego
    przypisania skanu do kąta.
    """
    files = glob.glob(os.path.join(labels_dir, "*.txt"))

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return (0, int(stem))      # pliki numeryczne najpierw, rosnąco
        except ValueError:
            return (1, stem)           # ewentualne nienumeryczne na końcu

    return sorted(files, key=sort_key)


def compute_tumor_volume(labels_dir, verbose=True):
    """
    Wykonuje pełny algorytm wolumetrii i zwraca słownik z wynikami:
        {
            "files": [...],            # rozpoznane pliki wejściowe
            "rows": [...],             # 25 wierszy (theta, area, r, moment)
            "M_array": np.ndarray,     # wektor momentów statycznych (25)
            "volume_mm3": float,       # objętość guza w mm^3
        }
    """
    files = list_label_files(labels_dir)
    if not files:
        raise FileNotFoundError(f"Brak plików .txt w folderze: {labels_dir}")

    if verbose:
        print("=" * 78)
        print("WOLUMETRIA GUZA OKA - radialne skany OCT (Pappus-Guldin + trapez)")
        print("=" * 78)
        print(f"\nFolder etykiet : {labels_dir}")
        print(f"Rozpoznano {len(files)} plików wejściowych (kolejność skanów):")
        for idx, path in enumerate(files, start=1):
            print(f"  Skan {idx:2d}: {os.path.basename(path)}")

    if len(files) != NUM_SCANS:
        print(
            f"\n[OSTRZEŻENIE] Oczekiwano {NUM_SCANS} skanów, znaleziono "
            f"{len(files)}. Wynik może być niepoprawny."
        )

    left_halves = []
    right_halves = []
    for path in files:
        polygon = load_yolo_polygon(path)
        left_half, right_half = split_along_axis(polygon)
        left_halves.append(left_half)
        right_halves.append(right_half)

    geometries = []
    geometries.extend(left_halves)          # indeksy 0..11
    geometries.extend(right_halves)         # indeksy 12..23
    geometries.append(left_halves[0])       # indeks 24 (360° == 0°)

    assert len(geometries) == NUM_INTEGRATION_POINTS, (
        f"Wektor całkowania ma {len(geometries)} elementów, "
        f"oczekiwano {NUM_INTEGRATION_POINTS}."
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
    print("Momenty statyczne dla 25 kątów theta")
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
    print(f"Całkowita Objętość Guza: {volume_mm3:.6e} mm^3")
    print(f"                       ( {volume_mm3:.8f} mm^3 )")
    print("=" * 78)


def default_labels_dir():
    """Domyślnie: data/segmentation_masks/patient_1/yolo_labels."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)          # tumor_algorythm
    return os.path.join(
        project_root, "data", "segmentation_masks", "patient_1", "yolo_labels"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Wolumetria guza oka na podstawie radialnych skanów OCT "
        "(maski YOLO).",
    )
    parser.add_argument(
        "labels_dir",
        nargs="?",
        default=default_labels_dir(),
        help="Folder z 12 plikami etykiet YOLO (.txt). "
        "Domyślnie: data/segmentation_masks/patient_1/yolo_labels",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.labels_dir):
        print(f"[BŁĄD] Folder nie istnieje: {args.labels_dir}", file=sys.stderr)
        return 1

    compute_tumor_volume(args.labels_dir, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
