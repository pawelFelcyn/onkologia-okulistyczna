import os
import sys
import math
import argparse

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tumor_volume import (  # noqa: E402
    AXIS_OF_ROTATION_X,
    PIXEL_SIZE_MM,
    DELTA_THETA_DEG,
    DELTA_THETA_RAD,
    list_label_files,
    load_yolo_polygon,
    split_along_axis,
    compute_tumor_volume,
    default_labels_dir,
)

SIMPLIFY_TOLERANCE_PX = 1.0

def build_radial_slices(labels_dir):
    files = list_label_files(labels_dir)
    if not files:
        raise FileNotFoundError(f"Brak plików .txt w folderze: {labels_dir}")

    left_halves, right_halves = [], []
    for path in files:
        polygon = load_yolo_polygon(path)
        left_half, right_half = split_along_axis(polygon)
        left_halves.append(left_half)
        right_halves.append(right_half)

    slices = []
    for i, geom in enumerate(left_halves):
        slices.append((math.radians(i * DELTA_THETA_DEG), geom))
    for i, geom in enumerate(right_halves):
        slices.append((math.radians(180.0 + i * DELTA_THETA_DEG), geom))

    enriched = []
    for theta, geom in slices:
        area_mm2 = geom.area * (PIXEL_SIZE_MM ** 2)
        centroid_xyz = None
        if (not geom.is_empty) and geom.area > 0.0:
            c = geom.centroid
            r_mm = abs(c.x - AXIS_OF_ROTATION_X) * PIXEL_SIZE_MM
            h_mm = c.y * PIXEL_SIZE_MM
            centroid_xyz = (r_mm * math.cos(theta), r_mm * math.sin(theta), h_mm)
        enriched.append((theta, geom, area_mm2, centroid_xyz))

    return enriched


def iter_polygons(geom):
    if geom is None or geom.is_empty:
        return
    gtype = geom.geom_type
    if gtype == "Polygon":
        yield geom
    elif gtype == "MultiPolygon":
        for sub in geom.geoms:
            yield sub
    elif gtype == "GeometryCollection":
        for sub in geom.geoms:
            if sub.geom_type in ("Polygon", "MultiPolygon"):
                yield from iter_polygons(sub)


def ring_to_3d(coords_px, theta):
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    pts = np.empty((len(coords_px), 3), dtype=np.float64)
    for i, (x_px, y_px) in enumerate(coords_px):
        r_mm = abs(x_px - AXIS_OF_ROTATION_X) * PIXEL_SIZE_MM
        h_mm = y_px * PIXEL_SIZE_MM
        pts[i, 0] = r_mm * cos_t
        pts[i, 1] = r_mm * sin_t
        pts[i, 2] = h_mm
    return pts


def polygon_exterior_px(poly, simplify=False):
    if simplify:
        poly = poly.simplify(SIMPLIFY_TOLERANCE_PX, preserve_topology=True)
        if poly.is_empty:
            return None
    return list(poly.exterior.coords)


def plot_slices(ax, slices, norm, cmap):
    faces, colors = [], []
    for theta, geom, area_mm2, _ in slices:
        face_color = cmap(norm(area_mm2))
        for poly in iter_polygons(geom):
            ext = polygon_exterior_px(poly, simplify=False)
            if ext is None or len(ext) < 3:
                continue
            faces.append(ring_to_3d(ext, theta))
            colors.append(face_color)

    if faces:
        coll = Poly3DCollection(faces, alpha=0.55)
        coll.set_facecolor(colors)
        coll.set_edgecolor((0.15, 0.15, 0.15, 0.45))
        coll.set_linewidth(0.3)
        ax.add_collection3d(coll)

    cent = np.array([c for *_, c in slices if c is not None])
    if len(cent):
        ax.scatter(cent[:, 0], cent[:, 1], cent[:, 2],
                   color="crimson", s=14, depthshade=True,
                   label="środki ciężkości")

    ax.set_title("Przekroje radialne")


def plot_solid(ax, slices, norm, cmap):
    half = DELTA_THETA_RAD / 2.0
    faces, colors = [], []

    for theta, geom, area_mm2, _ in slices:
        face_color = cmap(norm(area_mm2))
        for poly in iter_polygons(geom):
            ext = polygon_exterior_px(poly, simplify=True)
            if ext is None or len(ext) < 3:
                continue

            cap_minus = ring_to_3d(ext, theta - half)
            cap_plus = ring_to_3d(ext, theta + half)

            faces.append(cap_minus)
            colors.append(face_color)
            faces.append(cap_plus)
            colors.append(face_color)

            for i in range(len(ext) - 1):
                quad = np.array([cap_minus[i], cap_minus[i + 1],
                                 cap_plus[i + 1], cap_plus[i]])
                faces.append(quad)
                colors.append(face_color)

    if faces:
        coll = Poly3DCollection(faces, alpha=0.9)
        coll.set_facecolor(colors)
        coll.set_edgecolor((0.2, 0.2, 0.2, 0.15))
        coll.set_linewidth(0.1)
        ax.add_collection3d(coll)

    ax.set_title("Rekonstrukcja bryły (sweep przekrojów co 15°)")


def decorate_axes(ax, slices):
    all_pts = []
    for theta, geom, _, _ in slices:
        for poly in iter_polygons(geom):
            ext = polygon_exterior_px(poly, simplify=False)
            if ext:
                all_pts.append(ring_to_3d(ext, theta))
    if not all_pts:
        return
    pts = np.vstack(all_pts)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)

    z0, z1 = mins[2], maxs[2]
    ax.plot([0, 0], [0, 0], [z0, z1], color="black", lw=1.2,
            linestyle="--", label="oś obrotu")

    span = (maxs - mins)
    span[span == 0] = 1.0
    ax.set_box_aspect(span)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("głębokość (mm)")
    ax.invert_zaxis()          
    ax.view_init(elev=28, azim=-55)



def visualize(labels_dir, mode="both", out_path=None, show=True):
    result = compute_tumor_volume(labels_dir, verbose=True)
    volume_mm3 = result["volume_mm3"]

    slices = build_radial_slices(labels_dir)

    areas = [a for _, _, a, _ in slices if a > 0]
    vmin = min(areas) if areas else 0.0
    vmax = max(areas) if areas else 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    if mode == "both":
        fig = plt.figure(figsize=(18, 9))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        plot_slices(ax1, slices, norm, cmap)
        decorate_axes(ax1, slices)
        plot_solid(ax2, slices, norm, cmap)
        decorate_axes(ax2, slices)
        ax1.legend(loc="upper right", fontsize=8)
        axes_for_bar = [ax1, ax2]
    else:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        if mode == "slices":
            plot_slices(ax, slices, norm, cmap)
            ax.legend(loc="upper right", fontsize=8)
        else:  # solid
            plot_solid(ax, slices, norm, cmap)
        decorate_axes(ax, slices)
        axes_for_bar = [ax]

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axes_for_bar, shrink=0.6, pad=0.08)
    cbar.set_label("Pole przekroju [mm$^2$]")

    fig.suptitle(
        f"Wolumetria guza oka (OCT radialny)   |   "
        f"Objętość = {volume_mm3:.4e} mm$^3$",
        fontsize=13, fontweight="bold",
    )

    if out_path is None:
        out_path = default_out_path(labels_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nZapisano wizualizację 3D: {out_path}")

    if show:
        try:
            plt.show()
        except Exception as exc:  # brak środowiska graficznego
            print(f"Nie udało się otworzyć okna ({exc}). "
                  f"Obraz zapisano do pliku.")
    plt.close(fig)
    return out_path


def default_out_path(labels_dir):
    """Plik wyjściowy: <projekt>/output/tumor_3d_<pacjent>.png."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)               # tumor_algorythm
    patient = os.path.basename(os.path.dirname(os.path.normpath(labels_dir)))
    if not patient:
        patient = "tumor"
    return os.path.join(project_root, "output", f"tumor_3d_{patient}.png")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Wizualizacja 3D guza oka z radialnych skanów OCT (YOLO).",
    )
    parser.add_argument(
        "labels_dir", nargs="?", default=default_labels_dir(),
        help="Folder z 12 plikami etykiet YOLO (.txt). "
        "Domyślnie: data/segmentation_masks/patient_1/yolo_labels",
    )
    parser.add_argument(
        "--mode", choices=["both", "slices", "solid"], default="both",
        help="Tryb wizualizacji (domyślnie: both).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Ścieżka pliku PNG do zapisu (domyślnie: output/tumor_3d_<pacjent>.png).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Nie otwieraj okna interaktywnego (tylko zapis do pliku).",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.labels_dir):
        print(f"[BŁĄD] Folder nie istnieje: {args.labels_dir}", file=sys.stderr)
        return 1

    if args.no_show:
        matplotlib.use("Agg")

    visualize(args.labels_dir, mode=args.mode, out_path=args.out,
              show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
