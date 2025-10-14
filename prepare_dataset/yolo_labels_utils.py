import os
from PIL import Image
import supervision as sv
import numpy as np
from tqdm import tqdm
import tempfile
import shutil
#import pytest
import cv2

TUMOR_LABEL = 1
FLUID_LABEL = 0

FLUID_MASK_COLOR_GRAYSCALE = 127
TUMOR_MASK_COLOR_GRAYSCALE = 255

def yolo2mask(image_path: str, yolo_label_path: str, output_mask_path: str, override: bool = False):
    """
    Converts YOLO polygon annotations into a grayscale segmentation mask.
    
    Parameters:
        image_path (str): Path to the input image (supported formats: PNG, JPG, JPEG).
        yolo_label_path (str): Path to the YOLO label file (.txt) containing polygon annotations.
        output_mask_path (str): Path where the generated mask image will be saved.
        override (bool, optional): If True, overwrites the output mask if it already exists. Defaults to False.
    
    Raises:
        Exception: If the input image path is invalid or unsupported.
        Exception: If the YOLO label file is missing or does not have a .txt extension.
        Exception: If the output mask file already exists and override=False.
    
    Description:
        This function reads an image and its corresponding YOLO polygon label file, 
        generates binary masks for each polygon, and combines them into a single 
        grayscale segmentation mask. Tumor regions are encoded with intensity 255, 
        while fluid regions are encoded with intensity 127. The mask is saved as a 
        new image at the specified output path.
    """
    if not image_path.endswith(".png") and not image_path.endswith(".jpg") and not image_path.endswith(".jpeg") and not os.path.isfile(image_path):
        raise Exception("Image must be in png, jpg, jpeg and it ust exist")
    if not os.path.isfile(yolo_label_path) and not yolo_label_path.endswith(".txt"):
        raise Exception("Yolo label must exista dn have .txt extension")
    if os.path.exists(output_mask_path) and not override:
        raise Exception("Output mask already exists. Set override=True to override")
    
    with Image.open(image_path) as img:
        width, height = img.size
    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()
    
    partial_masks: list[np.ndarray] = []
    labels: list[int] = []
    for _, polygon in enumerate(lines):
        polygon = polygon.split(' ')
        label = int(polygon[0])
        labels.append(label)
        polygon = list(map(float, polygon[1:-2]))
        p = []
        for i in range(0, len(polygon) - 1, 2):
            p.append([int(polygon[i] * width), int(polygon[i + 1] * height)])
        mask = sv.polygon_to_mask(np.array(p), (width, height))
        partial_masks.append(mask)
    
    final_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(partial_masks)):
        numeric_value = TUMOR_MASK_COLOR_GRAYSCALE if labels[i] == TUMOR_LABEL else FLUID_MASK_COLOR_GRAYSCALE
        for y in range(height):
            for x in range(width):
                if partial_masks[i][y][x] and final_mask[y][x] == 0:
                    final_mask[y][x] = numeric_value
    
    mask_image = Image.fromarray(final_mask)
    mask_image.save(output_mask_path)
    
def generate_masks_from_labels(images_dir: str, yolo_labels_dir: str, output_masks_dir: str, override: bool = False):
    """
    Generates segmentation masks from YOLO label files for all images in a directory.

    Parameters:
        images_dir (str): Path to the directory containing input images.
        yolo_labels_dir (str): Path to the directory containing YOLO label files (.txt).
        output_masks_dir (str): Path to the directory where generated mask images will be saved.
        override (bool, optional): If True, existing mask files will be overwritten. Defaults to False.

    Raises:
        Exception: If the images directory or YOLO labels directory does not exist.

    Description:
        This function iterates over all images in the specified images directory. For each image,
        it looks for a corresponding YOLO label file in the YOLO labels directory. If a label file exists,
        it calls `yolo2mask` to generate a grayscale segmentation mask and saves it in the output masks directory.
        If a mask for an image already exists and `override` is False, the mask is not regenerated.
        Images without corresponding YOLO labels are skipped, and a message is printed.

    Progress:
        The function can optionally display a progress bar and the name of the currently processed image.
    """
    if not os.path.isdir(images_dir):
        raise Exception("Images directory must exist")
    if not os.path.isdir(yolo_labels_dir):
        raise Exception("Yolo labels directory must exist")
    os.makedirs(output_masks_dir, exist_ok=True)
    
    images = os.listdir(images_dir)
    for image in tqdm(images, desc="Generating masks", unit="image"):
        extension = image.split(".")[-1]
        image_path = os.path.join(images_dir, image)
        yolo_label_path = os.path.join(yolo_labels_dir, image.replace(extension, "txt"))
        if not os.path.isfile(yolo_label_path):
            print(f'Skipping image {image} because it does not have a yolo label')
            continue
        output_mask_path = os.path.join(output_masks_dir, image)
        yolo2mask(image_path, yolo_label_path, output_mask_path, override)

def mask2yolo(mask_path: str, image_path: str, output_yolo_path: str, override: bool = False):
    """
    Converts a grayscale segmentation mask into YOLO polygon annotations.
    Normalizes coordinates using the original image dimensions.
    """
    if not os.path.isfile(mask_path):
        raise Exception("Mask file does not exist.")
    if not os.path.isfile(image_path):
        raise Exception("Original image file does not exist.")
    if os.path.exists(output_yolo_path) and not override:
        raise Exception("Output YOLO label already exists. Set override=True to override.")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise Exception(f"Could not read mask file at {mask_path}")
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    yolo_lines = []
    class_map = {
        TUMOR_LABEL: TUMOR_MASK_COLOR_GRAYSCALE,
        FLUID_LABEL: FLUID_MASK_COLOR_GRAYSCALE
    }

    for label, grayscale_value in class_map.items():
        class_mask = np.uint8(mask == grayscale_value) * 255
        contours, _ = cv2.findContours(class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 3:
                continue
            norm_contour = contour.astype(np.float32)
            norm_contour[:, 0, 0] /= img_width
            norm_contour[:, 0, 1] /= img_height
            flat_contour = norm_contour.flatten()
            yolo_line = f"{label} " + " ".join(map(str, flat_contour))
            yolo_lines.append(yolo_line)

    with open(output_yolo_path, 'w') as f:
        f.write("\n".join(yolo_lines))

def generate_labels_from_masks(masks_dir: str, images_dir: str, output_labels_dir: str, override: bool = False):
    """
    Generates YOLO label files from segmentation masks for all images in a directory.
    Matches mask and image by filename (ignoring extension).
    """
    if not os.path.isdir(masks_dir):
        raise Exception("Masks directory must exist")
    if not os.path.isdir(images_dir):
        raise Exception("Images directory must exist")
    os.makedirs(output_labels_dir, exist_ok=True)

    mask_files = os.listdir(masks_dir)
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir)}

    for mask_name in tqdm(mask_files, desc="Generating YOLO labels", unit="mask"):
        mask_base, _ = os.path.splitext(mask_name)
        if mask_base not in image_files:
            print(f"Skipping mask {mask_name}: no matching image in {images_dir}")
            continue
        mask_path = os.path.join(masks_dir, mask_name)
        image_path = os.path.join(images_dir, image_files[mask_base])
        output_yolo_path = os.path.join(output_labels_dir, mask_base + ".txt")
        mask2yolo(mask_path, image_path, output_yolo_path, override)

def visualize_yolo_labels_on_image(image_path: str, yolo_label_path: str, output_path: str, color_map=None):
    """
    Draws YOLO polygon labels on the original image and saves the visualization.
    Args:
        image_path: Path to the original image.
        yolo_label_path: Path to the YOLO label file (.txt).
        output_path: Path to save the visualization image.
        color_map: Optional dict {label: (B, G, R)} for custom colors.
    """
    if color_map is None:
        color_map = {0: (0, 255, 255), 1: (0, 0, 255)}  # fluid: yellow, tumor: red
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue  # skip invalid lines
        label = int(parts[0])
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue
        pts = np.array([
            [int(coords[i] * width), int(coords[i + 1] * height)]
            for i in range(0, len(coords), 2)
        ], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=color_map.get(label, (0,255,0)), thickness=2)
        cv2.fillPoly(image, [pts], color=color_map.get(label, (0,255,0)))
    cv2.imwrite(output_path, image)

def batch_visualize_yolo_labels(images_dir: str, labels_dir: str, output_dir: str):
    """
    Batch visualizes YOLO labels for all images in a directory.
    Args:
        images_dir: Directory with images.
        labels_dir: Directory with YOLO label files.
        output_dir: Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir)}
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    for label_file in tqdm(label_files, desc="Visualizing YOLO labels", unit="label"):
        base = os.path.splitext(label_file)[0]
        if base not in image_files:
            print(f"Skipping {label_file}: no matching image.")
            continue
        image_path = os.path.join(images_dir, image_files[base])
        label_path = os.path.join(labels_dir, label_file)
        output_path = os.path.join(output_dir, base + "_vis.png")
        try:
            visualize_yolo_labels_on_image(image_path, label_path, output_path)
        except Exception as e:
            print(f"Error visualizing {label_file}: {e}")

def test_yolo_mask_consistency(image_path, yolo_label_path, temp_dir=None):
    """
    Test: YOLO label -> mask -> YOLO label, and mask -> YOLO label -> mask.
    Checks pixel-wise and label-wise consistency.
    """
    import filecmp
    import hashlib
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    base = os.path.splitext(os.path.basename(image_path))[0]
    mask_from_label = os.path.join(temp_dir, f"{base}_mask_from_label.png")
    yolo_from_mask = os.path.join(temp_dir, f"{base}_yolo_from_mask.txt")
    mask_from_yolo_from_mask = os.path.join(temp_dir, f"{base}_mask_from_yolo_from_mask.png")

    # 1. YOLO label -> mask
    yolo2mask(image_path, yolo_label_path, mask_from_label, override=True)
    # 2. mask -> YOLO label
    mask2yolo(mask_from_label, image_path, yolo_from_mask, override=True)
    # 3. YOLO label (z maski) -> mask
    yolo2mask(image_path, yolo_from_mask, mask_from_yolo_from_mask, override=True)

    # Compare YOLO label files (original vs. z maski)
    with open(yolo_label_path, 'r') as f1, open(yolo_from_mask, 'r') as f2:
        label1 = f1.read().strip()
        label2 = f2.read().strip()
    if label1 == label2:
        print("YOLO label files are identical after round-trip conversion.")
    else:
        print("YOLO label files differ after round-trip conversion.")

    # Compare masks pixel-wise
    mask1 = cv2.imread(mask_from_label, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask_from_yolo_from_mask, cv2.IMREAD_GRAYSCALE)
    if mask1 is None or mask2 is None:
        print("Could not read generated masks for comparison.")
        return
    if mask1.shape != mask2.shape:
        print(f"Mask shapes differ: {mask1.shape} vs {mask2.shape}")
    else:
        diff = np.sum(mask1 != mask2)
        if diff == 0:
            print("Masks are identical pixel-wise after round-trip conversion.")
        else:
            print(f"Masks differ in {diff} pixels after round-trip conversion.")
    print(f"Temporary files: {mask_from_label}, {yolo_from_mask}, {mask_from_yolo_from_mask}")
    # Optionally, remove temp files
    #os.remove(mask_from_label)
    #os.remove(yolo_from_mask)
    #os.remove(mask_from_yolo_from_mask)

