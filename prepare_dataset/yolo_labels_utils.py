import os
from PIL import Image
import supervision as sv
import numpy as np
from tqdm import tqdm
import cv2

TUMOR_LABEL = 1
FLUID_LABEL = 0

TUMOR = 'tumor'
FLUID = 'fluid'

FLUID_MASK_COLOR_GRAYSCALE = 127
TUMOR_MASK_COLOR_GRAYSCALE = 255

def mask2yolo_separate_inputs(
tumor_mask_path: str,
fluid_mask_path: str,
yolo_label_output_path: str,
override: bool = False,
):
    if os.path.exists(yolo_label_output_path) and not override:
        raise Exception("Output label already exists. Set override=True to override")
    tumor_mask = np.array(Image.open(tumor_mask_path).convert("L"))
    fluid_mask = np.array(Image.open(fluid_mask_path).convert("L"))
    height, width = tumor_mask.shape
    lines: list[str] = []

    def process_mask(mask: np.ndarray, label: int):
        bin_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            cnt = cnt.squeeze(axis=1)
            coords = []
            for (x, y) in cnt:
                coords.append(f"{x / width:.6f} {y / height:.6f}")
            line = f"{label} " + " ".join(coords)
            lines.append(line)
            
    process_mask(tumor_mask, TUMOR_LABEL)
    process_mask(fluid_mask, FLUID_LABEL)
    with open(yolo_label_output_path, "w") as f:
        for l in lines:
            f.write(l + "\n")

def yolo2mask_separate_outputs(yolo_label_path: str, tumor_output_mask_path: str, fluid_output_mask_path: str, width: int, height: int, override: bool = False):
    """
    Converts YOLO polygon annotations into a grayscale segmentation masks.
    
    Parameters:
        yolo_label_path (str): Path to the YOLO label file (.txt) containing polygon annotations.
        tumor_output_mask_path (str): Path where the generated tumor mask image will be saved.
        fluid_output_mask_path (str): Path where the generated fluid mask image will be saved.
        override (bool, optional): If True, overwrites the output mask if it already exists. Defaults to False.
    """
    if (os.path.exists(tumor_output_mask_path) or os.path.exists(tumor_output_mask_path)) and not override:
        raise Exception("Output mask already exists. Set override=True to override")
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
    
    tumor_final_mask = np.zeros((height, width), dtype=np.uint8)
    fluid_final_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(partial_masks)):
        kind = TUMOR if labels[i] == TUMOR_LABEL else FLUID
        for y in range(height):
            for x in range(width):
                if partial_masks[i][y][x] and tumor_final_mask[y][x] == 0 and kind == TUMOR:
                    tumor_final_mask[y][x] = 255
                elif partial_masks[i][y][x] and fluid_final_mask[y][x] == 0 and kind == FLUID:
                    fluid_final_mask[y][x] = 255
    
    tumor_mask_image = Image.fromarray(tumor_final_mask)
    fluid_mask_image = Image.fromarray(fluid_final_mask)
    tumor_dirname = os.path.dirname(tumor_output_mask_path)
    fluid_dirname = os.path.dirname(fluid_output_mask_path)
    os.makedirs(tumor_dirname, exist_ok=True)
    os.makedirs(fluid_dirname, exist_ok=True)
    tumor_mask_image.save(tumor_output_mask_path)
    fluid_mask_image.save(fluid_output_mask_path)

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