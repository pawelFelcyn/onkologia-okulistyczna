import os
import glob
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "segmentation_masks")

def yolo2mask(yolo_txt_path: str, output_mask_path: str, width: int, height: int, mask_value: int = 255):
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if os.path.getsize(yolo_txt_path) == 0:
        cv2.imwrite(output_mask_path, mask)
        return

    with open(yolo_txt_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        coords = list(map(float, parts[1:]))
        points = []
        
        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i+1] * height)
            points.append([x, y])
            
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], color=mask_value)
        
    cv2.imwrite(output_mask_path, mask)


def verify_dataset(base_data_dir: str = DEFAULT_DATA_DIR):
    patient_dirs = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
    
    for patient in patient_dirs:
        print(f"\nChecking patient: {patient}")
        patient_path = os.path.join(base_data_dir, patient)

        yolo_labels_dir = os.path.join(patient_path, "yolo_labels")
        original_masks_dir = os.path.join(patient_path, "masks")
        check_dir = os.path.join(patient_path, "check_masks_from_yolo")

        if not os.path.exists(yolo_labels_dir):
            print(f"  -> No 'yolo_labels' folder, skipping.")
            continue

        os.makedirs(check_dir, exist_ok=True)

        yolo_files = glob.glob(os.path.join(yolo_labels_dir, "*.txt"))

        if not yolo_files:
            print(f"  -> 'yolo_labels' folder is empty.")
            continue
            
        for yolo_path in yolo_files:
            filename = os.path.basename(yolo_path)
            base_name = os.path.splitext(filename)[0]
            
            original_mask_path = os.path.join(original_masks_dir, base_name + ".png")
            output_mask_path = os.path.join(check_dir, base_name + ".png")
            
            if not os.path.exists(original_mask_path):
                print(f"  No reference file {base_name}.png to read dimensions.")
                continue

            orig_img = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
            if orig_img is None:
                print(f"  Cannot read image {original_mask_path}")
                continue

            height, width = orig_img.shape

            try:
                yolo2mask(yolo_path, output_mask_path, width, height)
                print(f"  Generated test mask: {base_name}.png")
            except Exception as e:
                print(f"  Error on file {filename}: {e}")

if __name__ == "__main__":
    verify_dataset()