import cv2
import numpy as np
import os
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "segmentation_masks")

def mask2yolo(mask_path: str, yolo_txt_path: str, label_id: int = 0, epsilon_factor: float = 0.0005):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Nie można wczytać maski: {mask_path}")
        
    height, width = mask.shape
    _, bin_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    lines = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
            
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        approx = approx.reshape(-1, 2)
        
        if len(approx) < 3:
            continue
            
        coords = []
        for (x, y) in approx:
            norm_x = max(0.0, min(1.0, x / width))
            norm_y = max(0.0, min(1.0, y / height))
            coords.append(f"{norm_x:.6f} {norm_y:.6f}")
            
        line = f"{label_id} " + " ".join(coords)
        lines.append(line)
        
    os.makedirs(os.path.dirname(yolo_txt_path) or '.', exist_ok=True)
    with open(yolo_txt_path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def process_dataset(base_data_dir: str = DEFAULT_DATA_DIR):
    patient_dirs = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
    
    for patient in patient_dirs:
        print(f"Rozpoczynam przetwarzanie: {patient}")
        patient_path = os.path.join(base_data_dir, patient)
        
        masks_dir = os.path.join(patient_path, "masks")
        yolo_labels_dir = os.path.join(patient_path, "yolo_labels")
        
        os.makedirs(yolo_labels_dir, exist_ok=True)
        
        if not os.path.exists(masks_dir):
            print(f"  -> Brak folderu 'masks' dla {patient}, pomijam.")
            continue
            
        mask_files = glob.glob(os.path.join(masks_dir, "*.png"))
        
        if not mask_files:
            print(f"  -> Folder 'masks' jest pusty dla {patient}.")
            continue
            
        for mask_path in mask_files:
            filename = os.path.basename(mask_path)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            
            yolo_txt_path = os.path.join(yolo_labels_dir, txt_filename)
            
            try:
                mask2yolo(mask_path, yolo_txt_path)
                print(f"  ✓ Zapisano: {txt_filename}")
            except Exception as e:
                print(f"  ✗ Błąd przy pliku {filename}: {e}")

if __name__ == "__main__":
    process_dataset()