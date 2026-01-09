from PIL import Image
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_WORKERS = 12

def resize_image(in_path: str, size: tuple, mode: str = "L") -> Image.Image:
    try:
        with Image.open(in_path) as img:
            img = img.convert(mode)
            return img.resize(size, Image.LANCZOS)
    except Exception as e:
        print(f"Error loading {in_path}: {e}")
        return None

def resize_image_and_save(in_path: str, out_path: str, size: tuple, mode: str = "L") -> None:
    final_image = resize_image(in_path, size, mode)
    if final_image:
        try:
            final_image.save(out_path)
        except Exception as e:
            print(f"Failed to save {out_path}: {e}")

def _resize_task(task):
    return resize_image_and_save(*task)

def find_and_resize_recursive(root_dir: str, target_folder: str, size: tuple, overwrite: bool, max_workers: int = 8) -> None:
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} does not exist.")
        return

    tasks = []
    print(f"Scanning for folders named '{target_folder}' in {root_dir}...")

    for current_root, dirs, files in os.walk(root_dir):
        if os.path.basename(current_root) == target_folder:
            if overwrite:
                output_dir = current_root
            else:
                output_dir = current_root + "-resized"

            os.makedirs(output_dir, exist_ok=True)

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    in_path = os.path.join(current_root, file)
                    out_path = os.path.join(output_dir, file)
                    tasks.append((in_path, out_path, size, "L"))

    if not tasks:
        print(f"No images found in folders named '{target_folder}'.")
        return

    print(f"Found {len(tasks)} images. Overwrite mode: {overwrite}. Starting...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_resize_task, task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            pass

if __name__ == "__main__":
    """
    DOCUMENTATION / USAGE EXAMPLES:
    
    1. RECURSIVE MASK OVERWRITE:
       Resize all 'masks' folders and REPLACE original files:
       python script.py --path "C:/Data/Project" --folder_name "masks" --overwrite
       
    2. RECURSIVE IMAGES WITH NEW FOLDER:
       Find 'original_images' and save to 'original_images-resized':
       python script.py --path "C:/Data/Project" --folder_name "original_images" --width 1024 --height 1024
       
    3. CUSTOM SETTINGS:
       python script.py --path "./data" --folder_name "images" --width 256 --height 256 --workers 4
    """
    parser = argparse.ArgumentParser(description="Image Resizing Script")
    parser.add_argument("--path", type=str, required=True, help="Root path to scan")
    parser.add_argument("--folder_name", type=str, default="masks", help="Name of folders to process (e.g. masks or original_images)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite images in their original folder")

    args = parser.parse_args()
    size = (args.width, args.height)

    find_and_resize_recursive(args.path, args.folder_name, size, args.overwrite, args.workers)