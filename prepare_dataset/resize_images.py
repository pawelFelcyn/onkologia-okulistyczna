from PIL import Image
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
PATH_TO_DATA = "../data/new_dataset/splits"

    
def resize_image(in_path: str) -> Image.Image:
    with Image.open(in_path) as img:
        img = img.convert("L")  
        return img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

def resize_image_and_save(in_path: str, out_path: str) -> None:
        final_image_with_background = resize_image(in_path)
        final_image_with_background.save(out_path)
    
def _resize_task(task):
    return resize_image_and_save(*task)

def batch_resize_images(input_dir: str, output_dir: str, max_workers: int = 8) -> None:
    """Resize all images in input_dir and save them to output_dir."""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist.")
    
    images = [image for image in os.listdir(input_dir) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tasks = [(os.path.join(input_dir, img), os.path.join(output_dir, img)) for img in images]

    os.makedirs(output_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_resize_task, task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Resizing images"):
            pass


if __name__ == "__main__":
    batch_resize_images(PATH_TO_DATA + "/test/images",PATH_TO_DATA + "/test/images-resized", 12)
    batch_resize_images(PATH_TO_DATA + "/test/masks",PATH_TO_DATA + "/test/masks-resized", 12)
    batch_resize_images(PATH_TO_DATA + "/train/images",PATH_TO_DATA + "/train/images-resized", 12)
    batch_resize_images(PATH_TO_DATA + "/train/masks",PATH_TO_DATA + "/train/masks-resized", 12)
    batch_resize_images(PATH_TO_DATA + "/val/images",PATH_TO_DATA + "/val/images-resized", 12)
    batch_resize_images(PATH_TO_DATA + "/val/masks",PATH_TO_DATA + "/val/masks-resized", 12)
    