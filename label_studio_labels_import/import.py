#TODO this file must be cleaned, but it just works for now and we have more important things to do

import json
from export_label_studio import export_label_studio_data, filter_by_image_type
import os
from datetime import datetime
from PIL import Image
import dotenv
import sys

prepare_dataset_dir = os.path.join(os.path.dirname(__file__), "..", "prepare_dataset")
sys.path.append(prepare_dataset_dir)

from yolo_labels_utils import mask2yolo_separate_inputs

dir = os.path.dirname(os.path.abspath(__file__))

env_file = os.path.join(dir, ".env")
dotenv.load_dotenv(env_file) 

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
TOKEN = os.getenv("TOKEN")
PROJECT_ID = os.getenv("PROJECT_ID")

if not LABEL_STUDIO_URL or not TOKEN or not PROJECT_ID:
    raise ValueError("Missing environment variables: LABEL_STUDIO_URL, TOKEN, PROJECT_ID")

MASK_SIZE = (512, 512)


payload = {
    "task_filter_options": {
        "annotated": "only"
    }
}

now = datetime.now()
date_str = now.strftime("%d%m%Y%H%M%S") + f"{now.microsecond // 1000:03d}"
date_str = '06042026110258808'
export_result_dir = os.path.join(dir, 'exports', date_str)
os.makedirs(export_result_dir, exist_ok=True)
export_file = os.path.join(export_result_dir, "oct_completed.json")

# export_id = export_label_studio_data(
#     base_url=LABEL_STUDIO_URL,
#     api_key=TOKEN,
#     project_id=PROJECT_ID,
#     output_file=export_file,
#     export_type="JSON",
#     payload=payload
# )

zip_file = os.path.join(export_result_dir, "export.zip")

export_label_studio_data(
    base_url=LABEL_STUDIO_URL,
    api_key=TOKEN,
    project_id=PROJECT_ID,
    output_file=zip_file,
    export_type="BRUSH_TO_PNG",
    payload=payload
)


# print(f'[+] Export ID: {export_id}')

masks_path = os.path.join(export_result_dir, "masks")

# filter_by_image_type(export_file, "OCT")

## TODO: currently i have to donwload the masks manually, but ideally this should also be automated here

data = json.load(open(export_file, "r", encoding="utf-8"))

# for item in data:
#     task_files = [f for f in os.listdir(masks_path) if f.startswith(f"task-{item['id']}-")]
#     task_files_full = [os.path.join(masks_path, f) for f in task_files]
#     tumor_mask = next((f for f in task_files_full if "Tumor" in f), None)
#     fluid_mask = next((f for f in task_files_full if "Fluid" in f), None)
#     image_path = 'Ophthalmic_Scans/' + item['data']['image'][44:]
#     fluid_mask_exists = fluid_mask is not None and os.path.isfile(fluid_mask)
#     tumor_mask_exists = tumor_mask is not None and os.path.isfile(tumor_mask)
#     masks_exist =fluid_mask_exists or tumor_mask_exists
    
#     metadata_path = image_path.replace('original_images', 'metadata').replace('.png', '.json')
#     label_path = image_path.replace('original_images', 'labels').replace('.png', '.txt')
#     tumor_mask_path = image_path.replace('original_images', 'masks/tumor')
#     fluid_mask_path = image_path.replace('original_images', 'masks/fluid')
#     resized_image_path = image_path.replace('original_images', 'resized_images')
    
#     os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
#     os.makedirs(os.path.dirname(label_path), exist_ok=True)
#     os.makedirs(os.path.dirname(tumor_mask_path), exist_ok=True)
#     os.makedirs(os.path.dirname(fluid_mask_path), exist_ok=True)
#     os.makedirs(os.path.dirname(resized_image_path), exist_ok=True)
    
#     if not masks_exist:
#         print(f"[-] No masks found for task {item['id']} ({image_path})")
#         metadata = json.load(open(metadata_path, "r", encoding="utf-8"))
#         metadata['annotated_by'] = 'Paweł Felcyn'
#         annotation_date = '2026-04-06'
#         metadata['annotation_date'] = annotation_date
#         with open(metadata_path, "w", encoding="utf-8") as f:
#             json.dump(metadata, f, indent=4)
#         original_image = Image.open(image_path)
#         resized_image = original_image.resize(MASK_SIZE, Image.LANCZOS)
#         resized_image.save(resized_image_path)
#         open(label_path, "w").close()
#         maks = Image.new("RGB", (512, 512), color=(0, 0, 0))
#         maks.save(tumor_mask_path)
#         maks.save(fluid_mask_path)
#     else:
#         metadata = json.load(open(metadata_path, "r", encoding="utf-8"))
#         metadata['annotated_by'] = 'Paweł Felcyn'
#         annotation_date = '2026-04-06'
#         metadata['annotation_date'] = annotation_date
#         with open(metadata_path, "w", encoding="utf-8") as f:
#             json.dump(metadata, f, indent=4)
#         original_image = Image.open(image_path)
#         resized_image = original_image.resize(MASK_SIZE, Image.LANCZOS)
#         resized_image.save(resized_image_path)
            
#         if tumor_mask_exists:
#             tumor_mask_image = Image.open(tumor_mask)
#             tumor_mask_image = tumor_mask_image.resize(MASK_SIZE, Image.NEAREST)
#             tumor_mask_image.save(tumor_mask_path)
#         else:
#             maks = Image.new("RGB", (512, 512), color=(0, 0, 0))
#             maks.save(tumor_mask_path)
#         if fluid_mask_exists:
#             fluid_mask_image = Image.open(fluid_mask)
#             fluid_mask_image = fluid_mask_image.resize(MASK_SIZE, Image.NEAREST)
#             fluid_mask_image.save(fluid_mask_path)
#         else:
#             maks = Image.new("RGB", (512, 512), color=(0, 0, 0))
#             maks.save(fluid_mask_path)
            
#         mask2yolo_separate_inputs(
#             tumor_mask_path,
#             fluid_mask_path,
#             label_path,
#             override=True
#         )
    
        