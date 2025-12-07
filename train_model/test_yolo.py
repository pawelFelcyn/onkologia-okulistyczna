from ultralytics import YOLO
import argparse
from utils import make_yolo_split
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='train_model/.env')

def main(test_csv: str, model_to_test: str) -> None:
    make_yolo_split(test_csv, "test")
    
    model = YOLO(model_to_test)
    model.val(
        data="data.yaml",
        split="test",
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO model from CSV splits with explicit labels.")
    
    default_split = os.getenv('SPLIT', 'Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct')
    
    parser.add_argument(
        "--test_csv",
        type=str,
        default=os.path.join(default_split, "test.csv"),
        help="Path to test.csv",
    )
    
    parser.add_argument("--model_to_test", type=str, default="models/weights.pt", help="Model that should be tested")
    
    args = parser.parse_args()
    
    main(**vars(args))