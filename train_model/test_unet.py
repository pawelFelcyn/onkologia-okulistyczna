import unet_utils
import torch
import os
from torch.utils.data import DataLoader
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path='train_model/.env')

def main(split: str, model_to_test: str, batch: int) -> None:
    root_dir = os.path.join("Ophthalmic_Scans")
    test_csv = os.path.join(split, 'test.csv')
    test_dataset = unet_utils.UNetDataset(test_csv, root_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    model = unet_utils.UNet(3, 2)
    model.load_state_dict(torch.load(model_to_test))
    model.test_model(test_loader)
    
    print("\n✅ Model tested.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO model from CSV splits with explicit labels.")
    
    default_split = os.getenv('SPLIT', 'Ophthalmic_Scans/splits/tumor_and_fluid_segmentation')
    
    parser.add_argument(
        "--split",
        type=str,
        default=default_split,
        help="Path to test.csv",
    )
    
    parser.add_argument("--model_to_test", type=str, default="models/unet/weights.pth", help="Model that should be tested")
    
    default_batch = int(os.getenv('BATCH', '16'))
    parser.add_argument("--batch", type=int, default=default_batch)
    
    args = parser.parse_args()
    
    main(**vars(args))