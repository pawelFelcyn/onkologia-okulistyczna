import unet_utils
import torch
import os
from torch.utils.data import DataLoader
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path='train_model/.env')


def main(split: str, model_to_test: str, batch: int, imgsz: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = os.path.join("Ophthalmic_Scans")
    test_csv = os.path.join(split, 'test.csv')
    test_dataset = unet_utils.UNetDataset(test_csv, root_dir, imgsz=imgsz)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    model = unet_utils.UNet(3, 2)
    model.load_state_dict(torch.load(model_to_test, map_location=device))
    model.test_model(test_loader, device=device)

    print("\nModel evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained UNet model on the test split.")

    default_split = os.getenv('SPLIT', 'Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct')
    default_batch = int(os.getenv('BATCH', '16'))

    parser.add_argument(
        "--split",
        type=str,
        default=default_split,
        help="Directory containing test.csv",
    )
    parser.add_argument(
        "--model_to_test",
        type=str,
        default="models/unet/weights.pth",
        help="Path to the .pth checkpoint to evaluate",
    )
    parser.add_argument("--batch", type=int, default=default_batch)
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Resize images to this size before inference (must match training imgsz)")

    args = parser.parse_args()
    main(**vars(args))