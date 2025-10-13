import argparse
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
import numpy as np


# -----------------------------
# Helper: load a given split
# -----------------------------
def load_split(dataset_path, split_name):
    split_dir = os.path.join(dataset_path, "splits", split_name)
    tsv_path = os.path.join(split_dir, f"{split_name}.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Missing file: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")
    return df, images_dir, masks_dir


# -----------------------------
# Dataset Explorer Class
# -----------------------------
class DatasetExplorer:
    def __init__(self, dataset_path, initial_split="test"):
        self.dataset_path = dataset_path
        self.split = initial_split
        self.df, self.images_dir, self.masks_dir = load_split(dataset_path, initial_split)
        self.index = 0
        self.show_fluid = True
        self.show_tumor = True

        # Setup figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(left=0.25, right=0.8, bottom=0.15)
        self.ax.axis("off")

        # Checkboxes (for toggling overlays)
        rax = plt.axes([0.02, 0.5, 0.18, 0.15])
        self.check = CheckButtons(rax, ["Fluid (blue)", "Tumor (red)"], [True, True])
        self.check.on_clicked(self.toggle_mask)

        # Buttons for navigation
        axprev = plt.axes([0.4, 0.05, 0.1, 0.05])
        axnext = plt.axes([0.52, 0.05, 0.1, 0.05])
        self.bnext = Button(axnext, "Next ▶")
        self.bprev = Button(axprev, "◀ Prev")
        self.bnext.on_clicked(self.next_image)
        self.bprev.on_clicked(self.prev_image)

        # Split selection buttons
        axtrain = plt.axes([0.68, 0.05, 0.08, 0.05])
        axval = plt.axes([0.77, 0.05, 0.08, 0.05])
        axtest = plt.axes([0.86, 0.05, 0.08, 0.05])
        self.btrain = Button(axtrain, "Train")
        self.bval = Button(axval, "Val")
        self.btest = Button(axtest, "Test")
        self.btrain.on_clicked(lambda e: self.change_split("train"))
        self.bval.on_clicked(lambda e: self.change_split("val"))
        self.btest.on_clicked(lambda e: self.change_split("test"))

        # Initial display
        self.update_display()

    # -----------------------------
    def change_split(self, new_split):
        """Change the dataset split (train/test/val)"""
        self.split = new_split
        self.df, self.images_dir, self.masks_dir = load_split(self.dataset_path, new_split)
        self.index = 0
        print(f"Switched to split: {new_split}")
        self.update_display()

    # -----------------------------
    def toggle_mask(self, label):
        if "Fluid" in label:
            self.show_fluid = not self.show_fluid
        if "Tumor" in label:
            self.show_tumor = not self.show_tumor
        self.update_display()

    # -----------------------------
    def next_image(self, event=None):
        self.index = (self.index + 1) % len(self.df)
        self.update_display()

    def prev_image(self, event=None):
        self.index = (self.index - 1) % len(self.df)
        self.update_display()

    # -----------------------------
    def update_display(self):
        row = self.df.iloc[self.index]
        image_path = os.path.join(self.images_dir, f"{row['id']}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.images_dir, f"{row['id']}.jpg")
        extension = os.path.splitext(image_path)[1].lower()
        mask_path = os.path.join(self.masks_dir, f"{row['id']}{extension}")

        self.ax.clear()

        if not os.path.exists(image_path):
            self.ax.text(0.5, 0.5, f"Image not found for ID {row['id']}", ha="center", va="center")
            self.ax.axis("off")
            plt.draw()
            return

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.ax.imshow(image)
        self.ax.axis("off")

        # Load and overlay mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
            if self.show_fluid:
                overlay[mask == 127] = [0, 0, 255, 100]  # blue overlay
            if self.show_tumor:
                overlay[mask == 255] = [255, 0, 0, 100]  # red overlay
            self.ax.imshow(overlay)

        # Metadata box
        info = (
            f"Split: {self.split}\n"
            f"ID: {row['id']}\n"
            f"Patient ID: {row['patient_id']}\n"
            f"Date: {row['date']}\n"
            f"Has Fluid: {row['has_fluid']}\n"
            f"Has Tumor: {row['has_tumor']}"
        )
        self.ax.text(
            1.05, 0.5, info, transform=self.ax.transAxes, fontsize=10,
            va="center", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        plt.draw()


# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Interactive OCT dataset explorer with masks.")
    default_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', "new_dataset")
    parser.add_argument("--path", type=str, default=default_dataset, help="Path to dataset root directory")
    args = parser.parse_args()

    explorer = DatasetExplorer(args.path)
    plt.show()


if __name__ == "__main__":
    main()