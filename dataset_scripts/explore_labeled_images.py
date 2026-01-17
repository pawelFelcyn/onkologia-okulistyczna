import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import utils

labeled_images = utils.get_all_labeled_images()
print(f"Found {len(labeled_images)} labeled images.")

class ImageBrowser:
    def __init__(self, master, images):
        self.master = master
        self.images = images
        self.index = 0

        self.show_masks = tk.BooleanVar(value=True)

        self.index_var = tk.IntVar(value=self.index)

        self.setup_ui()
        self.update_image()

    def setup_ui(self):
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top_frame, text="Index:").pack(side=tk.LEFT)
        self.index_entry = tk.Spinbox(top_frame, from_=0, to=len(self.images)-1,
                                      textvariable=self.index_var, command=self.on_index_change, width=5)
        self.index_entry.pack(side=tk.LEFT)

        tk.Checkbutton(top_frame, text="Show Masks", variable=self.show_masks,
                       command=self.update_image).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.master, width=512, height=512)
        self.canvas.pack()

        nav_frame = tk.Frame(self.master)
        nav_frame.pack(side=tk.BOTTOM)
        tk.Button(nav_frame, text="Prev", command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT)

    def load_image(self, idx):
        _, img_path, _, tumor_mask_path, fluid_mask_path = self.images[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.show_masks.get():
            if tumor_mask_path:
                tumor_mask = cv2.imread(tumor_mask_path, cv2.IMREAD_GRAYSCALE)
                tumor_mask = cv2.resize(tumor_mask, (img.shape[1], img.shape[0]))
                img = self.apply_mask(img, tumor_mask, color=(0,0,255))
            if fluid_mask_path:
                fluid_mask = cv2.imread(fluid_mask_path, cv2.IMREAD_GRAYSCALE)
                fluid_mask = cv2.resize(fluid_mask, (img.shape[1], img.shape[0]))
                img = self.apply_mask(img, fluid_mask, color=(255,0,0))

        return Image.fromarray(img)

    def apply_mask(self, img, mask, color=(255,0,0), alpha=0.5):
        binary_mask = mask > 0
        colored_mask = np.zeros_like(img)
        colored_mask[binary_mask] = color
        return cv2.addWeighted(img, 1.0, colored_mask, alpha, 0)

    def update_image(self):
        self.index = self.index_var.get()
        pil_img = self.load_image(self.index)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0,0, anchor="nw", image=self.tk_img)

    def prev_image(self):
        self.index = max(0, self.index - 1)
        self.index_var.set(self.index)
        self.update_image()

    def next_image(self):
        self.index = min(len(self.images)-1, self.index + 1)
        self.index_var.set(self.index)
        self.update_image()

    def on_index_change(self):
        self.index = self.index_var.get()
        self.update_image()

root = tk.Tk()
root.title("Image Browser with Masks")
browser = ImageBrowser(root, labeled_images)
root.mainloop()