# Dataset Explorer

A simple interactive tool to explore OCT images and their segmentation masks.

## Features
- Browse through train/val/test splits
- Toggle fluid (blue) and tumor (red) mask overlays
- View image metadata (patient ID, date, labels)
- Navigate images with prev/next buttons

## Usage

Run with default dataset path:
```bash
python explorer.py
```

Or specify a custom dataset path:
```bash
python explorer.py --path "path/to/your/dataset"
```

## Controls
- Use "Next ▶" and "◀ Prev" buttons to navigate through images
- Toggle overlays using checkboxes
- Switch between splits using Train/Val/Test buttons
- Image metadata is displayed on the right side
