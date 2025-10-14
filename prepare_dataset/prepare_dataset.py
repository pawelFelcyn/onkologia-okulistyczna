#!/usr/bin/env python3
import argparse
import os
import yolo_labels_utils
import split_utils
#import augment as MyA
import shutil
from resize_images import batch_resize_images

def get_arguments():
    parser = argparse.ArgumentParser()
    default_root_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--root_dir", type=str, default=default_root_dir, help="Root directory of the dataset (directory that contains images and labels directories)")
    default_dest = os.path.join(os.path.abspath(__file__), "dataset")
    parser.add_argument("--dest", type=str, default=default_dest, help="Destination directory for the generated dataset")
    return parser.parse_args()

def generate_masks(root_dir: str):
    images_dir = os.path.join(root_dir, "images")
    yolo_labels_dir = os.path.join(root_dir, "labels")
    output_masks_dir = os.path.join(root_dir, "masks")
    yolo_labels_utils.generate_masks_from_labels(images_dir, yolo_labels_dir, output_masks_dir)
    
def prepare_images(root_dir: str, max_workers: int) -> None:
    img_path = os.path.join(root_dir, "images")
    img_original_path = os.path.join(root_dir, "images_original")
    
    if os.path.exists(img_original_path):
        shutil.rmtree(img_original_path)
    os.rename(img_path, img_original_path)
    
    batch_resize_images(img_original_path, img_path, max_workers)
    
def process_dataset(root_dir: str, dest: str) -> None:
    split_utils.make_dataset(root_dir, dest)
    traning_data_dir = os.path.join(dest, "splits", "train")
    MyA.augment_training_data(traning_data_dir)


def generate_labels_from_resized_masks(root_dir: str):
    masks_dir = os.path.join(root_dir, "masks-resized")
    images_dir = os.path.join(root_dir, "images")
    output_labels_dir = os.path.join(root_dir, "labels_from_masks_resized")
    yolo_labels_utils.generate_labels_from_masks(masks_dir, images_dir, output_labels_dir)


def main():
    arguments = get_arguments()
    # set max_workers according to your CPU
    prepare_images(arguments.root_dir, 12)
    generate_masks(arguments.root_dir)
    generate_labels_from_resized_masks(arguments.root_dir)
    process_dataset(arguments.root_dir, arguments.dest)
  


if __name__ == "__main__":
    main()