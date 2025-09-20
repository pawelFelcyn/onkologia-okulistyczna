#!/usr/bin/env python3
import argparse
import os
import yolo_labels_utils
import split_utils
import augment as MyA

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

def main():
    arguments = get_arguments()
    generate_masks(arguments.root_dir)
    split_utils.make_dataset(arguments.root_dir, arguments.dest)
    traning_data_dir = os.path.join(arguments.dest, "splits", "train")
    MyA.augment_training_data(traning_data_dir)



if __name__ == "__main__":
    main()