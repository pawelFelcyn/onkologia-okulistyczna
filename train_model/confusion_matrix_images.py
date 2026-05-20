from ultralytics import YOLO
import argparse
import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path='train_model/.env')

CLASSES = {0: 'fluid', 1: 'tumor'}


def get_output_dir(base='runs/segment/c_matrix'):
    os.makedirs(base, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(base, str(i))):
        i += 1
    return os.path.join(base, str(i))


def parse_gt_labels(label_path, img_w, img_h):
    """Returns dict: class_id -> list of binary masks (H, W uint8)."""
    gt = {}
    if not os.path.exists(label_path):
        return gt
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue
            points = np.array(coords).reshape(-1, 2)
            points[:, 0] *= img_w
            points[:, 1] *= img_h
            points = np.round(points).astype(np.int32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            gt.setdefault(cls, []).append(mask)
    return gt


def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter / union) if union > 0 else 0.0


def classify_case(pred_masks, gt_masks, threshold):
    best_iou = 0.0
    best_pred_idx = None
    best_gt_idx = None

    has_pred = len(pred_masks) > 0
    has_gt = len(gt_masks) > 0

    if has_pred and has_gt:
        for pred_idx, pred_mask in enumerate(pred_masks):
            for gt_idx, gt_mask in enumerate(gt_masks):
                iou = compute_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
                    best_gt_idx = gt_idx

    if has_pred and has_gt and best_iou >= threshold:
        return 'tp', best_pred_idx, best_gt_idx, best_iou

    if has_pred and (not has_gt or best_iou < threshold):
        p_idx = best_pred_idx if best_pred_idx is not None else 0
        return 'fp', p_idx, best_gt_idx, best_iou

    if not has_pred and has_gt:
        return 'fn', None, 0, 0.0

    return 'tn', None, None, 0.0


def draw_mask_overlay(img, mask, color, alpha=0.4):
    out = img.copy()
    region = mask > 0
    out[region] = (out[region] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return out


def save_pred_image(img, mask, box, path):
    vis = draw_mask_overlay(img, mask, color=(0, 255, 0))
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(path, vis)


def save_label_image(img, mask, path):
    vis = draw_mask_overlay(img, mask, color=(0, 0, 255))
    cv2.imwrite(path, vis)


def collect_predicted_masks(result, img_w, img_h, conf_threshold):
    pred_by_class = {}
    if result.masks is None or result.boxes is None:
        return pred_by_class

    for j in range(len(result.masks)):
        confidence = float(result.boxes.conf[j].item())
        if confidence < conf_threshold:
            continue

        cls = int(result.boxes.cls[j].item())
        mask = (result.masks.data[j].cpu().numpy() > 0.5).astype(np.uint8)
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        box = result.boxes.data[j].cpu().numpy()
        pred_by_class.setdefault(cls, []).append((mask, box))

    return pred_by_class


def main(test_csv, model_to_test, iou_threshold, conf_threshold):
    model = YOLO(model_to_test)
    out_dir = get_output_dir()

    df = pd.read_csv(test_csv)

    for i, row in df.iterrows():
        img_path = os.path.join('Ophthalmic_Scans', row['image_path'])
        lbl_path = os.path.join('Ophthalmic_Scans', row['label_path'])
        ext = os.path.splitext(img_path)[1]

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        gt_by_class = parse_gt_labels(lbl_path, img_w, img_h)

        results = model.predict(img_path, verbose=False, conf=0.001)
        result = results[0]

        pred_by_class = collect_predicted_masks(result, img_w, img_h, conf_threshold)

        for cls_id, cls_name in CLASSES.items():
            pred_items = pred_by_class.get(cls_id, [])
            gt_masks = gt_by_class.get(cls_id, [])
            pred_masks = [m for m, _ in pred_items]
            pred_boxes = [b for _, b in pred_items]

            category, pred_idx, gt_idx, _ = classify_case(pred_masks, gt_masks, iou_threshold)

            dir_path = os.path.join(out_dir, cls_name, category)
            os.makedirs(dir_path, exist_ok=True)

            sample_id = f'{i}'
            pred_path = os.path.join(dir_path, f'{sample_id}_pred{ext}')
            label_path = os.path.join(dir_path, f'{sample_id}_label{ext}')

            if pred_idx is not None:
                save_pred_image(img, pred_masks[pred_idx], pred_boxes[pred_idx], pred_path)
            else:
                cv2.imwrite(pred_path, img)

            if gt_idx is not None:
                save_label_image(img, gt_masks[gt_idx], label_path)
            elif gt_masks:
                save_label_image(img, gt_masks[0], label_path)
            else:
                cv2.imwrite(label_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save per-image confusion matrix visualizations for segmentation.")

    default_split = os.getenv('SPLIT', 'Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct')
    default_test_model = os.getenv('TEST_MODEL', 'models/weights.pt')
    default_threshold = float(os.getenv('IOU_THRESHOLD', '0.8'))
    default_conf_threshold = float(os.getenv('CONF_THRESHOLD', '0.5'))

    parser.add_argument('--test_csv', type=str, default=os.path.join(default_split, 'test.csv'))
    parser.add_argument('--model_to_test', type=str, default=default_test_model)
    parser.add_argument('--iou_threshold', type=float, default=float(default_threshold))
    parser.add_argument('--conf_threshold', type=float, default=default_conf_threshold)

    args = parser.parse_args()
    main(**vars(args))