from ultralytics import YOLO
import argparse
import os
import cv2
import numpy as np
import pandas as pd
from utils import make_yolo_split
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
            points = np.array(coords).reshape(-1, 2)
            points[:, 0] *= img_w
            points[:, 1] *= img_h
            points = points.astype(np.int32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            gt.setdefault(cls, []).append(mask)
    return gt


def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter / union) if union > 0 else 0.0


def greedy_match(pred_masks, gt_masks, threshold):
    """
    Greedy IoU matching. Returns list of (category, pred_idx_or_None, gt_idx_or_None).
    """
    if not pred_masks and not gt_masks:
        return [('tn', None, None)]

    n_pred, n_gt = len(pred_masks), len(gt_masks)
    iou_matrix = np.zeros((n_pred, n_gt))
    for pi, pm in enumerate(pred_masks):
        for gi, gm in enumerate(gt_masks):
            iou_matrix[pi, gi] = compute_iou(pm, gm)

    matched_pred, matched_gt = set(), set()
    pairs = sorted(
        [(iou_matrix[pi, gi], pi, gi) for pi in range(n_pred) for gi in range(n_gt)],
        reverse=True,
    )

    results = []
    for iou, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        if iou >= threshold:
            results.append(('tp', pi, gi))
        else:
            results.append(('fp', pi, None))
            results.append(('fn', None, gi))

    for pi in range(n_pred):
        if pi not in matched_pred:
            results.append(('fp', pi, None))
    for gi in range(n_gt):
        if gi not in matched_gt:
            results.append(('fn', None, gi))

    return results


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


def main(test_csv, model_to_test, iou_threshold):
    make_yolo_split(test_csv, "test")
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

        results = model.predict(img_path, verbose=False)
        result = results[0]

        pred_by_class = {}
        if result.masks is not None:
            for j in range(len(result.masks)):
                cls = int(result.boxes.cls[j].item())
                mask = (result.masks.data[j].cpu().numpy() > 0.5).astype(np.uint8)
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                box = result.boxes.data[j].cpu().numpy()
                pred_by_class.setdefault(cls, []).append((mask, box))

        for cls_id, cls_name in CLASSES.items():
            pred_items = pred_by_class.get(cls_id, [])
            gt_masks = gt_by_class.get(cls_id, [])
            pred_masks = [m for m, _ in pred_items]
            pred_boxes = [b for _, b in pred_items]

            for category, pred_idx, gt_idx in greedy_match(pred_masks, gt_masks, iou_threshold):
                dir_path = os.path.join(out_dir, cls_name, category)
                os.makedirs(dir_path, exist_ok=True)

                pred_path = os.path.join(dir_path, f'{i}_pred{ext}')
                label_path = os.path.join(dir_path, f'{i}_label{ext}')

                if category == 'tn':
                    cv2.imwrite(pred_path, img)
                    cv2.imwrite(label_path, img)
                elif category == 'tp':
                    save_pred_image(img, pred_masks[pred_idx], pred_boxes[pred_idx], pred_path)
                    save_label_image(img, gt_masks[gt_idx], label_path)
                elif category == 'fp':
                    save_pred_image(img, pred_masks[pred_idx], pred_boxes[pred_idx], pred_path)
                    cv2.imwrite(label_path, img)
                elif category == 'fn':
                    cv2.imwrite(pred_path, img)
                    save_label_image(img, gt_masks[gt_idx], label_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save per-image confusion matrix visualizations for segmentation.")

    default_split = os.getenv('SPLIT', 'Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct')
    default_test_model = os.getenv('TEST_MODEL', 'models/weights.pt')

    parser.add_argument('--test_csv', type=str, default=os.path.join(default_split, 'test.csv'))
    parser.add_argument('--model_to_test', type=str, default=default_test_model)
    parser.add_argument('--iou_threshold', type=float, default=0.8)

    args = parser.parse_args()
    main(**vars(args))
