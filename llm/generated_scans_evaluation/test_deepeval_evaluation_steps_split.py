import os
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase, MLLMImage
from deepeval import evaluate
import glob
from deepeval.evaluate import AsyncConfig
from deepeval.evaluate.types import EvaluationResult, TestResult
import random
from datetime import datetime
import pandas as pd
from typing import List
from deepeval.metrics.g_eval import Rubric

class TestCaseMetadata:
    def __init__(self, original_file_path: str, is_valid: bool) -> None:
        self.original_file_path = original_file_path
        self.is_valid = is_valid

INVALID_IMAGES_DIR = "./Ophthalmic_Scans/generated/OCT/invalid/images"
VALID_IMAGES_DIR = "./Ophthalmic_Scans/generated/OCT/valid/images"

BASE_STEPS = [
    "You are a structural image validator. Your task is to assess whether the image follows the required anatomical structure. Do not be overly strict on minor variations. Focus only on clear, confident structural violations.",
    "Confidence Rule: Only mark a violation if you are confident that the structural rule is broken. If you are uncertain, assume the structure is correct.",
    "Top Layer Check: The top layer should generally appear dark (near black). Small variations in shade or shape are acceptable. Mark it invalid only if it is clearly not dark or missing.",
    "Second Layer Check: The second layer should follow the contour of the top layer and usually contain a single centered narrowing. Mark invalid only if you are confident there are multiple narrowings or a clearly off-center narrowing.",
    "Third Layer Check: The third layer should be a thin dark band below the second layer. Mark invalid only if it is clearly missing, broken, or displaced.",
    "Fourth Layer Check: The fourth layer should have a generally irregular, bubbly appearance. Do not penalize small irregularities in texture. Mark invalid only if the structure is clearly non-vesicular or uniform in a way that contradicts the expected anatomy.",
    "Bottom Layer Check: The bottom layer should generally form a gray gradient from lighter (top) to darker (bottom). Mark invalid only if the gradient is clearly absent or reversed.",
    "Critical Error Rule: If you are confident that ANY single major structural rule is violated, classify the image as invalid immediately.",
    "Final Decision Rule: The image is valid (score = 1) unless at least one clearly identified structural violation is present. A single confident critical violation is sufficient to mark the image invalid (score = 0)."
]
ASYNC_CONFIG = AsyncConfig(
    max_concurrent=1,
    throttle_value=2
)

def get_images_from_dir(directory, valid: bool):
    abs_path = os.path.abspath(directory)
    if not os.path.exists(abs_path):
        return []
    paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        paths.extend(glob.glob(os.path.join(abs_path, ext)))
    return [(MLLMImage(url=p, local=True), TestCaseMetadata(p, valid)) for p in paths]


def split_images():
    valid_imgs = get_images_from_dir(VALID_IMAGES_DIR, True)
    invalid_imgs = get_images_from_dir(INVALID_IMAGES_DIR, False)

    random.shuffle(valid_imgs)
    random.shuffle(invalid_imgs)

    valid_mid = len(valid_imgs) // 2
    invalid_mid = len(invalid_imgs) // 2

    valid_A, valid_B = valid_imgs[:valid_mid], valid_imgs[valid_mid:]
    invalid_A, invalid_B = invalid_imgs[:invalid_mid], invalid_imgs[invalid_mid:]

    set_A = valid_A + invalid_A
    set_B = valid_B + invalid_B

    random.shuffle(set_A)
    random.shuffle(set_B)

    return set_A, set_B


def get_results_df(results: List[TestResult]) -> pd.DataFrame:
    rows = []

    for result in results:
        score = None
        reason = None

        if result.metrics_data:
            metric = result.metrics_data[0]
            score = getattr(metric, "score", None)
            reason = getattr(metric, "reason", None)

        rows.append({
            "score": score,
            "reason": reason,
            "valid": result.additional_metadata.get("valid") if result.additional_metadata else None,
            "image_path": result.additional_metadata.get("image_path") if result.additional_metadata else None,
        })

    return pd.DataFrame(rows)

def _compute_prf(df: pd.DataFrame, threshold: float):
    tp = fp = tn = fn = 0

    for _, row in df.iterrows():
        pred = row["score"] >= threshold
        true = bool(row["valid"])

        if pred and true:
            tp += 1
        elif pred and not true:
            fp += 1
        elif not pred and not true:
            tn += 1
        elif not pred and true:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return precision, recall, f1


def save_threshold_analysis(df: pd.DataFrame, evaluation_steps: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    thresholds = [i / 100 for i in range(0, 101)]
    results = []

    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        p, r, f1 = _compute_prf(df, t)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        results.append({
            "threshold": t,
            "precision": p,
            "recall": r,
            "f1": f1
        })

    best = max(results, key=lambda x: x["f1"])

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(thresholds, precisions, label="precision")
    plt.plot(thresholds, recalls, label="recall")
    plt.plot(thresholds, f1s, label="f1")
    plt.xlabel("threshold")
    plt.legend()
    plt.title("Threshold vs metrics")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "threshold_metrics.png"))
    plt.close()

    plt.figure()
    plt.plot(recalls, precisions)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-Recall curve")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "precision_recall_curve.png"))
    plt.close()

    import numpy as np

    valid_scores = df[df["valid"] == True]["score"].dropna().values
    invalid_scores = df[df["valid"] == False]["score"].dropna().values

    separation = {
        "valid_mean": float(np.mean(valid_scores)) if len(valid_scores) else None,
        "invalid_mean": float(np.mean(invalid_scores)) if len(invalid_scores) else None,
        "valid_std": float(np.std(valid_scores)) if len(valid_scores) else None,
        "invalid_std": float(np.std(invalid_scores)) if len(invalid_scores) else None,
        "mean_gap": float(np.mean(valid_scores) - np.mean(invalid_scores))
        if len(valid_scores) and len(invalid_scores) else None,
    }

    import json

    with open(os.path.join(out_dir, "threshold_results.json"), "w") as f:
        json.dump({
            "best_threshold": best,
            "all_thresholds": results,
            "separation_stats": separation,
        }, f, indent=2)

    with open(os.path.join(out_dir, "evaluation_steps.json"), "w") as f:
        json.dump({"evaluation_steps": evaluation_steps}, f, indent=2)


def save_evaluation_result(result: EvaluationResult, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = get_results_df(result.test_results)
    df.to_csv(os.path.join(out_dir, "evaluation_result.csv"), index=False)
    save_threshold_analysis(df, BASE_STEPS, out_dir)


def run_evaluation(images, out_dir: str) -> None:
    metric = GEval(
        name="Valid Image Integrity",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=BASE_STEPS,
        threshold=0.75,
        rubric=[
            Rubric(
                score_range=(0, 0),
                expected_outcome="Any structural violation in layer order, shape, or continuity."
            ),
            Rubric(
                score_range=(1, 1),
                expected_outcome="All layers strictly correct, no violations present."
            )
        ]
    )

    test_cases = [LLMTestCase(
        multimodal=True,
        input="Analyze OCT scan structural integrity.",
        actual_output=f"{image[0]}",
        additional_metadata={
            "valid": image[1].is_valid,
            "image_path": image[1].original_file_path
        }
    ) for image in images]

    result: EvaluationResult = evaluate(test_cases=test_cases, metrics=[metric], async_config=ASYNC_CONFIG)
    save_evaluation_result(result, out_dir)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("deepeval_evaluation_results", timestamp)

    set_A, set_B = split_images()

    print(f"Set A: {len(set_A)} images ({sum(1 for _, m in set_A if m.is_valid)} valid, {sum(1 for _, m in set_A if not m.is_valid)} invalid)")
    print(f"Set B: {len(set_B)} images ({sum(1 for _, m in set_B if m.is_valid)} valid, {sum(1 for _, m in set_B if not m.is_valid)} invalid)")
    print(f"Results will be saved to: {base_dir}")

    print("\n--- Running evaluation for Set A ---")
    run_evaluation(set_A, os.path.join(base_dir, "set_A"))

    print("\n--- Running evaluation for Set B ---")
    run_evaluation(set_B, os.path.join(base_dir, "set_B"))

    print(f"\nDone. Results saved to: {base_dir}")
