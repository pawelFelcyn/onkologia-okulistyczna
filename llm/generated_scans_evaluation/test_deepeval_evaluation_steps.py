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
    "Your task is to evaluate the structural integrity of the image provided in ACTUAL_OUTPUT based on the given criteria and classify the image as valid or invalid",
    "Assess the Top Layer: It must be dark (near black). The bottom edge can be straight or an asymmetrical/elliptical arc (one side higher).",
    "Examine the Second Layer (Bright Band): It must follow the contour of the top boundary. It is allowed exactly one centered narrowing. Reject if there are two narrowings or if the narrowing is off-center.",
    "Verify the Third Layer: It must be a thin, dark or black band located immediately below the second layer.",
    "Evaluate the Fourth Layer (Vesicular Layer): Check for an irregular, 'bubbly' texture. It does not need a uniform height and can significantly taper at the edges.",
    "Check the Bottom Layer: It must be a gray gradient that starts light at the top and transitions downward.",
    "Scan for Critical Structural Errors: Strictly penalize any sudden, unnatural cut-offs of layers or mirrored/duplicated layers that break logical flow.",
    "Apply Lenience for Organic Shapes: Do not penalize for imperfect lines; focus on whether the required structural layers and their sequence are preserved.",
    "The result should be between 0 and 1: 1 if the image is valid, and 0 if it is invalid."
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

def images():
    valid_imgs = get_images_from_dir(VALID_IMAGES_DIR, True)
    invalid_imgs = get_images_from_dir(INVALID_IMAGES_DIR, False)
    ret = valid_imgs + invalid_imgs
    random.shuffle(ret)
    return ret
    

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
 
def save_evaluation_result(result: EvaluationResult) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    evaluation_results_dir = os.path.join(
        "deepeval_evaluation_results",
        timestamp
    )
    os.makedirs(evaluation_results_dir, exist_ok=True)
    
    df = get_results_df(result.test_results)
    df.to_csv(os.path.join(evaluation_results_dir, "evaluation_result.csv"), index=False)
    save_threshold_analysis(df, BASE_STEPS, evaluation_results_dir)

def test_image_structural_valid(images):
    metric = GEval(
        name="Valid Image Integrity",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=BASE_STEPS,
        threshold=0.5,
        rubric= [
            Rubric(
                score_range=(0, 0),
                expected_outcome="The image is classified as invalid because it does not meet a key structural requirement. The top layer is missing or does not appear as a dark (near black) region with an acceptable shape. The second layer does not correctly follow the contour of the top boundary and may contain more than one narrowing or a narrowing that is not centered. The third layer is absent or not a thin dark band directly below the second layer. The fourth layer does not exhibit the required irregular, bubbly texture or shows incorrect uniformity. The bottom layer is missing or does not form a proper gray gradient transitioning from lighter at the top to darker at the bottom. The image may also contain critical structural errors such as abrupt layer cut-offs or mirrored/duplicated structures that break logical consistency."
            ),
            Rubric(
                score_range=(1, 1),
                expected_outcome="The image is classified as valid because it satisfies all structural requirements. The top layer is present and appears as a dark (near black) region with an acceptable straight or asymmetrical/elliptical shape. The second layer correctly follows the contour of the top boundary and contains exactly one centered narrowing. The third layer is present as a thin dark or black band immediately below the second layer. The fourth layer exhibits an irregular, bubbly texture with natural variation in height and possible edge tapering. The bottom layer is present as a gray gradient transitioning smoothly from light at the top to darker at the bottom. No critical structural errors are present, and there are no abrupt cut-offs, duplications, or mirrored artifacts disrupting the structure."
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
    save_evaluation_result(result)


if __name__ == "__main__":
    test_image_structural_valid(images())