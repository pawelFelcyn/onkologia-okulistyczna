import os
import glob
import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase, MLLMImage
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate import AsyncConfig
from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.metrics.g_eval import Rubric
from openai import OpenAI
import base64
import re

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class Qwen3VLJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="qwen/qwen3-vl-235b-a22b-instruct"):
        api_key = os.getenv("OPENROUTER_API_KEY", "TU_WPISZ_KLUCZ_JEŚLI_NIE_MASZ_ENV")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model_name = model_name
        self.name = model_name

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        image_path_match = re.search(r"IMAGE_PATH:(.*?)(?:\n|$)", prompt)
        
        if image_path_match:
            img_path = image_path_match.group(1).strip()
            base64_image = encode_image(img_path)
            
            clean_prompt = prompt.replace(image_path_match.group(0), "")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": clean_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name
    
    def supports_multimodal(self):
        return True

from deepeval.metrics.utils import MULTIMODAL_SUPPORTED_MODELS
MULTIMODAL_SUPPORTED_MODELS[Qwen3VLJudge] = {
    "qwen/qwen3-vl-235b-a22b-instruct": Qwen3VLJudge()
}

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

ASYNC_CONFIG = AsyncConfig(max_concurrent=1, throttle_value=2)

def get_images_from_dir(directory, valid: bool):
    abs_path = os.path.abspath(directory)
    if not os.path.exists(abs_path):
        print(f"Warning: Directory {abs_path} not found.")
        return []
    paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        paths.extend(glob.glob(os.path.join(abs_path, ext)))
    return [(MLLMImage(url=p, local=True), TestCaseMetadata(p, valid)) for p in paths]

def load_all_images():
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
            "valid": result.additional_metadata.get("valid"),
            "image_path": result.additional_metadata.get("image_path"),
        })
    return pd.DataFrame(rows)

def save_threshold_analysis(df: pd.DataFrame, evaluation_steps: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    
    thresholds = [i / 100 for i in range(0, 101)]
    stats = []
    for t in thresholds:
        tp = ((df["score"] >= t) & (df["valid"] == True)).sum()
        fp = ((df["score"] >= t) & (df["valid"] == False)).sum()
        fn = ((df["score"] < t) & (df["valid"] == True)).sum()
        tn = ((df["score"] < t) & (df["valid"] == False)).sum()
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        stats.append({"threshold": t, "precision": p, "recall": r, "f1": f1})

    best = max(stats, key=lambda x: x["f1"])
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, [s["precision"] for s in stats], label="Precision")
    plt.plot(thresholds, [s["recall"] for s in stats], label="Recall")
    plt.plot(thresholds, [s["f1"] for s in stats], label="F1 Score")
    plt.axvline(best["threshold"], color='r', linestyle='--', label=f'Best F1 @ {best["threshold"]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "metrics_plot.png"))
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"best": best, "steps": evaluation_steps}, f, indent=2)

def test_image_structural_valid(images_data):
    qwen_judge = Qwen3VLJudge()

    metric = GEval(
        name="OCT Structural Integrity",
        model=qwen_judge,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=BASE_STEPS,
        threshold=0.5
    )

    test_cases = []
    for img_obj, meta in images_data:
        test_cases.append(LLMTestCase(
            multimodal=True,
            input="Evaluate the structural integrity of this OCT scan.",
            actual_output=f"IMAGE_PATH:{meta.original_file_path}", 
            additional_metadata={
                "valid": meta.is_valid,
                "image_path": meta.original_file_path
            }
        ))

    result: EvaluationResult = evaluate(
        test_cases=test_cases, 
        metrics=[metric], 
        async_config=ASYNC_CONFIG
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(f"evaluation_qwen3", f"{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    df = get_results_df(result.test_results)
    df.to_csv(os.path.join(out_dir, "evaluation_result.csv"), index=False)
    save_threshold_analysis(df, BASE_STEPS, out_dir)
    print(f"Evaluation complete. Results saved in: {out_dir}")

if __name__ == "__main__":
    data = load_all_images()
    if not data:
        print("No images found. Check your paths.")
    else:
        test_image_structural_valid(data)