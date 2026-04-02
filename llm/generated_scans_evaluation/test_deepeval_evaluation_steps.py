import os
import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase, MLLMImage
from deepeval import evaluate
import glob
from deepeval.evaluate import AsyncConfig

INVALID_FINAL_STEP = "The result should be between 0 and 1: 1 if the image is invalid, and 0 if it is valid."
VALID_FINAL_STEP = "The result should be between 0 and 1: 1 if the image is valid, and 0 if it is invalid."
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
    "Apply Lenience for Organic Shapes: Do not penalize for imperfect lines; focus on whether the required structural layers and their sequence are preserved."
]

ASYNC_CONFIG = AsyncConfig(
    max_concurrent=1,
    throttle_value=2
)

def get_images_from_dir(directory):
    abs_path = os.path.abspath(directory)
    if not os.path.exists(abs_path):
        return []
    paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        paths.extend(glob.glob(os.path.join(abs_path, ext)))
    return [MLLMImage(url=p, local=True) for p in paths]

@pytest.fixture
def invalid_images():
    imgs = get_images_from_dir(INVALID_IMAGES_DIR)
    assert len(imgs) > 0, f"No images in {INVALID_IMAGES_DIR}"
    return imgs

@pytest.fixture
def valid_images():
    imgs = get_images_from_dir(VALID_IMAGES_DIR)
    assert len(imgs) > 0, f"No images in {VALID_IMAGES_DIR}"
    return imgs
    

def test_image_structural_invalid(invalid_images):
    metric = GEval(
        name="Invalid Image Integrity",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=BASE_STEPS + [INVALID_FINAL_STEP],
        threshold=0.5
    )

    test_cases = [LLMTestCase(
        multimodal=True,
        input="Analyze OCT scan structural integrity.",
        actual_output=f"{image}"
    ) for image in invalid_images]

    evaluate(test_cases=test_cases, metrics=[metric], async_config=ASYNC_CONFIG)

def test_image_structural_valid(valid_images):
    metric = GEval(
        name="Valid Image Integrity",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=BASE_STEPS + [VALID_FINAL_STEP],
        threshold=0.5
    )

    test_cases = [LLMTestCase(
        multimodal=True,
        input="Analyze OCT scan structural integrity.",
        actual_output=f"{image}"
    ) for image in valid_images]

    evaluate(test_cases=test_cases, metrics=[metric], async_config=ASYNC_CONFIG)