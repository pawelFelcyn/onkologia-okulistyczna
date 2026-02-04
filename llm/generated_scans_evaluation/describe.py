#!/usr/bin/env python3

from transformers import pipeline, BitsAndBytesConfig
from PIL import Image
import torch
import pandas as pd
import json
from dotenv import load_dotenv
import os
import utils
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
load_dotenv('env.local')

model = os.getenv("MODEL_NAME", 'google/medgemma-4b-it')
print("Using model:", model)
prompt_path = os.getenv("PROMPT_PATH", 'Ophthalmic_Scans/prompts/recognize_generation_failures.json')

def save_outputs(df: pd.DataFrame, model_name: str):
    model_dir_name = model_name.replace("/", "_")

    base_dir = "llm_eval_outputs"
    model_dir = os.path.join(base_dir, model_dir_name)

    os.makedirs(model_dir, exist_ok=True)

    i = 1
    while True:
        filename = f"outputs{i}.csv"
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            break
        i += 1

    df.to_csv(path, index=False)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

pipe = pipeline(
    "image-text-to-text",
    model=model,
    model_kwargs={
        "quantization_config": bnb_config,
        "dtype": torch.bfloat16,
    },
    device=device
)

messages = utils.build_messages_from_prompt(prompt_path)
df = pd.read_csv("Ophthalmic_Scans/splits/oct_scan_llm_description/test.csv") 
df_outputs = pd.DataFrame(columns=["valid", "description", "image_path"])
length = len(df)
for row in tqdm.tqdm(df.iloc):
    image_url = row["image_path"]
    image = Image.open(os.path.join("Ophthalmic_Scans", image_url)).convert("RGB")
    local_messages = messages.copy()
    local_messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image}
        ]
    })

    output = pipe(text=local_messages, max_new_tokens=2000)
    obj = json.loads(output[0]["generated_text"][-1]['content'])
    df_outputs = pd.concat([df_outputs, pd.DataFrame([{"valid": obj["valid"], "description": obj["notes"], "image_path": image_url}])], ignore_index=True)
    
save_outputs(df_outputs, model)