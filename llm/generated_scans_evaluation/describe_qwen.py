#!/usr/bin/env python3

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
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

model_name = "Qwen/Qwen2-VL-7B-Instruct"
print("Using model:", model_name)
prompt_path = os.getenv("PROMPT_PATH", 'Ophthalmic_Scans/prompts/recognize_generation_failures_qwen.json')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    model_name, 
    min_pixels=256*28*28, 
    max_pixels=1280*28*28
)

messages_base = utils.build_messages_from_prompt(prompt_path)
df = pd.read_csv("Ophthalmic_Scans/splits/oct_scan_llm_description/test.csv") 
df_outputs = pd.DataFrame(columns=["valid", "description", "image_path"])

for row in tqdm.tqdm(df.iloc):
    image_url = row["image_path"]
    full_image_path = os.path.join("Ophthalmic_Scans", image_url)
    
    local_messages = messages_base.copy()
    local_messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": full_image_path}
        ]
    })

    text = processor.apply_chat_template(
        local_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(local_messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    try:
        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        obj = json.loads(clean_json)
        
        new_row = {
            "valid": obj.get("valid"),
            "description": obj.get("notes"),
            "image_path": image_url
        }
    except Exception as e:
        print(f"Error parsing JSON for {image_url}: {e}")
        new_row = {"valid": "ERROR", "description": response_text, "image_path": image_url}

    df_outputs = pd.concat([df_outputs, pd.DataFrame([new_row])], ignore_index=True)

utils.save_outputs(df_outputs, model_name)