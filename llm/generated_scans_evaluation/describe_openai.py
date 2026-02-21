#!/usr/bin/env python3

import os
import json
import base64
import pandas as pd
import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import utils

load_dotenv()
load_dotenv('env.local')

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
print("Using model:", model_name)

prompt_path = os.getenv("PROMPT_PATH", 'Ophthalmic_Scans/prompts/recognize_generation_failures_gpt.json')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

df = pd.read_csv("Ophthalmic_Scans/splits/oct_scan_llm_description_generated_only/test.csv") 
df_outputs = pd.DataFrame(columns=["valid", "description", "image_path"])

base_messages_raw = utils.build_openai_messages_from_prompt(prompt_path)

for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    image_rel_path = row["image_path"]
    full_image_path = os.path.join("Ophthalmic_Scans", image_rel_path)
    
    try:
        base64_image = encode_image(full_image_path)
        
        messages = base_messages_raw.copy()
        messages.append(
            {
                "role": "user",
                "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            }]
            }                  
            )

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500,
        )

        response_content = response.choices[0].message.content
        
        obj = json.loads(response_content)
        
        new_row = {
            "valid": obj.get("valid"),
            "description": obj.get("notes") if obj.get("notes") else obj.get("description"),
            "image_path": image_rel_path
        }

    except Exception as e:
        print(f"Error processing {image_rel_path}: {e}")
        new_row = {"valid": "ERROR", "description": str(e), "image_path": image_rel_path}

    df_outputs = pd.concat([df_outputs, pd.DataFrame([new_row])], ignore_index=True)

utils.save_outputs(df_outputs, model_name)
print("Done.")