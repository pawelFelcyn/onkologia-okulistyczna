#!/usr/bin/env python3

import os
import json
import base64
import copy
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

prompt_path = os.getenv("PROMPT_PATH", 'Ophthalmic_Scans/prompts/recognize_generation_failures_gpt2.json')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

df = pd.read_csv("Ophthalmic_Scans/splits/oct_scan_llm_description_generated_only2/test.csv") 

base_messages_raw = utils.build_openai_messages_from_prompt(prompt_path)
batch_requests = []
jsonl_filename = "batch_input.jsonl"

print("Preparing batch requests...")
for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    image_rel_path = row["image_path"]
    full_image_path = os.path.join("Ophthalmic_Scans", image_rel_path)
    
    try:
        base64_image = encode_image(full_image_path)
        
        messages = copy.deepcopy(base_messages_raw)
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

        batch_request = {
            "custom_id": image_rel_path,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": messages,
                "max_tokens": 500,
            }
        }
        batch_requests.append(batch_request)

    except Exception as e:
        print(f"Error preparing {image_rel_path}: {e}")

with open(jsonl_filename, "w", encoding="utf-8") as f:
    for req in batch_requests:
        f.write(json.dumps(req) + "\n")

print(f"Saved {len(batch_requests)} requests to {jsonl_filename}")

print("Sending batch requests...")
batch_input_file = client.files.create(
  file=open(jsonl_filename, "rb"),
  purpose="batch"
)

print("Creating batch job...")
batch_job = client.batches.create(
  input_file_id=batch_input_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)

batch_job_id = batch_job.id
print(f"Success! Batch Job ID: {batch_job_id}")
print(f"State: {batch_job.status}")

with open("batch_job_info.json", "w", encoding="utf-8") as f:
    json.dump({"batch_job_id": batch_job_id, "status": batch_job.status}, f, indent=4)

print("Job info saved to batch_job_info.json")