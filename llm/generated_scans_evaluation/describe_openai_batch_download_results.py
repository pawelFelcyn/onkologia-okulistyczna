#!/usr/bin/env python3

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import utils
import sys

load_dotenv()
load_dotenv('env.local')

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

ORIGINAL_CSV_PATH = "Ophthalmic_Scans/splits/oct_scan_llm_description_generated_only2/test.csv"
BATCH_INFO_FILE = "batch_job_info.json"

try:
    with open(BATCH_INFO_FILE, "r") as f:
        job_info = json.load(f)
        batch_id = job_info.get("batch_job_id")
        if not batch_id:
            raise ValueError(f"No 'batch_job_id' found in {BATCH_INFO_FILE}")
except FileNotFoundError:
    print(f"Error: {BATCH_INFO_FILE} not found. Please run the submission script first.")
    sys.exit(1)

print(f"Checking status for Batch ID: {batch_id}...")
batch = client.batches.retrieve(batch_id)

if batch.status == "failed":
    print(f"Batch job failed. Errors: {batch.errors}")
    sys.exit(1)
elif batch.status in ["in_progress", "validating", "finalizing"]:
    print(f"Batch job is currently: {batch.status}")
    print("Please try again later. The results are not ready yet.")
    sys.exit(0)
elif batch.status == "completed":
    print("Batch job completed! Downloading results...")
elif batch.status == "expired":
    print("Batch job expired.")
    sys.exit(1)
elif batch.status == "cancelled":
    print("Batch job was cancelled.")
    sys.exit(1)

if batch.output_file_id:
    file_response = client.files.content(batch.output_file_id)
    results_content = file_response.text
else:
    print("Batch completed, but no output file ID returned.")
    sys.exit(1)

results_map = {}

print("Parsing batch results...")
for line in results_content.strip().split('\n'):
    if not line.strip():
        continue
        
    res = json.loads(line)
    custom_id = res['custom_id']
    
    if res.get('response', {}).get('status_code') == 200:
        try:
            response_body = res['response']['body']
            choice = response_body['choices'][0]
            content_str = choice['message']['content']
            
            parsed_content = json.loads(content_str)
            
            results_map[custom_id] = {
                "valid": parsed_content.get("valid"),
                "description": parsed_content.get("notes") if parsed_content.get("notes") else parsed_content.get("description"),
                "image_path": custom_id
            }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for {custom_id}: {e}")
            results_map[custom_id] = {
                "valid": "ERROR",
                "description": f"JSON parse error: {str(e)}",
                "image_path": custom_id
            }
        except Exception as e:
            print(f"Unexpected error processing {custom_id}: {e}")
            results_map[custom_id] = {
                "valid": "ERROR",
                "description": f"Processing error: {str(e)}",
                "image_path": custom_id
            }
    else:
        error_msg = f"API Error: {res.get('response', {}).get('status_code')}"
        print(f"Request failed for {custom_id}: {error_msg}")
        results_map[custom_id] = {
            "valid": "ERROR",
            "description": error_msg,
            "image_path": custom_id
        }

print(f"Loading original CSV from {ORIGINAL_CSV_PATH} to restore order...")
try:
    df_original = pd.read_csv(ORIGINAL_CSV_PATH)
except FileNotFoundError:
    print(f"Error: Original CSV file not found at {ORIGINAL_CSV_PATH}")
    sys.exit(1)

final_rows = []

for _, row in df_original.iterrows():
    image_rel_path = row["image_path"]
    
    if image_rel_path in results_map:
        final_rows.append(results_map[image_rel_path])
    else:
        print(f"Warning: No result found for {image_rel_path}")
        final_rows.append({
            "valid": "MISSING",
            "description": "Result not found in batch output",
            "image_path": image_rel_path
        })

df_outputs = pd.DataFrame(final_rows)

print(f"Saving {len(df_outputs)} results...")
utils.save_outputs(df_outputs, model_name)
print("Done.")