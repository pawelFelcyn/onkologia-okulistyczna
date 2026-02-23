#!/usr/bin/env python3

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import pandas as pd
import json
from dotenv import load_dotenv
import os
import tqdm
import copy
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
load_dotenv('env.local')

model_name = "Qwen/Qwen2-VL-7B-Instruct"

PROMPT_PATH_A = os.getenv("PROMPT_PATH_A", 'Ophthalmic_Scans/prompts/recognize_generation_failures_qwen3.json')
PROMPT_PATH_B = os.getenv("PROMPT_PATH_B", 'Ophthalmic_Scans/prompts/recognize_generation_failures_qwen4.json')

MAX_DEBATE_ROUNDS = os.getenv("MAX_DEBATE_ROUNDS", 3)

DEBATE_INJECTION = """
Disagreement detected. Another analysis of this specific image concluded: {other_verdict}.
Their reasoning: "{other_notes}"

Please re-examine the image specifically checking for the issues (or lack thereof) mentioned above. 
- If you missed a structural break or a subtle feature, correct your verdict.
- If you are confident they are wrong, explain why.

Respond strictly in the same JSON format: {{"valid": boolean, "confidence": float, "notes": "..."}}.
"""

print(f"Loading model: {model_name}")

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

def get_model_response(messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response_text

def parse_json_response(text):
    try:
        clean_text = text
        if "```json" in text:
            clean_text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            clean_text = text.split("```")[1].split("```")[0]
        
        obj = json.loads(clean_text.strip())
        return obj
    except Exception:
        return None


base_messages_A = utils.build_messages_from_prompt(PROMPT_PATH_A)
base_messages_B = utils.build_messages_from_prompt(PROMPT_PATH_B)

df = pd.read_csv("Ophthalmic_Scans/splits/oct_scan_llm_description/test.csv") 
df_outputs = pd.DataFrame(columns=["valid", "description", "image_path", "method", "rounds"])

print("Starting analysis loop...")

for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    image_url = row["image_path"]
    full_image_path = os.path.join("Ophthalmic_Scans", image_url)
    
    msgs_A = copy.deepcopy(base_messages_A)
    msgs_B = copy.deepcopy(base_messages_B)
    
    new_user_msg = {
        "role": "user",
        "content": [{"type": "image", "image": full_image_path}]
    }
    msgs_A.append(new_user_msg)
    msgs_B.append(new_user_msg)

    resp_A = get_model_response(msgs_A)
    msgs_A.append({"role": "assistant", "content": resp_A})
    obj_A = parse_json_response(resp_A)

    resp_B = get_model_response(msgs_B)
    msgs_B.append({"role": "assistant", "content": resp_B})
    obj_B = parse_json_response(resp_B)

    final_obj = None
    method_used = "direct"
    rounds_done = 0

    valid_A = obj_A.get("valid") if obj_A else None
    valid_B = obj_B.get("valid") if obj_B else None

    if obj_A and obj_B:
        if valid_A == valid_B:
            final_obj = obj_A 
        else:
            method_used = "debate"
            
            consensus = False
            for r in range(MAX_DEBATE_ROUNDS):
                rounds_done += 1
                
                verdict_str_B = "VALID" if obj_B.get("valid") else "INVALID"
                debate_prompt_A = DEBATE_INJECTION.format(
                    other_verdict=verdict_str_B,
                    other_notes=obj_B.get("notes", "No notes provided.")
                )
                msgs_A.append({"role": "user", "content": debate_prompt_A})
                
                resp_A = get_model_response(msgs_A)
                msgs_A.append({"role": "assistant", "content": resp_A})
                obj_A = parse_json_response(resp_A)

                if obj_A:
                    verdict_str_A = "VALID" if obj_A.get("valid") else "INVALID"
                    notes_A = obj_A.get("notes", "")
                else:
                    verdict_str_A = "UNKNOWN (JSON Error)"
                    notes_A = "Error parsing response"

                debate_prompt_B = DEBATE_INJECTION.format(
                    other_verdict=verdict_str_A,
                    other_notes=notes_A
                )
                msgs_B.append({"role": "user", "content": debate_prompt_B})
                
                resp_B = get_model_response(msgs_B)
                msgs_B.append({"role": "assistant", "content": resp_B})
                obj_B = parse_json_response(resp_B)

                if obj_A and obj_B:
                    if obj_A.get("valid") == obj_B.get("valid"):
                        consensus = True
                        final_obj = obj_A
                        final_obj["notes"] = f"[Consensus after {rounds_done} rounds] " + final_obj.get("notes", "")
                        break
            
            if not consensus:
                is_valid = False 
                notes = f"[NO CONSENSUS] Model A said {obj_A.get('valid')}, Model B said {obj_B.get('valid')}. Defaulting to Invalid."
                final_obj = {"valid": is_valid, "notes": notes}
                method_used = "debate_failed"

    else:
        final_obj = {"valid": "ERROR", "notes": "JSON Parsing Error in initial pass"}

    new_row = {
        "valid": final_obj.get("valid"),
        "description": final_obj.get("notes"),
        "image_path": image_url,
        "method": method_used,
        "rounds": rounds_done
    }
    
    df_outputs = pd.concat([df_outputs, pd.DataFrame([new_row])], ignore_index=True)

utils.save_outputs(df_outputs, model_name)
print("Done.")