import json
import os
from PIL import Image
import pandas as pd
import base64
import io

def _get_content(prompt_json):
    messages = []
    for message in prompt_json["messages"]:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                path = os.path.join("Ophthalmic_Scans", item["image"])
                image = Image.open(path).convert("RGB")
                content.append({"type": "image", "image": image})
        messages.append({
            "role": message["role"],
            "content": content
        })
    return messages

def build_messages_from_prompt(prompt_path: str) -> list:
    with open(prompt_path, 'r') as file:
        prompt = json.load(file)
    return _get_content(prompt)

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
    
def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def build_openai_messages_from_prompt(prompt_path: str) -> list:
    with open(prompt_path, 'r') as file:
        prompt_json = json.load(file)
    
    openai_messages = []
    
    for message in prompt_json["messages"]:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({
                    "type": "text", 
                    "text": item["text"]
                })
            elif item["type"] == "image":
                path = os.path.join("Ophthalmic_Scans", item["image"])
                image = Image.open(path).convert("RGB")
                base64_str = encode_image_to_base64(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_str}",
                        "detail": "high"
                    }
                })
        
        openai_messages.append({
            "role": message["role"],
            "content": content
        })
        
    return openai_messages