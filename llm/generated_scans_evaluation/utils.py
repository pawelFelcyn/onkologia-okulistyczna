import json
import os
from PIL import Image

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