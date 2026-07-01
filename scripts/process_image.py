import os
import json
import base64
import requests
import sys

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
IMAGE_PATH = "images/invoice.jpg"

print("========== AI Image Processing ==========")
print(f"Ollama URL: {OLLAMA_URL}")
print(f"Image Path: {IMAGE_PATH}")

# Check image exists
if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Image not found: {IMAGE_PATH}")
    sys.exit(1)

print("Reading image...")

with open(IMAGE_PATH, "rb") as f:
    image = base64.b64encode(f.read()).decode()

print("Sending request to Ollama...")

payload = {
    "model": "llama3.2-vision",
    "stream": False,
    "messages": [
        {
            "role": "user",
            "content": """
Describe this image.

Return ONLY valid JSON.

{
    "description":"",
    "summary":""
}
""",
            "images": [image]
        }
    ]
}

response = requests.post(
    f"{OLLAMA_URL}/api/chat",
    json=payload
)

print("\n========== RESPONSE ==========")
print("Status Code:", response.status_code)
print(response.text)

if response.status_code != 200:
    sys.exit(1)

result = response.json()

os.makedirs("output", exist_ok=True)

with open("output/result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nSaved output/result.json")