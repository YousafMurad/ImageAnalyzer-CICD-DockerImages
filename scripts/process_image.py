import base64
import json
import requests

IMAGE_PATH = "images/invoice.jpg"

with open(IMAGE_PATH, "rb") as f:
    image = base64.b64encode(f.read()).decode()

payload = {
    "model": "llama3.2-vision",
    "stream": False,
    "messages": [
        {
            "role": "user",
            "content": """
Describe this image.

Return JSON only.

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
    "http://localhost:11434/api/chat",
    json=payload,
)

response.raise_for_status()

data = response.json()

print(data)

with open("output/result.json", "w") as f:
    json.dump(data, f, indent=2)