import os
import json
from transformers import pipeline
from PIL import Image

IMAGE_PATH = "images/invoice.jpg"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "result.json")

print("========== AI Image Processing ==========")
if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Image not found: {IMAGE_PATH}")
    import sys
    sys.exit(1)

print("Loading image captioning model...")
# Updated task to 'image-text-to-text' to match latest Transformers registry
pipe = pipeline(
    "image-text-to-text",
    model="Salesforce/blip-image-captioning-base"
)
print("Model loaded successfully!")

print(f"Processing image: {IMAGE_PATH}...")
image = Image.open(IMAGE_PATH).convert("RGB")

# For image-text-to-text, pass the image and keep text prompt empty for raw captioning
raw_results = pipe(image, text="")

# Format into structured JSON
result_data = {
    "file_processed": IMAGE_PATH,
    "predictions": raw_results,
    "summary": raw_results[0].get("generated_text", "") if raw_results else ""
}

# Ensure output directory exists and write results
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(result_data, f, indent=2)

print(f"Successfully saved results to {OUTPUT_FILE}")