import os
import json
import sys
from transformers import pipeline
from PIL import Image

IMAGE_PATH = "images/invoice.jpg"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "result.json")

print("========== Document AI Processing ==========")
if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Image not found: {IMAGE_PATH}")
    sys.exit(1)

print("Loading Document QA model...")
# LayoutLM relies on local layout syntax to answer document queries
qa_pipeline = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa"
)
print("Model loaded successfully!")

print(f"Analyzing structure of: {IMAGE_PATH}...")
image = Image.open(IMAGE_PATH).convert("RGB")

# Targeted keys to extract structured fields from the invoice
queries = {
    "invoice_number": "What is the invoice number?",
    "invoice_date": "What is the invoice date?",
    "total_amount": "What is the total amount due?",
    "vendor_name": "What is the vendor name or company name?"
}

extracted_data = {}

# Interrogate the layout structure sequentially
for key, question in queries.items():
    print(f"Querying: '{question}'...")
    try:
        response = qa_pipeline(image=image, question=question)
        if response and len(response) > 0:
            # Extract highest confidence score result
            top_answer = response[0]
            extracted_data[key] = {
                "answer": top_answer.get("answer", "").strip(),
                "confidence": round(top_answer.get("score", 0), 4)
            }
        else:
            extracted_data[key] = {"answer": None, "confidence": 0.0}
    except Exception as e:
        print(f"Warning: Failed to extract {key}: {str(e)}")
        extracted_data[key] = {"answer": None, "error": str(e)}

# Structuring final JSON payload
result_payload = {
    "file_processed": IMAGE_PATH,
    "structured_data": extracted_data
}

# Ensure directory execution paths align
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(result_payload, f, indent=2)

print(f"\nSuccessfully written structured data metadata to {OUTPUT_FILE}")