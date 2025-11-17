# app.py
import base64
from io import BytesIO
from fastapi import FastAPI, Request, File, UploadFile
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

app = FastAPI(title="OlmOCR API")

MODEL_NAME = "allenai/olmOCR-2-7B-1025"

print("Loading model... (this may take several minutes)")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# -----------------------
# Helper: load image file
# -----------------------
def load_image(upload_file: UploadFile):
    img_bytes = upload_file.file.read()
    return Image.open(BytesIO(img_bytes)).convert("RGB")


# -----------------------
# OCR endpoint
# -----------------------
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    img = load_image(image)

    inputs = processor(images=img, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=512)
    text = processor.decode(output[0], skip_special_tokens=True)

    return {"text": text}


# -----------------------
# Multimodal generation
# -----------------------
@app.post("/generate")
async def generate(
    request: Request,
    image: UploadFile = File(None)
):
    body = await request.json()
    prompt = body.get("prompt", "")

    if image:
        img = load_image(image)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
    else:
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=512
    )

    response = processor.decode(output[0], skip_special_tokens=True)
    return {"response": response}
