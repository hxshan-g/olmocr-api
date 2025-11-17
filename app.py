import base64
from io import BytesIO
from fastapi import FastAPI, Request, File, UploadFile,Form
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

app = FastAPI(title="OlmOCR API")

MODEL_NAME = "allenai/olmOCR-2-7B-1025"

# -----------------------
# Choose device
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model on: {device} with dtype={dtype}")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
)

model = model.to(device)
model.eval()

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

    # Pass empty text for OCR
    inputs = processor(text="", images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(**inputs, max_new_tokens=512)
    text = processor.decode(output[0], skip_special_tokens=True)

    return {"text": text}


# -----------------------
# Multimodal generation
# -----------------------
from fastapi import UploadFile, File, Form

@app.post("/generate")
async def generate(
    image: UploadFile = File(None),
    prompt: str = Form("")  # Get prompt from form-data instead of JSON
):
    try:
        print(f"[DEBUG] Original prompt: {prompt}")
        print(f"[DEBUG] Image provided: {bool(image)}")

        # Auto-inject <image> placeholder if image exists but prompt doesn't have it
        if image and "<image>" not in prompt:
            prompt = "<image> " + (prompt or "Describe the image.")
            print(f"[DEBUG] Modified prompt with <image>: {prompt}")

        # Load and process image
        if image:
            img = load_image(image)
            print(f"[DEBUG] Loaded image size: {img.size}")
            inputs = processor(text=prompt, images=img, return_tensors="pt")
        else:
            inputs = processor(text=prompt, return_tensors="pt")

        # Move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        for k, v in inputs.items():
            print(f"[DEBUG] Input tensor {k} shape: {v.shape}")

        # Generate output
        output = model.generate(**inputs, max_new_tokens=512)
        print(f"[DEBUG] Generated raw output length: {len(output[0])}")

        # Decode response
        response = processor.decode(output[0], skip_special_tokens=True)
        print(f"[DEBUG] Decoded response: {response[:200]}...")  # first 200 chars

        return {"response": response}

    except Exception as e:
        # Catch all exceptions with traceback for better debugging
        import traceback
        print(f"[ERROR] Exception occurred: {e}")
        traceback.print_exc()
        return {"error": str(e)}
