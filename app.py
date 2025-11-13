import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import json
import time

app = FastAPI(title="olmOCR API", description="OpenAI-compatible API for allenai/olmOCR-2-7B-1025")

MODEL_ID = "allenai/olmOCR-2-7B-1025"

print("Loading model... (first load can take several minutes)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    max_tokens: int = 512
    temperature: float = 0.7

@app.get("/v1/models")
def list_models():
    return {"data": [{"id": MODEL_ID}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    user_text = "\n".join([f"{m.role}: {m.content}" for m in req.messages]) + "\nassistant:"

    inputs = tokenizer(user_text, return_tensors="pt").to(model.device)

    if req.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def event_stream():
            for token in streamer:
                data = {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "delta": {"content": token},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming response
    output = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature
    )
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = response_text.split("assistant:")[-1].strip()

    return JSONResponse({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }
        ],
        "model": req.model
    })
