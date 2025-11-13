# Use NVIDIA PyTorch container for GPU support
FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

# Launch FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
