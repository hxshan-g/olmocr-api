FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System setup
RUN apt update && apt install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .

EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
