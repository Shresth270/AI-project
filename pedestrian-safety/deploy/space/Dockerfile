# Crowd Safety Console - container for Hugging Face Spaces (free CPU: 2 vCPU, 16 GB RAM)
FROM python:3.11-slim

# libraries OpenCV (used by YOLO) needs
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the app as user 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    HOST=0.0.0.0 \
    PORT=7860
WORKDIR /code

# CPU-only PyTorch (about 200 MB instead of 2.5 GB for the GPU build)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

COPY --chown=user . .
USER user
# download the pretrained backbones + YOLO now, so the first photo check is fast
RUN python warmup.py

EXPOSE 7860
# one worker (the models are shared and big), threads for the live agent stream
CMD ["gunicorn", "--chdir", "app", "--workers", "1", "--worker-class", "gthread", "--threads", "8", \
     "--timeout", "600", "--bind", "0.0.0.0:7860", "app:app"]
