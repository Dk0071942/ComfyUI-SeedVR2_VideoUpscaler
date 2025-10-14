FROM python:3.11-slim

# Prevents creation of .pyc files and forces stdout/stderr to flush immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies required at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying the full source tree to leverage Docker layer caching.
COPY requirements.txt .

# Torch wheels differ between CPU/GPU builds, so allow overriding the index at build time.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_PACKAGES="torch torchvision torchaudio"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir ${TORCH_PACKAGES} --index-url "${TORCH_INDEX_URL}" && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY . .

# Default auth configuration; override via environment in production.
ENV GRADIO_AUTH_ENABLED="false" \
    GRADIO_AUTH_USERNAME="" \
    GRADIO_AUTH_PASSWORD="" \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    PORT="7860"

EXPOSE 7860

CMD ["python", "app_video_upscaler.py"]
