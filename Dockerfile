# GPU-ready runtime aligned with the repo's CUDA 12.1 PyTorch install
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app:/opt/app/src \
    HF_HOME=/opt/app/.cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3-dev \
    git \
    wget \
    unzip \
    ca-certificates \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 point to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /opt/app

# Copy requirements first for better layer caching
COPY requirements.txt /opt/app/requirements.txt

# Upgrade pip and install Python dependencies
# Flask is installed explicitly here so inference_app.py works even if
# requirements.txt in the repo is not yet updated in the same commit.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir \
        pytorch_fid==0.3.0 \
        accelerate==0.26.1

# Copy the full repository
COPY . /opt/app

# Download checkpoints and training data if they are not already present
RUN if [ -d /opt/app/data/ckpts ] && [ -d /opt/app/data/train_data ]; then \
        echo "Data directories already present; skipping Zenodo download."; \
    else \
        wget -O /tmp/LatentDiffusion-2DChestCT-CCIA24_data.zip https://zenodo.org/records/19053504/files/LatentDiffusion-2DChestCT-CCIA24_data.zip && \
        unzip -n /tmp/LatentDiffusion-2DChestCT-CCIA24_data.zip -d /opt/app && \
        rm -r /tmp/LatentDiffusion-2DChestCT-CCIA24_data.zip; \
    fi

# Optional OCI labels
LABEL org.opencontainers.image.title="LatentDiffusion-2DChestCT-CCIA24" \
      org.opencontainers.image.description="Docker image for Characterization of Synthetic Lung Nodules in Conditional Latent Diffusion of Chest CT Scans" \
      org.opencontainers.image.source="https://github.com/multimedia-eurecat/LatentDiffusion-2DChestCT-CCIA24"

# Flask app serves on port 7860
EXPOSE 7860

# Default to the web inference app. Override at runtime if needed, e.g.:
# docker run IMAGE train_lidc.py ...
# docker run IMAGE generate_N_images.py ...
ENTRYPOINT ["python3"]
CMD ["inference_app.py"]
