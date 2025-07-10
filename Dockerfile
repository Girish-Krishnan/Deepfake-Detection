# Use NVIDIA CUDA base image with support for Ubuntu 20.04 and development tools
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables to make apt-get installations non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set the working directory
WORKDIR /app

# Install Python 3.9, essential build tools, and development headers in one step
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3.9-distutils wget build-essential git \
    libgl1-mesa-glx libglib2.0-0 cmake libopenblas-dev libboost-all-dev \
    nano vim tmux \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
# Clone Wavelet-CLIP repository
RUN git clone https://github.com/Girish-Krishnan/Wavelet-CLIP.git /app/Wavelet_CLIP

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install required Python dependencies in one step
RUN pip install numpy==1.21.5 pandas==1.4.2 Pillow==9.0.1 imageio==2.9.0 imgaug==0.4.0 tqdm==4.61.0 scipy==1.7.3 \
    seaborn==0.11.2 pyyaml==6.0 imutils==0.5.4 opencv-python==4.6.0.66 scikit-image==0.19.2 scikit-learn==1.0.2 \
    albumentations==1.1.0 timm==0.6.12 einops transformers simplejson \
    git+https://github.com/openai/CLIP.git torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install pytorch_wavelets for wavelet-based operations
RUN pip install pytorch_wavelets

# Verify nvcc is available
RUN nvcc --version
