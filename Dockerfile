# Use the official NVIDIA CUDA 12.1 development image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and basic utilities
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set the working directory
WORKDIR /app

# Copy the PopBrain source code
COPY . /app/popbrain/

# Install PopBrain (triggers CUDA JIT/AOT compilation safely inside the container)
RUN cd /app/popbrain && pip install -e .

# Default command
CMD ["python", "-c", "import popbrain; print('PopBrain successfully initialized in Docker!')"]
