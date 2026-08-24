# Use NVIDIA CUDA 12 as base image to support GPU operations
#FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    tmux \
    vim \
    bzip2 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Run conda init for bash
RUN /opt/conda/bin/conda init bash

# Set the shell to bash
SHELL ["/bin/bash", "-c"]

# Create the conda environment
RUN conda create -n huggingface python=3.10 -y

# Set up conda environment activation in bash
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate huggingface" >> ~/.bashrc

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install dependencies in the conda environment
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate huggingface && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Make script executable
RUN chmod +x /app/scripts/filter_fineweb_bert.sh \
    && chmod +x /app/scripts/parallel_lib.sh

# Create a directory for HuggingFace Home
RUN mkdir -p /app/huggingface/

# Set HuggingFace cache directory
ENV HF_HOME="/app/huggingface/"

# Create an entrypoint script that properly activates conda
# Use echo -e to interpret newline characters (\n) correctly
RUN echo -e '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate huggingface\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]


# Default command (can be overridden at runtime)
CMD ["/app/scripts/filter_fineweb_bert.sh"]
