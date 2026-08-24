# Docker Usage Instructions for FineWebSecurity

## Building the Docker Image

From `data/FineWebSecurity`, build the Docker image:

```bash
docker build -t fineweb-security .
```

## Running the Container

To run the container with GPU support:

```bash
docker run --gpus all \
  -e HF_TOKEN=your_huggingface_token \
  -v /path/to/data:/app/data \
  -v /path/to/outputs:/app/outputs \
  -v /path/to/huggingface_cache:/app/huggingface \
  fineweb-security
```

### Environment Variables

- `HF_TOKEN`: Your HuggingFace token for accessing models and datasets

### Volume Mounts

- `/app/outputs`: Mount a local directory to store outputs
- `/app/huggingface`: Mount a local directory to store HuggingFace cache (models, datasets)

### Specifying Different Years

To filter a specific year other than the default (2023):

```bash
docker run --gpus all \
  -e HF_TOKEN=your_huggingface_token \
  -v /path/to/data:/app/data \
  -v /path/to/outputs:/app/outputs \
  fineweb-security /app/scripts/filter_fineweb_bert.sh 2024
```

## Accessing the Container

To access the running container for debugging:

```bash
docker exec -it container_id /bin/bash
```

Replace `container_id` with the actual container ID from `docker ps`.
