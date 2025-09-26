# VLLM Server Launcher

This launcher starts a VLLM server that serves LLM models via HTTP API for the RefineEA pipeline.

## Setup

1. **Copy environment file**:
   ```bash
   cp ".env copy" .env
   ```

2. **Edit `.env`** with your configuration:
   - `MODEL_NAME`: Model to load (default: `HuggingFaceTB/SmolLM2-135M-Instruct`)
   - `DGX_OPEN_PORT`: External port (default: `1444`)
   - `HF_TOKEN`: Your HuggingFace token

## Usage

### Using SLURM (Recommended)
```bash
sbatch sbatch.sh
```

### Using Docker Compose
```bash
docker-compose up
```

## API Endpoint

The server exposes an OpenAI-compatible API at:
- **URL**: `http://172.16.40.54:1444/v1/chat/completions`
- **Method**: POST
- **Content-Type**: application/json

## Example Request

```bash
curl -X POST "http://172.16.40.54:1444/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## Configuration

- **Port**: 8000 (internal) → 1444 (external)
- **Model**: HuggingFaceTB/SmolLM2-135M-Instruct
- **GPU Memory**: 98% utilization
- **Cache**: `/raid/ml-data/hf_cache`

## Troubleshooting

### Common Issues

1. **Docker networking error**: "all predefined address pools have been fully subnetted"
   - **Solution**: The script automatically cleans up existing containers and networks
   - **Manual fix**: `docker system prune -f` (removes unused networks)

2. **Server not starting**:
   - Check logs: `tail -f output.log` or `tail -f error.log`
   - Verify GPU availability: `nvidia-smi`
   - Check if port is in use: `netstat -tulpn | grep 1444`

3. **Permission issues**:
   - Ensure you're in the docker group: `groups`
   - Check user permissions: `id`

### Testing the Server

Run the test script to verify the server is working:
```bash
chmod +x test_server.sh
./test_server.sh
```

### Manual Debugging

1. **Check container status**:
   ```bash
   docker ps | grep vllm_server
   ```

2. **Check container logs**:
   ```bash
   docker logs ${SERVER_REFINE_EA_CONTAINER_NAME}
   ```

3. **Test API manually**:
   ```bash
   curl -X POST "http://172.16.40.54:1444/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{"model":"HuggingFaceTB/SmolLM2-135M-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":10}'
   ``` 