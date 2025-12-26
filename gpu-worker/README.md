# Hunyuan3D GPU Worker for RunPod

This directory contains the RunPod Serverless worker that runs Hunyuan3D on GPU.

## Deployment

### 1. Build Docker Image

```bash
docker build -t hunyuan3d-worker:latest .
```

### 2. Push to Docker Hub

```bash
docker tag hunyuan3d-worker:latest your-dockerhub/hunyuan3d-worker:latest
docker push your-dockerhub/hunyuan3d-worker:latest
```

### 3. Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `hunyuan3d-generator`
   - **Docker Image**: `your-dockerhub/hunyuan3d-worker:latest`
   - **GPU Type**: A100 40GB (recommended) or L4
   - **Min Workers**: 0 (scale to zero)
   - **Max Workers**: 5
   - **Idle Timeout**: 30 seconds
   - **Flash Boot**: Enabled
4. Copy the Endpoint ID

### 4. Configure Actor

Set these environment variables in your Apify Actor:

```
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
```

## API

### Input

```json
{
  "input": {
    "prompt": "a detailed dragon figurine",
    "steps": 30,
    "guidance_scale": 7.5,
    "octree_depth": 8,
    "output_format": "glb"
  }
}
```

Or with image:

```json
{
  "input": {
    "image_url": "https://example.com/image.png",
    "steps": 30,
    "guidance_scale": 7.5,
    "octree_depth": 8
  }
}
```

### Output

```json
{
  "glb_base64": "...",
  "generation_time": 45.2,
  "vertices": 12345,
  "faces": 24680,
  "file_size_bytes": 1234567
}
```

## Quality Settings

| Quality | Steps | Guidance | Octree Depth | ~Time | ~Cost |
|---------|-------|----------|--------------|-------|-------|
| Lite | 20 | 5.0 | 7 | ~15s | $0.01 |
| Standard | 30 | 7.5 | 8 | ~30s | $0.02 |
| High | 50 | 10.0 | 9 | ~60s | $0.04 |

## Cost Estimate

- A100 40GB: $0.40/hour = $0.0067/min
- Average generation: 30 seconds = **~$0.02 per model**

With scale-to-zero, you only pay for actual generation time.
