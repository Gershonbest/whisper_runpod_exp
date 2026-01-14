# RunPod Serverless Deployment Guide

This guide explains how to deploy the Whisper transcription service to RunPod serverless.

## Handler File

The main handler file is `build_code/src/runpod_handler.py`. This file contains the `handler` function that RunPod will call for each request.

## Deployment Steps

### 1. Prepare Your Code

Ensure all your code is in the repository, including:
- `build_code/src/runpod_handler.py` (the handler)
- All other source files in `build_code/src/`
- `pyproject.toml` (for dependencies)

### 2. Create a Dockerfile for RunPod

RunPod serverless requires a specific Dockerfile structure. You can use the existing Dockerfile or create a RunPod-specific one.

**Key requirements for RunPod:**
- The handler function must be importable
- Set the handler path in RunPod dashboard: `runpod_handler.handler`
- Ensure all dependencies are installed

### 3. Configure RunPod Serverless

1. Go to RunPod dashboard
2. Create a new serverless endpoint
3. Upload your Docker image or connect your repository
4. Set the handler path to: `runpod_handler.handler`
5. Configure environment variables if needed (e.g., `HF_TOKEN`)

### 4. Environment Variables

Make sure to set these environment variables in RunPod:
- `HF_TOKEN`: Your HuggingFace token (if required for model access)

### 5. Test the Endpoint

Use the RunPod API or dashboard to test your endpoint with a sample request:

```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "extra_data": {},
    "is_ml_agent": false,
    "dispatcher_endpoint": "https://your-dispatcher.com/api"
  }
}
```

## Request Format

### Audio Transcription Request

```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "extra_data": {
      "custom_field": "value"
    },
    "dispatcher_endpoint": "https://your-dispatcher.com/api"
  }
}
```

### ML Agent Request

```json
{
  "input": {
    "transcript": [
      ["speaker1", "Hello, how are you?"],
      ["speaker2", "I'm doing well, thank you!"]
    ],
    "is_ml_agent": true,
    "extra_data": {
      "languages": {
        "defaultLanguage": "en"
      }
    },
    "dispatcher_endpoint": "https://your-dispatcher.com/api"
  }
}
```

## Response Format

### Success Response

```json
{
  "data": {
    "text": "Transcribed text...",
    "diarized_transcript": "Speaker 1: Text...",
    "translation": "Translated text...",
    "diarized_translation": "Speaker 1: Translated...",
    "duration": 123.45,
    "language": "en",
    "extra_data": {
      "billing": {
        "taskDuration": 123,
        "taskCost": 0.0861
      }
    }
  }
}
```

### Error Response

```json
{
  "error": "Error message here",
  "error_type": "ValueError"
}
```

## Local Testing

You can test the handler locally before deploying:

```bash
cd build_code/src
python runpod_handler.py
```

Or test programmatically:

```python
from runpod_handler import handler

test_event = {
    "input": {
        "audio_url": "https://example.com/test.mp3",
        "extra_data": {},
        "is_ml_agent": False
    }
}

result = handler(test_event)
print(result)
```

## Notes

- The handler automatically handles both audio transcription and ML agent requests
- Make sure your audio URLs are publicly accessible or the handler has network access
- The dispatcher_endpoint is optional - if not provided, the handler will still process but won't send results to a dispatcher
- GPU support is available in RunPod - ensure your Dockerfile uses the appropriate CUDA base image

