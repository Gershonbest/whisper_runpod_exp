# Speech-to-Text Service

A production-ready Speech-to-Text (STT) service built with FastAPI, Whisper, and Pyannote.audio. Supports multi-language transcription and speaker diarization.

## Features

- **Multi-language Support**: Transcribe audio in 100+ languages with auto-detection
- **Speaker Diarization**: Identify and separate different speakers in audio
- **Translation**: Translate non-English transcripts to English
- **Request Queuing**: GPU semaphore limits concurrent requests to avoid server overload; queue status and size endpoints
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **RESTful API**: Clean FastAPI-based API with OpenAPI documentation
- **RunPod Compatible**: Includes serverless handler for RunPod deployment

## Project Structure

```
build_code/src/
├── main.py                 # FastAPI application entry point
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py        # Environment-based settings
├── controllers/            # Route handlers
│   ├── __init__.py
│   ├── health.py          # Health and ping
│   ├── transcription.py  # Transcribe and invocations (with queue)
│   ├── languages.py       # Supported languages
│   └── queue.py           # Queue status and size
├── schemas/                # Request/response models
│   ├── __init__.py
│   ├── requests.py        # Request schemas
│   └── responses.py       # Response schemas
├── services/               # Business logic services
│   ├── __init__.py
│   ├── stt_service.py     # Whisper transcription service
│   ├── diarization_service.py  # Speaker diarization service
│   ├── transcription_service.py # Main orchestration service
│   └── audio_service.py    # Audio processing wrapper
├── models/                 # Model loading
│   ├── __init__.py
│   └── whisper_model.py   # Whisper model loader
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── logger.py          # Logging configuration
│   ├── audio_processing.py # Audio download and processing
│   ├── transcription_utils.py # Transcription helper functions
│   └── languages.py        # Supported languages
└── handlers/              # External handlers
    ├── __init__.py
    └── runpod_handler.py  # RunPod serverless handler
```

## Installation

### Prerequisites

- Python 3.10-3.12
- CUDA-capable GPU (recommended) or CPU
- HuggingFace token (for model access)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd whisper_runpod_exp
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. Set environment variables:
```bash
# Create .env file
HF_TOKEN=your_huggingface_token_here
COMPUTE_RATE_PER_SECOND=0.0007
WHISPER_MODEL_SIZE=distil-large-v3
```

## Configuration

Environment variables can be set in a `.env` file or as system environment variables:

### Required
- `HF_TOKEN`: HuggingFace token for accessing models

### Optional
- `WHISPER_MODEL_SIZE`: Whisper model size (default: `distil-large-v3`)
- `WHISPER_DEVICE`: Device to use (`auto`, `cuda`, `cpu`) - default: `auto`
- `WHISPER_COMPUTE_TYPE`: Compute type (`auto`, `int8`, `float16`) - default: `auto`
- `DIARIZATION_MODEL`: Diarization model (default: `pyannote/speaker-diarization-3.1`)
- `DEFAULT_NUM_SPEAKERS`: Default number of speakers (default: `2`)
- `MAX_CONCURRENCY`: Max concurrent GPU requests (default: `2`). Limits load via a semaphore; excess requests are queued.
- `API_HOST`: API host (default: `0.0.0.0`)
- `API_PORT`: API port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

See `config/settings.py` for all available configuration options.

## Usage

### Running the Service

```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000

# Or using Python
python main.py
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### API Endpoints

#### Health Check
```bash
GET /health
GET /ping
```

#### Queue Status (Queuing)
```bash
GET /queue_status
GET /queue_size
```

See [Request Queuing](#request-queuing) below for details.

#### Transcribe Audio (Synchronous)
```bash
POST /api/v1/transcribe
```

Waits for an available GPU slot (respecting `MAX_CONCURRENCY`), then returns the transcription in the response.

Request body:
```json
{
  "audio_url": "https://example.com/audio.mp3",
  "language": "en",
  "enable_diarization": true,
  "num_speakers": 2,
  "translate_to_english": false,
  "extra_data": {}
}
```

Response:
```json
{
  "text": "Full transcribed text...",
  "diarized_text": "SPEAKER_1: [00:00:00 - 00:00:02] Hello...",
  "language": "EN",
  "duration": 123.45,
  "segments": [...],
  "num_speakers": 2,
  "processing_time": 2.5,
  "cost": 0.00175
}
```

#### Invocations (Asynchronous, Queued)
```bash
POST /invocations
```

Returns immediately with an acceptance message. The request is processed in the background; GPU work is limited by the same semaphore. Use `dispatcher_endpoint` in the body to receive results when done.

Request body: same as `/api/v1/transcribe` (include `dispatcher_endpoint` to get results).

Response (immediate):
```json
{
  "status": "accepted",
  "message": "Request queued for processing",
  "queue_position": 1
}
```

Results are sent to `dispatcher_endpoint` when processing completes (if provided).

### Request Queuing

The service uses a **GPU semaphore** to cap concurrent transcription jobs and avoid overloading the server. All GPU work (sync and async) goes through this limit.

- **`MAX_CONCURRENCY`** (default: `2`): Maximum number of transcription requests running at once. Extra requests wait in a queue.
- **`GET /queue_status`**: Returns queue and GPU usage:
  - `max_concurrency`: Configured limit
  - `available_slots`: Free GPU slots
  - `requests_in_queue`: Requests waiting for a slot
  - `active_requests`: Requests currently processing
- **`GET /queue_size`**: Returns only `requests_in_queue`.

**Sync** (`POST /api/v1/transcribe`): The client waits for a free slot, then receives the full transcription in the response.

**Async** (`POST /invocations`): The client gets `"status": "accepted"` and `queue_position` right away. Processing runs in the background; provide `dispatcher_endpoint` in the body to receive the transcription when it finishes.

### Request Parameters

- `audio_url` (required): URL of the audio file to transcribe
- `language` (optional): Language code (ISO 639-1). Auto-detected if not provided
- `task` (optional): `"transcribe"` or `"translate"` (default: `"transcribe"`)
- `enable_diarization` (optional): Enable speaker diarization (default: `true`)
- `num_speakers` (optional): Number of speakers. Auto-detected if not provided
- `translate_to_english` (optional): Translate non-English to English (default: `false`)
- `extra_data` (optional): Additional metadata dictionary
- `dispatcher_endpoint` (optional): Endpoint to send results to after processing

### Supported Languages

The service supports 100+ languages. See `utils/languages.py` for the complete list. Common languages include:

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Portuguese (pt)
- And many more...

## RunPod Deployment

The service includes a RunPod serverless handler. Use `handlers/runpod_handler.py`:

```python
from handlers import handler

# Handler expects RunPod event format
event = {
    "input": {
        "audio_url": "https://example.com/audio.mp3",
        "language": "en",
        "enable_diarization": True
    }
}

result = handler(event)
```

## Docker Deployment

Build and run with Docker:

```bash
# Build
docker build -t stt-service .

# Run
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token \
  stt-service
```

## Development

### Code Structure

The codebase follows a clean architecture pattern:

- **Config**: Centralized configuration management
- **Schemas**: Pydantic models for request/response validation
- **Services**: Business logic separated by domain
- **Models**: Model loading and management
- **Utils**: Reusable utility functions
- **Handlers**: External integrations (RunPod, etc.)

### Adding New Features

1. **New Service**: Add to `services/` directory
2. **New Endpoint**: Add a controller in `controllers/` and include its router in `main.py`
3. **New Schema**: Add to `schemas/` directory
4. **Configuration**: Add to `config/settings.py`

### Testing

```bash
# Run tests (when available)
pytest

# Test API locally
curl -X POST http://localhost:8080/api/v1/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/audio.mp3"}'
```

## Performance

- **GPU Recommended**: Significantly faster on CUDA-capable GPUs
- **Model Size**: Larger models (e.g., `large-v3`) provide better accuracy but slower inference
- **Diarization**: Adds ~20-30% processing time
- **Request Queuing**: `MAX_CONCURRENCY` (default 2) limits concurrent GPU jobs so the server does not overload; use `/queue_status` and `/queue_size` to monitor the queue

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check `HF_TOKEN` is set correctly
   - Verify internet connection for model download
   - Check GPU/CUDA availability

2. **Out of Memory**
   - Use smaller model size
   - Reduce `num_speakers` for diarization
   - Use CPU instead of GPU

3. **Audio Download Fails**
   - Check audio URL is accessible
   - Verify network connectivity
   - Check audio format is supported

4. **Requests Queued or Slow**
   - Check `GET /queue_status` for `requests_in_queue` and `active_requests`
   - Increase `MAX_CONCURRENCY` (e.g. to 4) if the GPU can handle more; lower it if the server is overloaded