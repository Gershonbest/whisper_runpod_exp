# Whisper GPU Worker

A FastAPI-based GPU worker that polls a Redis queue and processes transcription jobs with GPU concurrency control. Built with Whisper and Pyannote.audio for multi-language transcription and speaker diarization.

## Features

- **Multi-language Support**: Transcribe audio in 100+ languages with auto-detection
- **Speaker Diarization**: Identify and separate different speakers in audio
- **Translation**: Translate non-English transcripts to English
- **GPU Concurrency Control**: Configurable max concurrent jobs to optimize GPU usage
- **Redis Queue**: Polls Redis queue for transcription jobs

## Requirements

- Python 3.10+
- Docker (optional)
- Redis instance (local or hosted)
- CUDA-capable GPU
- HuggingFace token (for model access)

## Local Development

### Install dependencies

```bash
uv sync
```

### Configure environment

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
REDIS_URL=redis://<username>:<password>@<host>:<port>/0
QUEUE_NAME=transcription_queue
MAX_CONCURRENCY=5
API_PORT=8000
WHISPER_MODEL_SIZE=distil-large-v3
```

### Run locally

```bash
cd build_code/src
python main.py
```

Or from project root:

```bash
python build_code/src/main.py
```

## Docker

### Build image

```bash
docker build -t whisper-gpu-worker:latest .
```

### Run container

```bash
docker run -p 8000:8000 --gpus all \
  -e HF_TOKEN=your_huggingface_token \
  -e REDIS_URL=redis://<username>:<password>@<host>:<port>/0 \
  -e MAX_CONCURRENCY=5 \
  -e QUEUE_NAME=transcription_queue \
  whisper-gpu-worker:latest
```

## API Endpoints

| Endpoint       | Method | Description                          |
|----------------|--------|--------------------------------------|
| `/health`      | GET    | Health check, returns worker status  |
| `/queue_size`  | GET    | Returns current queue length         |

## Environment Variables

| Variable              | Default                      | Description                              |
|-----------------------|------------------------------|------------------------------------------|
| `HF_TOKEN`            | -                            | HuggingFace token for model access (required) |
| `REDIS_URL`           | `redis://localhost:6379/0`   | Full Redis connection URL with auth      |
| `QUEUE_NAME`          | `transcription_queue`        | Redis list name to consume jobs from     |
| `MAX_CONCURRENCY`     | `5`                          | Max parallel jobs processed              |
| `API_PORT`            | `8000`                       | HTTP server port                         |
| `WHISPER_MODEL_SIZE`  | `distil-large-v3`            | Whisper model size                       |
| `WHISPER_DEVICE`      | `auto`                       | Device to use (`auto`, `cuda`, `cpu`)    |
| `WHISPER_COMPUTE_TYPE`| `auto`                       | Compute type (`auto`, `int8`, `float16`) |
| `DIARIZATION_MODEL`   | `pyannote/speaker-diarization-3.1` | Diarization model           |
| `DEFAULT_NUM_SPEAKERS`| `2`                          | Default number of speakers               |
| `LOG_LEVEL`           | `INFO`                       | Logging level                            |

## Job Payload Format

Push jobs to the Redis queue as JSON:

```json
{
  "job_id": "unique-job-id",
  "request": {
    "audio_url": "https://example.com/audio.mp3",
    "language": "en",
    "task": "transcribe",
    "enable_diarization": true,
    "num_speakers": 2,
    "translate_to_english": false
  }
}
```

### Request Fields

| Field                  | Required | Description                                      |
|------------------------|----------|--------------------------------------------------|
| `audio_url`            | Yes      | URL of the audio file to transcribe              |
| `language`             | No       | Language code (ISO 639-1). Auto-detected if not provided |
| `task`                 | No       | `"transcribe"` or `"translate"` (default: `"transcribe"`) |
| `enable_diarization`   | No       | Enable speaker diarization (default: `true`)     |
| `num_speakers`         | No       | Number of speakers. Auto-detected if not provided |
| `translate_to_english` | No       | Translate non-English to English (default: `false`) |

### Response Format

```json
{
  "text": "Full transcribed text...",
  "diarized_text": "SPEAKER_1: [00:00:00 - 00:00:02] Hello...",
  "language": "EN",
  "duration": 123.45,
  "segments": [...],
  "num_speakers": 2,
  "processing_time": 2.5
}
```

## Project Structure

```
build_code/src/
├── main.py                 # FastAPI application entry point
├── config/
│   └── settings.py         # Environment-based settings
├── controllers/
│   ├── health.py           # Health check endpoints
│   ├── transcription.py    # Transcription endpoint
│   └── languages.py        # Supported languages
├── schemas/
│   ├── requests.py         # Request schemas
│   └── responses.py        # Response schemas
├── services/
│   ├── stt_service.py      # Whisper transcription service
│   ├── diarization_service.py  # Speaker diarization service
│   ├── transcription_service.py # Main orchestration service
│   └── audio_service.py    # Audio processing wrapper
├── models/
│   └── whisper_model.py    # Whisper model loader
└── utils/
    ├── logger.py           # Logging configuration
    ├── audio_processing.py # Audio download and processing
    ├── transcription_utils.py # Transcription helper functions
    └── languages.py        # Supported languages
```

## Supported Languages

The service supports 100+ languages including: English, Spanish, French, German, Chinese, Japanese, Korean, Portuguese, and many more. See `utils/languages.py` for the complete list.

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check `HF_TOKEN` is set correctly
   - Verify internet connection for model download
   - Check GPU/CUDA availability

2. **Out of Memory**
   - Use smaller model size
   - Reduce `MAX_CONCURRENCY`
   - Reduce `num_speakers` for diarization

3. **Audio Download Fails**
   - Check audio URL is accessible
   - Verify network connectivity
   - Check audio format is supported

4. **Redis Connection Issues**
   - Verify `REDIS_URL` is correct
   - Check Redis server is running
   - Verify network connectivity to Redis
