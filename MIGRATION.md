# Migration Guide

This document outlines the changes made during the codebase restructuring and how to migrate from the old structure to the new one.

## Major Changes

### 1. Project Structure

**Old Structure:**
```
build_code/src/
├── serve.py
├── voice_to_text.py
├── models.py
├── data_body.py
├── helpers.py
├── config.py
└── runpod_handler.py
```

**New Structure:**
```
build_code/src/
├── main.py                    # Renamed from serve.py
├── config/                    # Configuration module
│   └── settings.py
├── schemas/                   # Request/response models
│   ├── requests.py
│   └── responses.py
├── services/                  # Business logic
│   ├── stt_service.py
│   ├── diarization_service.py
│   ├── transcription_service.py
│   └── audio_service.py
├── models/                    # Model loading
│   └── whisper_model.py
├── utils/                     # Utilities
│   ├── logger.py
│   ├── audio_processing.py
│   ├── transcription_utils.py
│   └── languages.py
└── handlers/                  # External handlers
    └── runpod_handler.py
```

### 2. API Changes

#### Endpoints

- **Old**: `POST /invocations`
- **New**: `POST /api/v1/transcribe` (new standard endpoint)
- **Legacy**: `POST /invocations` (still supported for backward compatibility)

#### Request Format

**Old:**
```json
{
  "audio_url": "https://example.com/audio.mp3",
  "extra_data": {},
  "dispatcher_endpoint": "https://example.com/api"
}
```

**New:**
```json
{
  "audio_url": "https://example.com/audio.mp3",
  "language": "en",
  "task": "transcribe",
  "enable_diarization": true,
  "num_speakers": 2,
  "translate_to_english": false,
  "extra_data": {},
  "dispatcher_endpoint": "https://example.com/api"
}
```

#### Response Format

**Old:**
```json
{
  "data": {
    "text": "...",
    "diarized_transcript": "...",
    "translation": "...",
    "diarized_translation": "...",
    "duration": 123.45,
    "language": "en",
    "extra_data": {}
  }
}
```

**New:**
```json
{
  "text": "...",
  "diarized_text": "...",
  "translation": "...",
  "diarized_translation": "...",
  "language": "EN",
  "duration": 123.45,
  "segments": [...],
  "diarized_segments": [...],
  "num_speakers": 2,
  "processing_time": 2.5,
  "cost": 0.00175,
  "extra_data": {}
}
```

### 3. Configuration Changes

**Old:**
```python
from config import settings
HF_TOKEN = settings.HF_TOKEN
```

**New:**
```python
from config import get_settings
settings = get_settings()
hf_token = settings.hf_token
```

### 4. Model Loading

**Old:**
```python
from models import whisper_models
model = whisper_models()
```

**New:**
```python
from models import get_whisper_model
model = get_whisper_model()
```

### 5. Service Usage

**Old:**
```python
from voice_to_text import process_audio_request
result = process_audio_request(audio_url, extra_data, dispatcher_endpoint)
```

**New:**
```python
from services.transcription_service import TranscriptionService
from schemas.requests import TranscriptionRequest

service = TranscriptionService()
request = TranscriptionRequest(
    audio_url=audio_url,
    extra_data=extra_data,
    dispatcher_endpoint=dispatcher_endpoint
)
result = service.process(request)
```

## Migration Steps

1. **Update Imports**: Replace old imports with new module paths
2. **Update API Calls**: Use new endpoint `/api/v1/transcribe` or continue using `/invocations`
3. **Update Request Format**: Add new optional parameters for better control
4. **Update Response Handling**: Response structure has changed (no nested `data` key in new format)
5. **Update Configuration**: Use new `get_settings()` function

## Backward Compatibility

The following old features are still supported:

- `POST /invocations` endpoint (legacy)
- Old request format (with default values for new parameters)
- Old response format can be achieved by wrapping in `{"data": ...}`

## Deprecated Files

The following files are deprecated but kept for reference:

- `build_code/src/serve.py` → Use `main.py`
- `build_code/src/voice_to_text.py` → Use `services/transcription_service.py`
- `build_code/src/models.py` → Use `models/whisper_model.py`
- `build_code/src/data_body.py` → Use `schemas/requests.py` and `schemas/responses.py`
- `build_code/src/helpers.py` → Functionality moved to `utils/` modules
- `build_code/src/config.py` → Use `config/settings.py`

These files can be removed once migration is complete.

## New Features

1. **Language Selection**: Explicitly specify language or auto-detect
2. **Diarization Control**: Enable/disable diarization per request
3. **Speaker Count**: Specify number of speakers or auto-detect
4. **Translation Control**: Option to translate to English
5. **Health Checks**: `/health` and `/ping` endpoints
6. **Languages Endpoint**: `GET /api/v1/languages` to list supported languages
7. **Better Error Handling**: Comprehensive error responses
8. **Structured Logging**: Improved logging with proper levels

## Testing Migration

1. Test health endpoint: `GET /health`
2. Test transcription with old format: `POST /invocations`
3. Test transcription with new format: `POST /api/v1/transcribe`
4. Verify response format matches your expectations
5. Test RunPod handler if using serverless deployment
