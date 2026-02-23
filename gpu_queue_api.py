"""FastAPI control plane that enqueues transcription requests and manages autoscaling signals."""
import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
from typing import Dict, Optional
from urllib.parse import urlsplit
from uuid import uuid4

import redis.asyncio as redis
from redis.exceptions import AuthenticationError, RedisError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "build_code" / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from schemas.requests import TranscriptionRequest  # noqa: E402
from utils.logger import get_logger, setup_logging  # noqa: E402

setup_logging()
logger = get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
QUEUE_NAME = os.getenv("QUEUE_NAME", "transcription_queue")
AUTOSCALER_ENABLED = os.getenv("RUNPOD_AUTOSCALER_ENABLED", "true").lower() == "true"
QUEUE_IDLE_TIMEOUT = int(os.getenv("QUEUE_IDLE_TIMEOUT", 30))
QUEUE_POLL_INTERVAL = int(os.getenv("QUEUE_POLL_INTERVAL", 10))
RUNPOD_API_URL = os.getenv("RUNPOD_API_URL", "https://api.runpod.io/graphql")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "hfwcjpwkokcocwc")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "csadwdckjaciapsckls")

redis_client: Optional[redis.Redis] = None
autoscaler_task: Optional[asyncio.Task] = None
idle_start_time: Optional[float] = None
shutdown_requested = False
pod_running = False


def _redacted_redis_target(redis_url: str) -> str:
    parsed = urlsplit(redis_url)
    host = parsed.hostname or "localhost"
    port = f":{parsed.port}" if parsed.port else ""
    db_path = parsed.path or ""
    return f"{parsed.scheme}://{host}{port}{db_path}"


def _build_redis_client() -> redis.Redis:
    connection_kwargs: Dict[str, str] = {}
    if REDIS_USERNAME:
        connection_kwargs["username"] = REDIS_USERNAME
    if REDIS_PASSWORD:
        connection_kwargs["password"] = REDIS_PASSWORD
    return redis.from_url(REDIS_URL, **connection_kwargs)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global redis_client, autoscaler_task

    logger.info("Connecting to Redis at %s", _redacted_redis_target(REDIS_URL))
    redis_client = _build_redis_client()
    try:
        await redis_client.ping()
    except AuthenticationError as exc:
        redis_client = None
        logger.error(
            "Redis authentication failed. Set REDIS_USERNAME/REDIS_PASSWORD (or USERNAME/PASSWORD) and verify credentials."
        )
        raise RuntimeError("Redis authentication failed") from exc
    except RedisError as exc:
        redis_client = None
        logger.error("Unable to reach Redis at %s: %s", _redacted_redis_target(REDIS_URL), exc)
        raise RuntimeError("Redis connection failed") from exc

    if AUTOSCALER_ENABLED:
        if not RUNPOD_API_KEY or not RUNPOD_POD_ID:
            logger.warning(
                "Autoscaler enabled but RUNPOD_API_KEY or RUNPOD_POD_ID not configured; disabling autoscaler"
            )
        else:
            logger.info(
                "Starting autoscaler loop: idle_timeout=%ss, poll_interval=%ss",
                QUEUE_IDLE_TIMEOUT,
                QUEUE_POLL_INTERVAL,
            )
            autoscaler_task = asyncio.create_task(queue_autoscaler_loop())

    try:
        yield
    finally:
        if autoscaler_task is not None:
            autoscaler_task.cancel()
            try:
                await autoscaler_task
            except asyncio.CancelledError:
                pass
        if redis_client is not None:
            await redis_client.aclose()
        logger.info("Disconnected from Redis")


app = FastAPI(title="GPU Queue API", version="1.0.0", lifespan=lifespan)


class JobSubmissionResponse(BaseModel):
    """Response returned after enqueuing a transcription job."""

    job_id: str = Field(description="Identifier assigned to the queued job")
    status: str = Field(description="Submission status message")


@app.post("/jobs", response_model=JobSubmissionResponse)
async def submit_job(request: TranscriptionRequest) -> JobSubmissionResponse:
    """Accept a transcription request and push it into the Redis queue."""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")

    job_id = str(uuid4())
    request_payload = request.model_dump(mode="json")
    payload: Dict[str, object] = {"job_id": job_id, "request": request_payload}

    try:
        logger.info("Enqueuing job %s", payload)
        await redis_client.lpush(QUEUE_NAME, json.dumps(payload))
        logger.info("Queued job %s", job_id)
        return JobSubmissionResponse(job_id=job_id, status="queued")
    except Exception as exc:
        logger.error("Failed to enqueue job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to enqueue job") from exc


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "autoscaler_enabled": str(AUTOSCALER_ENABLED).lower(),
        "queue_idle_timeout": str(QUEUE_IDLE_TIMEOUT),
    }


@app.get("/queue_size")
async def queue_size() -> Dict[str, int]:
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")

    queue_length = await redis_client.llen(QUEUE_NAME)
    return {"queue_size": queue_length}


async def queue_autoscaler_loop() -> None:
    """Monitor Redis queue depth and trigger RunPod shutdown when idle."""
    global idle_start_time, shutdown_requested, pod_running

    while True:
        try:
            queue_length = await redis_client.llen(QUEUE_NAME) if redis_client else 0
        except Exception as exc:
            logger.error("Failed to read queue length: %s", exc, exc_info=True)
            await asyncio.sleep(QUEUE_POLL_INTERVAL)
            continue

        if queue_length == 0:
            if idle_start_time is None:
                idle_start_time = time.time()
                logger.info("Queue empty; starting idle timer")
            elif (time.time() - idle_start_time) >= QUEUE_IDLE_TIMEOUT:
                if not shutdown_requested:
                    await trigger_runpod_shutdown()
                    shutdown_requested = True
        else:
            idle_start_time = None
            shutdown_requested = False
            if not pod_running:
                await trigger_runpod_start()

        await asyncio.sleep(QUEUE_POLL_INTERVAL)


# async def trigger_runpod_shutdown() -> None:
#     """Invoke RunPod API to stop the current pod."""
#     global pod_running

#     if not pod_running:
#         logger.info("Shutdown requested but pod is already stopped")
#         return

#     logger.info("Queue idle threshold reached; initiating RunPod shutdown request")

#     mutation = (
#         "mutation podStop($podId: String!) {\n"
#         "  podStop(podId: $podId) {\n"
#         "    id\n"
#         "    status\n"
#         "    message\n"
#         "  }\n"
#         "}\n"
#     )

#     headers = {
#         "Authorization": f"Bearer {RUNPOD_API_KEY}",
#         "Content-Type": "application/json",
#     }

#     payload = {
#         "query": mutation,
#         "variables": {"podId": RUNPOD_POD_ID},
#     }

#     def _post():
#         return requests.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=30)

#     try:
#         response = await asyncio.to_thread(_post)
#         response.raise_for_status()
#         body = response.json()
#         logger.info("RunPod shutdown response: %s", body)
#         pod_running = False
#     except Exception as exc:
#         logger.error("Failed to stop RunPod pod %s: %s", RUNPOD_POD_ID, exc, exc_info=True)


# async def trigger_runpod_start() -> None:
#     """Invoke RunPod API to start/resume the pod when work is pending."""
#     global pod_running

#     if pod_running:
#         logger.debug("Pod already running; skip start request")
#         return

#     if not RUNPOD_API_KEY or not RUNPOD_POD_ID:
#         logger.warning("Cannot start RunPod pod; RUNPOD_API_KEY or RUNPOD_POD_ID missing")
#         return

#     logger.info("Queue has pending jobs; initiating RunPod start request")

#     mutation = (
#         "mutation podResume($podId: String!) {\n"
#         "  podResume(podId: $podId) {\n"
#         "    id\n"
#         "    status\n"
#         "    message\n"
#         "  }\n"
#         "}\n"
#     )

#     headers = {
#         "Authorization": f"Bearer {RUNPOD_API_KEY}",
#         "Content-Type": "application/json",
#     }

#     payload = {
#         "query": mutation,
#         "variables": {"podId": RUNPOD_POD_ID},
#     }

#     def _post():
#         return requests.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=30)

#     try:
#         response = await asyncio.to_thread(_post)
#         response.raise_for_status()
#         body = response.json()
#         logger.info("RunPod start response: %s", body)
#         pod_running = True
#     except Exception as exc:
#         logger.error("Failed to start RunPod pod %s: %s", RUNPOD_POD_ID, exc, exc_info=True)

async def trigger_runpod_shutdown() -> None:
    """Simulate RunPod shutdown for demo purposes."""
    global pod_running

    if not pod_running:
        logger.info("Shutdown requested but pod is already stopped")
        return

    logger.info("Shutdown requested; simulating RunPod shutdown")
    await asyncio.sleep(5)
    pod_running = False
    logger.info("Pod is now stopped (simulated)")

async def trigger_runpod_start() -> None:
    """Simulate RunPod start for demo purposes."""
    global pod_running

    if pod_running:
        logger.debug("Pod already running; skip start request")
        return

    logger.info("Start requested; simulating RunPod start")
    await asyncio.sleep(5)
    pod_running = True
    logger.info("Pod is now running (simulated)")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gpu_queue_api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8002)),
        reload=True,
        log_level="info",
    )
