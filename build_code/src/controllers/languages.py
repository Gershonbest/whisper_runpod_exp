"""Languages routes."""
from fastapi import APIRouter

from utils.languages import SUPPORTED_LANGUAGES

router = APIRouter(prefix="/api/v1", tags=["languages"])


@router.get("/languages")
async def get_languages():
    """Get list of supported languages."""
    return {
        "languages": SUPPORTED_LANGUAGES,
        "count": len(SUPPORTED_LANGUAGES)
    }
