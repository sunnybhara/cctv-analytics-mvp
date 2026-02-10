"""
Health Check Router
===================
Simple health check endpoint for Railway / load balancers.
"""

from datetime import datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    """Health check for Railway."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
