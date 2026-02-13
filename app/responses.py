"""
Standardized Response Format
=============================
Wrappers for consistent API responses across all endpoints.
"""

from datetime import datetime, UTC

from fastapi.responses import JSONResponse


def success_response(data, status_code=200, pagination=None):
    """Wrap data in standard success envelope."""
    body = {
        "status": "success",
        "data": data,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if pagination is not None:
        body["pagination"] = pagination
    return JSONResponse(content=body, status_code=status_code)


def error_response(message, status_code=400):
    """Wrap error in standard error envelope."""
    return JSONResponse(
        content={
            "status": "error",
            "message": message,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        status_code=status_code,
    )
