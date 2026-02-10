"""
Video Analytics MVP - Railway Backend
======================================
Thin entry point. All logic lives in the app/ package.

Deploy to Railway:
    railway login
    railway init
    railway up
"""

import os

# Import app and key objects for backward compatibility
# Tests import: from main import app, database, metadata, events, venues
from app import app  # noqa: F401
from app.database import database, metadata, events, venues  # noqa: F401

# Also re-export commonly used helpers for any external scripts
from app.video.helpers import generate_pseudo_id, get_zone  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
