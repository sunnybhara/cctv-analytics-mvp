"""
Application Configuration
=========================
Central config loaded from environment variables.
"""

import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./analytics.db")

# Handle Railway's postgres:// vs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# CORS origins (comma-separated in env, or * for dev)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

# Upload limits
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))

# Authentication
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

# Allowed video URL domains
ALLOWED_VIDEO_DOMAINS = ["youtube.com", "youtu.be", "www.youtube.com"]
