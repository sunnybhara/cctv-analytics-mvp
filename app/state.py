"""
Shared Mutable State
====================
Module-level singletons shared across async handlers and sync worker threads.
All modules importing from here get references to the same objects.
"""

import threading
from typing import Dict, Any

# Global processing jobs tracker (read by API endpoints, written by worker threads)
processing_jobs: Dict[str, Dict[str, Any]] = {}

# Batch queue state
_queue_processor_running = False
_queue_lock = threading.Lock()
_active_workers = 0
