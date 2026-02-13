"""
Test configuration — ensure sipp-pipeline root is on sys.path
so that `config.settings` and `pipeline.*` imports resolve.
"""

import os
import sys

# Add the sipp-pipeline directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
