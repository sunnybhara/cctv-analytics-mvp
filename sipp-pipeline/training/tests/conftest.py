"""Test configuration — force-load training/config.py over sipp-pipeline/config/ package."""

import importlib.util
import os
import sys

training_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, training_dir)

# Force-load training/config.py as the 'config' module.
# Without this, Python finds sipp-pipeline/config/ (a package) instead of
# training/config.py (a module) because packages take priority.
spec = importlib.util.spec_from_file_location(
    "config", os.path.join(training_dir, "config.py")
)
config_mod = importlib.util.module_from_spec(spec)
sys.modules["config"] = config_mod
spec.loader.exec_module(config_mod)
