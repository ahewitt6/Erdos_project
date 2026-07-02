"""Small shared utilities."""

import random
import time
from pathlib import Path

import numpy as np
import torch


def set_random_seeds(seed=42):
    """Make NumPy and PyTorch results reproducible where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_directories():
    """Create output folders used by the pipeline."""
    for directory in ["data", "results", "results/plots"]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def cache_is_fresh(path, cache_hours):
    """Return whether a local cache exists and is sufficiently recent."""
    path = Path(path)
    if not path.exists():
        return False
    return time.time() - path.stat().st_mtime < cache_hours * 60 * 60


def retry_request(function, description, retries=2, wait_seconds=20):
    """Retry a network call slowly to avoid hammering Yahoo Finance."""
    last_error = None
    for attempt in range(retries):
        try:
            return function()
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                wait = wait_seconds * (attempt + 1)
                print(f"{description} failed. Waiting {wait} seconds before one retry.")
                time.sleep(wait)
    raise last_error
