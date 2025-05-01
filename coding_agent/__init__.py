try:
    # For Python ≥3.8
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback for older Python versions
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("dspy-simba-agent")
except PackageNotFoundError:
    __version__ = "0.1.0"

import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default level, can be overridden by app
